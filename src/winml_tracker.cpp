#pragma comment(lib, "windowsapp") 

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNING 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>


// Include ROS files before Windows, as there are overlapping symbols
#include <vcruntime.h>
#include <windows.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Imaging.h>

#include "winml_tracker/yolo_box.h"

#include <string>
#include <codecvt>
#include <fstream>
#include <sstream>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Media;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage;
using namespace std;
using namespace yolo;

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;

LearningModel model = nullptr;
LearningModelSession session = nullptr;

ros::Publisher detect_pub;
image_transport::Publisher image_pub;

void ProcessImage(const sensor_msgs::ImageConstPtr& image) {
    //ROS_INFO_STREAM("Received image: " << image->header.seq);

    // Convert back to an OpenCV Image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // TODO: If the image is not the right dimensions, center crop or resize
    cv::Mat rgb_image;
    cv::Mat image_resized;
    cv::Size size(416, 416);
    cv::resize(cv_ptr->image, image_resized, size, 0, 0, cv::INTER_CUBIC);

    // Convert to RGB
    cv::cvtColor(image_resized, rgb_image, cv::COLOR_BGR2RGB);

    // Set the image to 32-bit floating point values for tensorization.
    cv::Mat image_32_bit;
    rgb_image.convertTo(image_32_bit, CV_32F);

    // Extract color channels from interleaved data
    cv::Mat channels[3];
    cv::split(image_32_bit, channels);

    // Setup the model binding
    LearningModelBinding binding(session);
    vector<int64_t> grid_shape({ 1, CHANNEL_COUNT, ROW_COUNT, COL_COUNT });
    binding.Bind(L"grid", TensorFloat::Create(grid_shape));

    // Create a Tensor from the CV Mat and bind it to the session
    std::vector<float> image_data(1 * 3 * 416 * 416);
    memcpy(&image_data[0], (float *)channels[0].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[416 * 416], (float *)channels[1].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[2 * 416 * 416], (float *)channels[2].data, 416 * 416 * sizeof(float));
    TensorFloat image_tensor = TensorFloat::CreateFromArray({ 1, 3, 416, 416 }, image_data);
    binding.Bind(L"image", image_tensor);

    // Call WinML    
    auto results = session.Evaluate(binding, L"RunId");
    if (!results.Succeeded())
    {
        ROS_ERROR("WINML: Evaluation of object tracker failed!");
        return;
    }

    // Convert the results to a vector and parse the bounding boxes
    auto grid_result = results.Outputs().Lookup(L"grid").as<TensorFloat>().GetAsVectorView();
    std::vector<float> grids(grid_result.Size());
    winrt::array_view<float> grid_view(grids);
    grid_result.GetMany(0, grid_view);
    auto boxes = YoloResultsParser::GetRecognizedObjects(grids, 0.3f);

    // If we found a person, send a message
    int count = 0;
    std::vector<visualization_msgs::Marker> markers;
    for (std::vector<YoloBox>::iterator it = boxes.begin(); it != boxes.end(); ++it)
    {
        if (it->label == "person" && it->confidence >= 0.5f)
        {
            ROS_INFO("Person detected!");

            visualization_msgs::Marker marker;
            marker.header.frame_id = "base_link";
            marker.header.stamp = ros::Time();
            marker.ns = "winml";
            marker.id = count++;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = it->x + it->width / 2;
            marker.pose.position.y = it->y + it->height / 2;
            marker.pose.position.z = 0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 1;
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;
            marker.color.a = 1.0;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;

            markers.push_back(marker);

            // Draw a bounding box on the CV image
            cv::Scalar color(255, 255, 0);
            cv::Rect box;
            box.x = std::max<int>((int)it->x, 0);
            box.y = std::max<int>((int)it->y, 0);
            box.height = std::min<int>(image_resized.rows - box.y, (int)it->height);
            box.width = std::min<int>(image_resized.cols - box.x, (int)it->width);
            cv::rectangle(image_resized, box, color, 2, 8, 0);
        }
    }

    if (count > 0)
    {
        detect_pub.publish(markers);
    }

    // Always publish the resized image
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_resized).toImageMsg();
    image_pub.publish(msg);

    // Only send one for now
    return;
}

int main(int argc, char **argv)
{
    init_apartment();

    ros::init(argc, argv, "winml_tracker");
    ros::NodeHandle nh;

    detect_pub = nh.advertise<visualization_msgs::MarkerArray>("tracked_objects", 1);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/cv_camera/image_raw", 1, ProcessImage);
    image_pub = it.advertise("tracked_objects/image", 1);

    // Load the ML model
    hstring modelPath = hstring(wstring_to_utf8().from_bytes("C:\\Users\\Stuart\\Downloads\\onnxzoo_winmlperf_tiny_yolov2.onnx"));
    model = LearningModel::LoadFromFilePath(modelPath);

    // Create a WinML session
    session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Cpu));

    ros::spin();

    nh.shutdown();
    return 0;
}