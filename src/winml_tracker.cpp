#pragma comment(lib, "windowsapp") 

#define NOMINMAX
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNING 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <vcruntime.h>
#include <windows.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Imaging.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/InteractiveMarker.h>

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

void ProcessImage(const sensor_msgs::ImageConstPtr& image) {
    ROS_INFO_STREAM("Received image: " << image->header.seq);

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
	// And set the image to 32-bit floating point values for tensorization.
	cv::Mat image_32_bit;
	if (cv_ptr->image.size().width != 416 || cv_ptr->image.size().height != 416)
	{
		cv::Mat image_resized;
		cv::Size size(416, 416);
		cv::resize(cv_ptr->image, image_resized, size, 0, 0, cv::INTER_CUBIC);
		image_resized.convertTo(image_32_bit, CV_32FC1);
	}
	else
	{
		cv_ptr->image.convertTo(image_32_bit, CV_32FC1);
	}

	// Setup the model binding
    LearningModelBinding binding(session);
	vector<int64_t> grid_shape({ 1, CHANNEL_COUNT, ROW_COUNT, COL_COUNT });
	binding.Bind(L"grid", TensorFloat::Create(grid_shape));

	// Create a Tensor from the CV Mat and bind it to the session
	std::vector<float> image_data;
	image_data.assign(1*3*416*416, (const float &)image_32_bit.data);
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
	for (std::vector<YoloBox>::iterator it = boxes.begin(); it != boxes.end(); ++it)
	{
		if (it->label == "person")
		{
			ROS_INFO("Person detected!");

			visualization_msgs::InteractiveMarker msg;
			msg.pose.position.x = it->x;
			msg.pose.position.y = it->y;
			msg.pose.orientation.z = 1.0;
			msg.name = "Person";	
			msg.description = "WINML Object Tracker";		

			detect_pub.publish(msg);

			// Only send one for now
			return;
		}
	}
}

int main(int argc, char **argv)
{
  init_apartment();

  ros::init(argc, argv, "winml_tracker");
  ros::NodeHandle nh;
  
  detect_pub = nh.advertise<visualization_msgs::InteractiveMarker >("tracked_objects", 1);

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/cv_camera/image_raw", 1, ProcessImage);

  // Load the ML model
  hstring modelPath = hstring(wstring_to_utf8().from_bytes("C:\\Users\\Stuart\\Downloads\\onnxzoo_winmlperf_tiny_yolov2.onnx"));
  model = LearningModel::LoadFromFilePath(modelPath);

  // Create a WinML session
  session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Cpu));

  ros::spin();
   
  nh.shutdown();
  return 0;
}