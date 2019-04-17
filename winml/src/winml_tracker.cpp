#pragma comment(lib, "windowsapp")

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/LinearMath/Quaternion.h>
#include <opencv2/opencv.hpp>


// Include ROS files before Windows, as there are overlapping symbols
#include <vcruntime.h>
#include <windows.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Imaging.h>

#include "winml_tracker/winml_tracker.h"
#include "winml_tracker/yolo_box.h"
#include "winml_tracker/pose_parser.h"

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

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;

WinMLProcessor::WinMLProcessor()
: _process(ImageProcessing::Scale)
, _confidence(0.70f)
, _debug(false)
{

}

bool WinMLProcessor::init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate)
{
    std::string imageProcessingType;
    if (nhPrivate.getParam("image_processing", imageProcessingType))
    {
        if (imageProcessingType == "crop")
        {
            _process = Crop;
        }
        else if (imageProcessingType == "scale")
        {
            _process = Scale;
        }
        else
        {
            // default;
        }
    }

    nhPrivate.param<bool>("winml_fake", _fake, false);

    nhPrivate.getParam("link_name", _linkName);

    if (!nhPrivate.getParam("onnx_model_path", _onnxModel) ||
        _onnxModel.empty())
    {
        ROS_ERROR("onnx_model_path parameter has not been set.");
        return false;
    }

    if (nhPrivate.getParam("calibration", _calibration))
    {
        try
        {
            cv::FileStorage fs(_calibration, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
            fs["camera_matrix"] >> _camera_matrix;
            fs["distortion_coefficients"] >> _dist_coeffs;
        }
        catch (std::exception &e)
        {
            ROS_ERROR("Failed to read the calibration file, continuing without calibration.\n%s", e.what());
            // no calibration for you.
            _calibration = "";
        }
    }

    float conf;
    if (nhPrivate.getParam("confidence", conf))
    {
        _confidence = conf;
    }

    bool d;
    if (nhPrivate.getParam("debug", d))
    {
        _debug = d;
    }

    

    std::string imageTopic;
    if (!nhPrivate.getParam("image_topic", imageTopic) ||
        imageTopic.empty())
    {
        imageTopic = "/cv_camera/image_raw";
    }

    _detect_pub = nh.advertise<visualization_msgs::MarkerArray>("tracked_objects", 1);

    image_transport::ImageTransport it(nh);
    _cameraSub = it.subscribe(imageTopic.c_str(), 1, &WinMLProcessor::ProcessImage, this);
    _image_pub = it.advertise("tracked_objects/image", 1);
    _debug_image_pub = it.advertise("debug/image", 1);

    // Load the ML model
    hstring modelPath = hstring(wstring_to_utf8().from_bytes(_onnxModel));
    _model = LearningModel::LoadFromFilePath(modelPath);

    // Create a WinML session
    _session = LearningModelSession(_model, LearningModelDevice(LearningModelDeviceKind::Cpu));

    return true;
}

void WinMLProcessor::ProcessImage(const sensor_msgs::ImageConstPtr& image) 
{
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

    cv::Size mlSize(416, 416);
    cv::Mat rgb_image;
    cv::Mat image_resized;
    cv::Size s = _original_image_size = cv_ptr->image.size();
    auto minSize = s.width < s.height ? s.width : s.height;
    ROS_INFO_ONCE("image size (%d, %d)", s.width, s.height);

    if (_calibration.empty())
    {
        // Borrowing from https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        double image_width = s.width;
        double image_height = s.height;
        cv::Point2d center = cv::Point2d(image_width / 2, image_height / 2);
        _camera_matrix = (cv::Mat_<double>(3, 3) << image_width, 0, center.x, 0, image_height, center.y, 0, 0, 1);
        _dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
    }
    else
    {
        _dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
    }

    if (_process == Crop && 
        s.width > 416 && s.height > 416)
    {
        // crop
        cv::Rect ROI((s.width - 416) / 2, (s.height - 416) / 2, 416, 416);
        image_resized = cv_ptr->image(ROI);
    }
    else
    {
        // now extract the center
        cv::Rect ROI((s.width - minSize) / 2, (s.height - minSize) / 2, minSize, minSize);
        image_resized = cv_ptr->image(ROI);

        // We want to extract a correct apsect ratio from the center of the image
        // but scale the whole frame so that there are no borders.

        // First downsample
        cv::Size downsampleSize;
        downsampleSize.width = 416;
        downsampleSize.height = 416;

        cv::resize(image_resized, image_resized, downsampleSize, 0, 0, cv::INTER_AREA);
    }

    // Convert to RGB
    cv::cvtColor(image_resized, rgb_image, cv::COLOR_BGR2RGB);

    // Set the image to 32-bit floating point values for tensorization.
    cv::Mat image_32_bit;
    //rgb_image.convertTo(image_32_bit, CV_32F);
    rgb_image.convertTo(image_32_bit, CV_32F);

    cv::normalize(image_32_bit, image_32_bit, 0.0f, 1.0f, cv::NORM_MINMAX);

    // Extract color channels from interleaved data
    cv::Mat channels[3];
    cv::split(image_32_bit, channels);

    // Setup the model binding
    LearningModelBinding binding(_session);
    vector<int64_t> grid_shape({ 1, _channelCount, _rowCount, _colCount });
    binding.Bind(_outName, TensorFloat::Create(grid_shape));

    // Create a Tensor from the CV Mat and bind it to the session
    std::vector<float> image_data(1 * 3 * 416 * 416);
    memcpy(&image_data[0], (float *)channels[0].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[416 * 416], (float *)channels[1].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[2 * 416 * 416], (float *)channels[2].data, 416 * 416 * sizeof(float));
    TensorFloat image_tensor = TensorFloat::CreateFromArray({ 1, 3, 416, 416 }, image_data);
    binding.Bind(_inName, image_tensor);

    if (_fake)
    {
        std::vector<float> grids;
        ProcessOutput(grids, image_resized);
    }
    else
    {
        // Call WinML
        auto results = _session.Evaluate(binding, L"RunId");
        if (!results.Succeeded())
        {
            ROS_ERROR("WINML: Evaluation of object tracker failed!");
            return;
        }

        // Convert the results to a vector and parse the bounding boxes
        auto grid_result = results.Outputs().Lookup(_outName).as<TensorFloat>().GetAsVectorView();
        std::vector<float> grids(grid_result.Size());
        winrt::array_view<float> grid_view(grids);
        grid_result.GetMany(0, grid_view);

        ProcessOutput(grids, image_resized);
    }

    return;
}

bool WinMLTracker::init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate)
{
    _nh = nh;
    _nhPrivate = nhPrivate;

    // Parameters.
    std::string trackerType;
    if (nhPrivate.getParam("tracker_type", trackerType))
    {
        if (trackerType == "yolo")
        {
            _processor = std::make_shared<yolo::YoloProcessor>();
        }
        else if (trackerType == "pose")
        {
            _processor = std::make_shared<pose::PoseProcessor>();
        }
    }

    if (_processor == nullptr)
    {
        _processor = std::make_shared<yolo::YoloProcessor>();
    }

    return _processor->init(_nh, nhPrivate);
}

bool WinMLTracker::shutdown()
{
    _nh.shutdown();
    _nhPrivate.shutdown();

    return true;
}
