#pragma comment(lib, "windowsapp")

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
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
#include "winml_tracker/pose_parser.h"
#include "winml_tracker/winml_tracker.h"

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

LearningModel model = nullptr;
LearningModelSession session = nullptr;

ros::Publisher detect_pub;
image_transport::Publisher image_pub;
image_transport::Publisher debug_image_pub;

WinMLTracker_Type TrackerType = WinMLTracker_Yolo;
WinMLTracker_ImageProcessing ImageProcessingType = WinMLTracker_Scale;
std::vector<cv::Point3d> modelBounds;


void processYoloOutput(std::vector<float> grids, cv::Mat& image_resized)
{
    auto boxes = yolo::YoloResultsParser::GetRecognizedObjects(grids, 0.3f);

    // If we found a person, send a message
    int count = 0;
    std::vector<visualization_msgs::Marker> markers;
    for (std::vector<yolo::YoloBox>::iterator it = boxes.begin(); it != boxes.end(); ++it)
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
}

void processPoseOutput(std::vector<float> grids, cv::Mat& image_resized)
{
    std::vector<int> cuboid_edges_v1({0,1,2,3,4,5,6,7,1,0,2,3});
    std::vector<int> cuboid_edges_v2({1,2,3,0,5,6,7,4,5,4,6,7});

    auto pose = pose::PoseResultsParser::GetRecognizedObjects(grids);

	if (modelBounds.size() > 0)
	{
		// Borrowing from https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
		double focal_length = 416; // Approximate focal length.
		cv::Point2d center = cv::Point2d(416 / 2, 416 / 2);
		cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
		cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

		cv::Mat rotation_vector;
		cv::Mat translation_vector;

		// Solve for pose
		cv::solvePnP(modelBounds, pose.bounds, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

		std::vector<cv::Point3f> AxisPoints3D;
		AxisPoints3D.push_back(cv::Point3f(0, 0, 0));
		AxisPoints3D.push_back(cv::Point3f(5, 0, 0));
		AxisPoints3D.push_back(cv::Point3f(0, 5, 0));
		AxisPoints3D.push_back(cv::Point3f(0, 0, 5));

		std::vector<cv::Point2f> AxisPoints2D;
		cv::projectPoints(AxisPoints3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, AxisPoints2D);

		cv::line(image_resized, AxisPoints2D[0], AxisPoints2D[1], cv::Scalar(255, 0, 0), 2);
		cv::line(image_resized, AxisPoints2D[0], AxisPoints2D[2], cv::Scalar(0, 255, 0), 2);
		cv::line(image_resized, AxisPoints2D[0], AxisPoints2D[3], cv::Scalar(0, 0, 255), 2);

		cv::Mat_<double> rosQuat = cv::Mat_<double>::eye(4, 1);
		cv::Mat_<double> rvec(3, 3);
		cv::Rodrigues(rotation_vector, rvec);

		double w = rvec(0, 0) + rvec(1, 1) + rvec(2, 2) + 1;
		if (w > 0.0)
		{
			w = sqrt(w);
			rosQuat(0, 0) = (rvec(2, 1) - rvec(1, 2)) / (w * 2.0);
			rosQuat(1, 0) = (rvec(0, 2) - rvec(2, 0)) / (w * 2.0);
			rosQuat(2, 0) = (rvec(1, 0) - rvec(0, 1)) / (w * 2.0);
			rosQuat(3, 0) = w / 2.0;

			int count = 0;
			std::vector<visualization_msgs::Marker> markers;
			visualization_msgs::Marker marker;
			marker.header.frame_id = "base_link";
			marker.header.stamp = ros::Time();
			marker.ns = "winml";
			marker.id = count++;
			marker.type = visualization_msgs::Marker::MESH_RESOURCE;
			marker.action = visualization_msgs::Marker::ADD;
			marker.mesh_resource = "package://winml_tracker/testdata/shoe.dae";

			marker.pose.position.x = translation_vector.at<float>(0);
			marker.pose.position.y = translation_vector.at<float>(1);
			marker.pose.position.z = translation_vector.at<float>(2);
			marker.pose.orientation.x = rosQuat(0, 0);
			marker.pose.orientation.y = rosQuat(1, 0);
			marker.pose.orientation.z = rosQuat(2, 0);
			marker.pose.orientation.w = rosQuat(3, 0);

			marker.scale.x = 1.0;
			marker.scale.y = 1.0;
			marker.scale.z = 1.0;
			marker.color.a = 1.0;
			marker.color.r = 0.0;
			marker.color.g = 0.0;
			marker.color.b = 1.0;

			markers.push_back(marker);
			detect_pub.publish(markers);
		}
	}

    cv::Scalar color(255, 255, 0);

    for (int i = 0; i < cuboid_edges_v2.size(); i++)
    {
        cv::Point2i pt1 = pose.bounds[cuboid_edges_v1[i]];
        cv::Point2i pt2 = pose.bounds[cuboid_edges_v2[i]];
        cv::line(image_resized, pt1, pt2, color, 5);
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_resized).toImageMsg();
    image_pub.publish(msg);
}


void ProcessImage(const sensor_msgs::ImageConstPtr& image) 
{
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

	cv::Size size(416, 416);
	cv::Mat rgb_image;
	cv::Mat image_resized;
	// TODO: If the image is not the right dimensions, center crop or resize
	cv::Size s = cv_ptr->image.size();
	if (ImageProcessingType == WinMLTracker_Crop && 
        s.width > 416 && s.height > 416)
	{
		// crop
		cv::Rect ROI((s.width - 416) / 2, (s.height - 416) / 2, 416, 416);
		image_resized = cv_ptr->image(ROI);
	}
	else
	{
		cv::resize(cv_ptr->image, image_resized, size, 0, 0, cv::INTER_CUBIC);
	}
	/*
	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_resized).toImageMsg();
	while (ros::ok())
	{
		debug_image_pub.publish(msg);
		ros::spinOnce();
	}
	*/

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

    int channelCount;
    int rowCount;
    int colCount;
	wstring outName;
	wstring inName;

    switch (TrackerType)
    {
        case WinMLTracker_Yolo:
            channelCount = yolo::CHANNEL_COUNT;
            rowCount = yolo::ROW_COUNT;
            colCount = yolo::COL_COUNT;
			outName = L"grid";
			inName = L"images";
        break;

        case WinMLTracker_Pose:
            channelCount = pose::CHANNEL_COUNT;
            rowCount = pose::ROW_COUNT;
            colCount = pose::COL_COUNT;
			outName = L"218";
			inName = L"0";
			break;

        default:
        return;
    }


    // Setup the model binding
    LearningModelBinding binding(session);
    vector<int64_t> grid_shape({ 1, channelCount, rowCount, colCount });
	binding.Bind(outName.c_str(), TensorFloat::Create(grid_shape));

    // Create a Tensor from the CV Mat and bind it to the session
    std::vector<float> image_data(1 * 3 * 416 * 416);
    memcpy(&image_data[0], (float *)channels[0].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[416 * 416], (float *)channels[1].data, 416 * 416 * sizeof(float));
    memcpy(&image_data[2 * 416 * 416], (float *)channels[2].data, 416 * 416 * sizeof(float));
    TensorFloat image_tensor = TensorFloat::CreateFromArray({ 1, 3, 416, 416 }, image_data);
    binding.Bind(inName.c_str(), image_tensor);
	/*
	std::cout << "Model input\n";

	for (auto v = image_data.begin(); v != image_data.end(); ++v)
	{
		std::cout << *v << ' ';
	}
	*/

    // Call WinML    
    auto results = session.Evaluate(binding, L"RunId");
    if (!results.Succeeded())
    {
        ROS_ERROR("WINML: Evaluation of object tracker failed!");
        return;
    }

    // Convert the results to a vector and parse the bounding boxes
    auto grid_result = results.Outputs().Lookup(outName.c_str()).as<TensorFloat>().GetAsVectorView();
    std::vector<float> grids(grid_result.Size());
    winrt::array_view<float> grid_view(grids);
    grid_result.GetMany(0, grid_view);

    switch (TrackerType)
    {
        case WinMLTracker_Yolo:
        processYoloOutput(grids, image_resized);
        break;

        case WinMLTracker_Pose:
        processPoseOutput(grids, image_resized);
        break;
    }

    return;
}

int WinMLTracker_Init(ros::NodeHandle& nh)
{
    detect_pub = nh.advertise<visualization_msgs::MarkerArray>("tracked_objects", 1);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/cv_camera/image_raw", 1, ProcessImage);
    image_pub = it.advertise("tracked_objects/image", 1);
	debug_image_pub = it.advertise("debug/image", 1);

    return 0;
}

int WinMLTracker_Startup(ros::NodeHandle& nh)
{
    // Parameters.
    std::string onnxModelPath;
    if (nh.getParam("onnx_model_path", onnxModelPath) ||
		onnxModelPath.empty())
    {
        ROS_ERROR("onnx_model_path parameter has not been set.");
        nh.shutdown();
        return 0;
    }

    std::string imageProcessingType;
    if (nh.getParam("image_processing", imageProcessingType))
    {
        if (imageProcessingType == "crop")
        {
            ImageProcessingType = WinMLTracker_Crop;
        }
        else if (imageProcessingType == "scale")
        {
            ImageProcessingType = WinMLTracker_Scale;
        }
        else
        {
            // default;
        }
    }

	std::string trackerType;
	if (nh.getParam("tracker_type", trackerType))
	{
		if (trackerType == "yolo")
		{
			TrackerType = WinMLTracker_Yolo;
		}
		else if (trackerType == "pose")
		{
			TrackerType = WinMLTracker_Pose;
		}
		else
		{
			// default;
		}
	}
	if (TrackerType == WinMLTracker_Pose)
	{
		std::vector<float> points;
		if (nh.getParam("model_bounds", points))
		{
			if (points.size() < 9 * 3)
			{
				ROS_ERROR("Model Bounds needs 9 3D floating points.");
				nh.shutdown();
				return 0;
			}

			for (int p = 0; p < points.size() / 3; p += 3)
			{
				modelBounds.push_back(cv::Point3d(points[0], points[1], points[2]));
			}
		}
		else
		{
			ROS_ERROR("Model Bounds needs to be specified for Pose processing.");
			nh.shutdown();
			return 0;
		}
	}

    // Load the ML model
    hstring modelPath = hstring(wstring_to_utf8().from_bytes(onnxModelPath));
    model = LearningModel::LoadFromFilePath(modelPath);

    // Create a WinML session
    session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Cpu));

    return 0;
}

int WinMLTracker_Shutdown(ros::NodeHandle& nh)
{
    nh.shutdown();

    return 0;
}
