#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include "winml_tracker/winml_tracker.h"
#include "winml_tracker/pose_parser.h"

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(10)
#include <Eigen/Eigen>

#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <codecvt>
#include <fstream>
#include <sstream>

const int IMAGE_WIDTH = 416;
const int IMAGE_HEIGHT = 416;

const int ROW_COUNT = 13;
const int COL_COUNT = 13;
const int CHANNEL_COUNT = 20;
const int CLASS_COUNT = 20;

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Media;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage;
using namespace std;
using namespace std;
using namespace pose;

bool g_init = false;
std::vector<float> PoseProcessor::_gridX;
std::vector<float> PoseProcessor::_gridY;

bool PoseProcessor::init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate)
{
	initPoseTables();

	WinMLProcessor::init(nh, nhPrivate);
	_channelCount = CHANNEL_COUNT;
	_rowCount = ROW_COUNT;
	_colCount = COL_COUNT;
	_outName = L"218";
	_inName = L"0";
	std::vector<float> points;
	if (nhPrivate.getParam("model_bounds", points))
	{
		if (points.size() < 9 * 3)
		{
			ROS_ERROR("Model Bounds needs 9 3D floating points.");
			return false;
		}

		for (int p = 0; p < points.size(); p += 3)
		{
			modelBounds.push_back(cv::Point3d(points[p], points[p + 1], points[p + 2]));
		}

		return true;
	}
	else
	{
		ROS_ERROR("Model Bounds needs to be specified for Pose processing.");
		return false;
	}
}


void PoseProcessor::initPoseTables()
{
	if (g_init)
	{
		return;
	}
	else
	{
		g_init = true;

		int xCount = 0;
		int yCount = 0;
		float yVal = 0.0f;

		for (int y = 0; y < ROW_COUNT; y++)
		{
			for (int x = 0; x <= COL_COUNT; x++) // confirm <= 
			{
				_gridX.push_back((float)xCount);
				_gridY.push_back(yVal);

				if (yCount++ == COL_COUNT - 1) // confirm col - 1
				{
					yVal += 1.0;
					yCount = 0;
				}

				if (xCount == COL_COUNT - 1)
				{
					xCount = 0;
				}
				else
				{
					xCount++;
				}
			}
		}
	}
}

std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b)
{
	std::vector<float> ret;
	std::vector<float>::const_iterator aptr = a.begin();
	std::vector<float>::const_iterator bptr = b.begin();
	for (; 
		aptr < a.end() && bptr < b.end(); 
		aptr++, bptr++)
	{
		ret.push_back(*aptr + *bptr);
	}

	return ret;
}

void PoseProcessor::ProcessOutput(std::vector<float> output, cv::Mat& image)
{
    std::vector<int> cuboid_edges_v1({0,1,2,3,4,5,6,7,1,0,2,3});
    std::vector<int> cuboid_edges_v2({1,2,3,0,5,6,7,4,5,4,6,7});

    auto pose = GetRecognizedObjects(output);

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

		cv::line(image, AxisPoints2D[0], AxisPoints2D[1], cv::Scalar(255, 0, 0), 2);
		cv::line(image, AxisPoints2D[0], AxisPoints2D[2], cv::Scalar(0, 255, 0), 2);
		cv::line(image, AxisPoints2D[0], AxisPoints2D[3], cv::Scalar(0, 0, 255), 2);

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
			_detect_pub.publish(markers);
		}
	}

    cv::Scalar color(255, 255, 0);

    for (int i = 0; i < cuboid_edges_v2.size(); i++)
    {
        cv::Point2i pt1 = pose.bounds[cuboid_edges_v1[i]];
        cv::Point2i pt2 = pose.bounds[cuboid_edges_v2[i]];
        cv::line(image, pt1, pt2, color, 5);
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    _image_pub.publish(msg);	
}



Pose PoseProcessor::GetRecognizedObjects(std::vector<float> modelOutputs)
{
	initPoseTables();

//	outputC = outputB.view(1, 19 + num_classes, h*w)
//		print(outputC)
//		output = outputC.transpose(0, 1).contiguous().view(19 + num_classes, h*w)
//		grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h*w)
//		grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h*w)
//		xs0 = torch.sigmoid(output[0]) + grid_x
//		ys0 = torch.sigmoid(output[1]) + grid_y
//		xs1 = output[2] + grid_x
//		ys1 = output[3] + grid_y
	/*
	for (auto v = modelOutputs.begin(); v != modelOutputs.end(); ++v)
	{
		std::cout << *v << ' ';
	}
	*/

	std::vector<std::vector<float>> output;
	for (int c = 0; c < CLASS_COUNT; c++)
	{
		std::vector<float> chanVec;
		for (int vec = 0; vec < ROW_COUNT * COL_COUNT; vec++)
		{
			chanVec.push_back(modelOutputs[GetOffset(vec, c)]);
		}

		output.push_back(chanVec);
	}

	auto xs0 = Sigmoid(output[0]) + _gridX;
	auto ys0 = Sigmoid(output[1]) + _gridY;
	auto xs1 = output[2] + _gridX;
	auto ys1 = output[3] + _gridY;
	auto xs2 = output[4] + _gridX;
	auto ys2 = output[5] + _gridY;
	auto xs3 = output[6] + _gridX;
	auto ys3 = output[7] + _gridY;
	auto xs4 = output[8] + _gridX;
	auto ys4 = output[9] + _gridY;
	auto xs5 = output[10] + _gridX;
	auto ys5 = output[11] + _gridY;
	auto xs6 = output[12] + _gridX;
	auto ys6 = output[13] + _gridY;
	auto xs7 = output[14] + _gridX;
	auto ys7 = output[15] + _gridY;
	auto xs8 = output[16] + _gridX;
	auto ys8 = output[17] + _gridY;
	auto det_confs = Sigmoid(output[18]);

	float max_conf = -1.0f;
	int max_ind = -1;
	for (int c = 0; c < ROW_COUNT * COL_COUNT; c++)
	{
		float conf = det_confs[c];

		if (conf > max_conf)
		{
			max_conf = conf;
			max_ind = c;
		}
	}

	if (max_ind >= 0)
	{
		Pose pose;
		pose.bounds.push_back({ (xs0[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys0[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs1[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys1[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs2[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys2[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs3[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys3[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs4[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys4[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs5[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys5[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs6[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys6[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs7[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys7[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
		pose.bounds.push_back({ (xs8[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys8[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });

		return pose;
	}

	return Pose();
}

int PoseProcessor::GetOffset(int o, int channel)
{
	static int channelStride = ROW_COUNT * COL_COUNT;
	return (channel * channelStride) + o;
}

std::vector<float> PoseProcessor::Sigmoid(const std::vector<float>& values)
{
	std::vector<float> ret;

	for (std::vector<float>::const_iterator ptr = values.begin(); ptr < values.end(); ptr++)
	{
		float k = (float)std::exp(*ptr);
		ret.push_back(k / (1.0f + k));
	}

	return ret;
}
