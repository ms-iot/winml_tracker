#pragma comment(lib, "windowsapp")
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>

#include "winml_tracker/winml_tracker.h"
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


ros::NodeHandle* g_nh;

class MarkerHelper
{
    bool _called;
public:
    MarkerHelper()
    : _called(false)
    {
    }

    void cb(const visualization_msgs::MarkerArray::ConstPtr& msg)
    {
        _called = true;
    }

    bool wasCalled()
    {
        return _called;
    }
};

TEST(TrackerTester, poseTableTest)
{
	pose::PoseResultsParser::initPoseTables();
/*
	for (auto v = pose::PoseResultsParser::_gridX.begin(); v != pose::PoseResultsParser::_gridX.end(); ++v)
	{
		std::cout << *v << ' ';
	}
	std::cout << '\n';

	for (auto v = pose::PoseResultsParser::_gridY.begin(); v != pose::PoseResultsParser::_gridY.end(); ++v)
	{
		std::cout << *v << ' ';
	}
	std::cout << '\n';
	*/
}

TEST(TrackerTester, poseTest)
{
    MarkerHelper mh;
	image_transport::ImageTransport it(*g_nh);
	image_transport::Publisher image_pub;
	image_pub = it.advertise("debug/image", 1, true);
    ros::Subscriber sub = g_nh->subscribe("tracked_objects", 0, &MarkerHelper::cb, &mh);
    cv::Mat image_data = cv::imread( "C:\\ws\\eden_ws\\src\\winml_tracker\\testdata\\sample_image_1.JPG");

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_data).toImageMsg();
    EXPECT_NE(msg, nullptr);
	image_pub.publish(msg);
	ros::spinOnce();
	/*
	modelBounds.push_back(cv::Point3d(14.0, 5.5, 0.0));
	modelBounds.push_back(cv::Point3d(14.0, -5.5, 0.0));
	modelBounds.push_back(cv::Point3d(- 14.0, -5.5, 0.0));
	modelBounds.push_back(cv::Point3d(- 14.0, 5.5, 0.0));
	modelBounds.push_back(cv::Point3d(14.0, 5.5, 10.0));
	modelBounds.push_back(cv::Point3d(14.0, -5.5, 10.0));
	modelBounds.push_back(cv::Point3d(- 14.0, -5.5, 10.0));
	modelBounds.push_back(cv::Point3d(- 14.0, 5.5, 10.0));
	*/
	modelBounds.push_back(cv::Point3d(-4.57, -10.50, -13.33));
	modelBounds.push_back(cv::Point3d(5.54, -10.50, -13.33));
	modelBounds.push_back(cv::Point3d(5.54, 0.00, -13.33));
	modelBounds.push_back(cv::Point3d(-4.57, 0.00, -13.33));
	modelBounds.push_back(cv::Point3d(-4.57, -10.50, 13.73));
	modelBounds.push_back(cv::Point3d(5.54, -10.50, 13.73));
	modelBounds.push_back(cv::Point3d(5.54, 0.00, 13.73));
	modelBounds.push_back(cv::Point3d(-4.57, 0.00, 13.73));
	modelBounds.push_back(cv::Point3d(0.48, -5.25, 0.20));
	











	hstring modelPath = hstring(wstring_to_utf8().from_bytes("C:\\ws\\eden_ws\\src\\winml_tracker\\testdata\\shoe.onnx"));
    model = LearningModel::LoadFromFilePath(modelPath);
	EXPECT_NE(model, nullptr);

    TrackerType = WinMLTracker_Pose;

    // Create a WinML session
    session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Cpu));
	EXPECT_NE(session, nullptr);

    ProcessImage(msg);

    //ros::spinOnce();
	ros::spin();

    EXPECT_TRUE(mh.wasCalled());
}

int main(int argc, char** argv)
{
    init_apartment();
    testing::InitGoogleTest(&argc, argv);
	ros::init(argc, argv, "tester");
	
	ros::NodeHandle nh;
	g_nh = &nh;

	EXPECT_EQ(WinMLTracker_Init(nh, nh), 0);

    int ret = RUN_ALL_TESTS();

    WinMLTracker_Shutdown(nh, nh);

    return ret;
}