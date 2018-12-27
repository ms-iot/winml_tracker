#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>

#include "winml_tracker/winml_tracker.h"

using namespace std;
using namespace winrt;

int main(int argc, char **argv)
{
    winrt::init_apartment();
    ros::init(argc, argv, "winml_tracker");

    ros::NodeHandle nh;

    int ret = WinMLTracker_Startup(nh);
    if (ret != 0)
    {
        ros::spin();

        WinMLTracker_Shutdown(nh);
    }
}