#define _USE_MATH_DEFINES
#include <ros/ros.h>
#include <angles/angles.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/InteractiveMarker.h>

// current heading in degrees
float yaw = 0.0f;
float yawRad = 0.0f;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    tf::Quaternion tfQ;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, tfQ);

	yawRad = (float)tfQ.getAngle();
    yaw = (float)angles::to_degrees(yawRad);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fake_laser_publisher");
    ros::NodeHandle n;
    ros::NodeHandle priv_nh("~");
    sensor_msgs::LaserScan scan;
    ros::Time lastScanTime = ros::Time::now();
    ros::Rate publishTime(10);

    ros::Publisher laser_pub = n.advertise<sensor_msgs::LaserScan>("scan", 1000);
    ros::Publisher detect_pub = n.advertise<visualization_msgs::Marker >("tracked_objects", 1);

	ros::Subscriber odomSub = n.subscribe("odom", 500, odomCallback);

    while (ros::ok())
    {
        ros::Time currentTime = ros::Time::now();
        ros::Duration increment = lastScanTime - currentTime;

        scan.header.frame_id = std::string("base_scan");
        scan.angle_increment = (float)(M_PI / 180.0);
        scan.angle_min = 0.0f;
        scan.angle_max = 2.0f*M_PI - scan.angle_increment;
        scan.range_min = 0.12f;
        scan.range_max = 3.5f;
        scan.time_increment = (float)increment.toSec();
        scan.header.stamp = ros::Time::now();
        scan.ranges.resize(360);
        scan.intensities.resize(360);

        for (int index = 0; index < 360; index++)
        {
            if (index > yaw + 10 && index < yaw + 20)
            {
                scan.ranges[index] = scan.range_min * 2.0f;
				scan.intensities[index] = scan.range_min * 2.0f;
			}
            else if (index > yaw + 30 && index < yaw + 40)
			{
				scan.ranges[index] = scan.range_min * 2.5f;
				scan.intensities[index] = scan.range_min * 2.5f;
			}
			else
            {
                scan.ranges[index] = scan.range_max / 2.0f;
				scan.intensities[index] = 1.0f;
			}
        }

        lastScanTime = currentTime;
        laser_pub.publish(scan);

        visualization_msgs::Marker msg;
        msg.header.frame_id = std::string("camera_rgb_optical_frame");
		msg.header.stamp = ros::Time::now();
        msg.ns = "Tracked";
        msg.id = 0;
        msg.type = visualization_msgs::Marker::CUBE;
        msg.action = visualization_msgs::Marker::MK_ADD;
        msg.scale.x = 0.10;
        msg.scale.y = 0.10;
        msg.scale.z = .01;
        msg.color.r = 0.0f;
        msg.color.g = 1.0f;
        msg.color.b = 0.0f;
        msg.color.a = 1.0f;
        msg.lifetime = ros::Duration();
		msg.pose.position.z = (scan.range_min * 2.0f + 0.01f) * cos(yawRad);
		msg.pose.position.x = (scan.range_min * 2.0f + 0.01f) * sin(yawRad);
		msg.pose.position.y = 0;
        msg.pose.orientation.z = 1.0;
        detect_pub.publish(msg);        

        ros::spinOnce();
        publishTime.sleep();
    }
    return 0;
}
