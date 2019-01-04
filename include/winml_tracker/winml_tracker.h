#pragma once

extern int WinMLTracker_Init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
extern int WinMLTracker_Startup(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
extern int WinMLTracker_Shutdown(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
extern void ProcessImage(const sensor_msgs::ImageConstPtr& image);


extern winrt::Windows::AI::MachineLearning::LearningModel model;
extern winrt::Windows::AI::MachineLearning::LearningModelSession session;

typedef enum 
{
    WinMLTracker_Yolo,
    WinMLTracker_Pose
} WinMLTracker_Type;

typedef enum 
{
    WinMLTracker_Scale,
    WinMLTracker_Crop
} WinMLTracker_ImageProcessing;

extern WinMLTracker_Type TrackerType;
extern WinMLTracker_ImageProcessing ImageProcessingType;

extern std::vector<cv::Point3d> modelBounds;

