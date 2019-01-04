#pragma once

#include <string>
#include <vector>

namespace pose
{
    const int ROW_COUNT = 13;
    const int COL_COUNT = 13;
    const int CHANNEL_COUNT = 20;

    class Pose
    {
    public:
		Pose() {}
		Pose(Pose&& p) : bounds(std::move(p.bounds)) {}

        std::vector<cv::Point2f> bounds;
    };

    class PoseProcessor : public WinMLProcessor
    {
		static std::vector<float> _gridX;
		static std::vector<float> _gridY;

    public:
        std::vector<cv::Point3d> modelBounds;

        virtual bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    private:
		void initPoseTables();

        Pose GetRecognizedObjects(std::vector<float> modelOutputs);
        int GetOffset(int o, int channel);
        std::vector<float> Sigmoid(const std::vector<float>& values);
    protected:
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);
    };
}