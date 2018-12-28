#pragma once

#include <string>
#include <vector>

namespace pose
{
    const int ROW_COUNT = 13;
    const int COL_COUNT = 13;
    const int CHANNEL_COUNT = 20;

    struct Point
    {
		float x;
		float y;
    };

    struct Pose
    {
    public:
        std::vector<Point> bounds;
    };

    class PoseResultsParser
    {
    public:
		static std::vector<float> _gridX;
		static std::vector<float> _gridY;

		static void initPoseTables();

        static Pose GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
    private:
        static int GetOffset(int o, int channel);
        static std::vector<float> Sigmoid(const std::vector<float>& values);
    };
}