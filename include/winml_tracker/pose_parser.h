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

    };

    struct Pose
    {
    public:
        std::vector<Point> bounds;
    };

    class PoseResultsParser
    {
    public:
		static std::vector<std::vector<float>> _gridX;
		static std::vector<std::vector<float>> _gridY;

		static void initPoseTables();

        static Pose GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
    private:
        static int GetOffset(int x, int y);
        static std::vector<float> Sigmoid(std::vector<float> values);
    };
}