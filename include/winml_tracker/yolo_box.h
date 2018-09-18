#pragma once

#include <string>
#include <vector>

namespace yolo
{
    const int ROW_COUNT = 13;
    const int COL_COUNT = 13;
    const int CHANNEL_COUNT = 125;
    const int BOXES_PER_CELL = 5;
    const int BOX_INFO_FEATURE_COUNT = 5;
    const int CLASS_COUNT = 20;
    const float CELL_WIDTH = 32;
    const float CELL_HEIGHT = 32;

    static const std::string labels[CLASS_COUNT] =
    {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    struct YoloBox
    {
    public:
        std::string label;
        float x, y, width, height, confidence;
    };

    class YoloResultsParser
    {
    public:
        static std::vector<YoloBox> GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
    private:
        static int GetOffset(int x, int y, int channel);
        static float IntersectionOverUnion(YoloBox a, YoloBox b);
        static float Sigmoid(float value);
        static void Softmax(std::vector<float> values);
    };
}