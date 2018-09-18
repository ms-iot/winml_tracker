#include "winml_tracker/yolo_box.h"

#include <algorithm>
#include <numeric>
#include <functional>

namespace yolo
{
    std::vector<YoloBox> YoloResultsParser::GetRecognizedObjects(std::vector<float> modelOutputs, float threshold)
    {
        static float anchors[] =
        {
            1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f
        };
        static int featuresPerBox = BOX_INFO_FEATURE_COUNT + CLASS_COUNT;
        static int stride = featuresPerBox * BOXES_PER_CELL;

        std::vector<YoloBox> boxes;

        for (int cy = 0; cy < ROW_COUNT; cy++)
        {
            for (int cx = 0; cx < COL_COUNT; cx++)
            {
                for (int b = 0; b < BOXES_PER_CELL; b++)
                {
                    int channel = (b * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));
                    float tx = modelOutputs[GetOffset(cx, cy, channel)];
                    float ty = modelOutputs[GetOffset(cx, cy, channel + 1)];
                    float tw = modelOutputs[GetOffset(cx, cy, channel + 2)];
                    float th = modelOutputs[GetOffset(cx, cy, channel + 3)];
                    float tc = modelOutputs[GetOffset(cx, cy, channel + 4)];

                    float x = ((float)cx + Sigmoid(tx)) * CELL_WIDTH;
                    float y = ((float)cy + Sigmoid(ty)) * CELL_HEIGHT;
                    float width = (float)exp(tw) * CELL_WIDTH * anchors[b * 2];
                    float height = (float)exp(th) * CELL_HEIGHT * anchors[b * 2 + 1];

                    float confidence = Sigmoid(tc);
                    if (confidence < threshold)
                        continue;

                    std::vector<float> classes(CLASS_COUNT);
                    float classOffset = channel + BOX_INFO_FEATURE_COUNT;

                    for (int i = 0; i < CLASS_COUNT; i++)
                        classes[i] = modelOutputs[GetOffset(cx, cy, i + classOffset)];

                    Softmax(classes);

                    // Get the index of the top score and its value
                    auto iter = std::max_element(classes.begin(), classes.end());
                    float topScore = (*iter) * confidence;
                    int topClass = std::distance(classes.begin(), iter);

                    if (topScore < threshold)
                        continue;

                    YoloBox top_box = {
                        labels[topClass],
                        (x - width / 2),
                        (y - height / 2),
                        width,
                        height,
                        topScore
                    };
                    boxes.push_back(top_box);
                }
            }
        }

        return boxes;
    }

    float YoloResultsParser::IntersectionOverUnion(YoloBox a, YoloBox b)
    {
        int areaA = a.width * a.height;

        if (areaA <= 0)
            return 0;

        int areaB = b.width * b.height;
        if (areaB <= 0)
            return 0;

        int minX = std::max(a.x, b.x);
        int minY = std::max(a.y, b.y);
        int maxX = std::min(a.x + a.width, b.x + b.width);
        int maxY = std::min(a.y + a.height, b.x + b.width);
        int intersectionArea = std::max(maxY - minY, 0) * std::max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    int YoloResultsParser::GetOffset(int x, int y, int channel)
    {
        // YOLO outputs a tensor that has a shape of 125x13x13, which 
        // WinML flattens into a 1D array.  To access a specific channel 
        // for a given (x,y) cell position, we need to calculate an offset
        // into the array
        static int channelStride = ROW_COUNT * COL_COUNT;
        return (channel * channelStride) + (y * COL_COUNT) + x;
    }

    float YoloResultsParser::Sigmoid(float value)
    {
        float k = (float)std::exp(value);
        return k / (1.0f + k);
    }

    void YoloResultsParser::Softmax(std::vector<float> values)
    {
        float max_val{ *std::max_element(values.begin(), values.end()) };
        std::transform(values.begin(), values.end(), values.begin(),
            [&](float x) { return std::exp(x - max_val); });

        float exptot = std::accumulate(values.begin(), values.end(), 0.0);
        std::transform(values.begin(), values.end(), values.begin(),
            [&](float x) { return (float)(x / exptot); });
    }
}