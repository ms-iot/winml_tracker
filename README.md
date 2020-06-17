# winml_tracker

![](https://github.com/ms-iot/winml_tracker/workflows/winml_tracker%20CI/badge.svg)

ROS node which uses Windows Machine Learning (WinML) to track people (or other objects) in camera frames.

The node is configured to use camera frames from the OpenCV based cv_camera node at /cv_camera/image_raw .  The Machine Learning model used by the node processes 416 x 416 pixel images as 32-bit RGB planar data.  The node will internally resize whatever is coming from the camera, but it is recommended to find an output dimension close to this size that is natively supported by the camera.

The node expects to use the Tiny YOLO model available in the ONNX model zoo.  Please download and extract this model from the [Onnx.ai Model Zoo](https://github.com/onnx/models) to your local system and update the launch file to refer to the location you have selected.

After processing, the node will publish two outputs.  The first is an array of Marker objects at /tracked_objects .  The second is an image with the ID boxes for tracked objects overlayed on the source frame at /tracked_objects/image .

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
