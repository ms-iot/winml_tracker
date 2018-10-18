# winml_tracker
ROS node which uses Windows Machine Learning (WinML) to track people (or other objects) in camera frames.

The node is configured to use camera frames from the OpenCV based cv_camera node at /cv_camera/image_raw .  The Machine Learning model used by the node processes 416 x 416 pixel images as 32-bit RGB planar data.  The node will internally resize whatever is coming from the camera, but it is recommended to find an output dimension close to this size that is natively supported by the camera.

The node expects to use the Tiny YOLO model available in the ONNX model zoo.  Please download and extract this model from https://www.cntk.ai/OnnxModels/tiny_yolov2/opset_7/tiny_yolov2.tar.gz to your local system and update the launch file to refer to the location you have selected.

After processing, the node will publish two outputs.  The first is an array of Marker objects at /tracked_objects .  The second is an image with the ID boxes for tracked objects overlayed on the source frame at /tracked_objects/image .


