cmake_minimum_required(VERSION 2.8.3)
project(winml_tracker)

## Compile as at least C++17 for cppwinrt
add_compile_options(/Od)
add_compile_options(/std:c++latest)
add_compile_options(/await)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  interactive_markers
  roscpp
  roslib
  std_msgs
  Eigen3 
  visualization_msgs
)
include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
add_definitions(${EIGEN3_DEFINITIONS})

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES winml_tracker
  CATKIN_DEPENDS cv_bridge geometry_msgs image_transport interactive_markers roscpp roslib std_msgs visualization_msgs
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/winml_tracker.cpp src/main.cpp src/yolo_box.cpp src/pose_parser.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} opencv_calib3d)

install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch testdata
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest_gtest(${PROJECT_NAME}_testnode test/testtracker.launch test/testtracker.cpp src/winml_tracker.cpp src/yolo_box.cpp  src/pose_parser.cpp)
  target_link_libraries(${PROJECT_NAME}_testnode ${catkin_LIBRARIES} opencv_calib3d)
endif()
