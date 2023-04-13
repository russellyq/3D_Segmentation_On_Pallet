cmake_minimum_required(VERSION 2.8.3)
project(palletunloader)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
  sensor_msgs
  laser_geometry
  visualization_msgs
  tf
  serialrotor
)


add_message_files(
  FILES
  pallet_report_msg.msg
  pallet_report_stamped_msg.msg
)


 add_service_files(
   FILES
   Capture3Dservice.srv
   pallet_report_srv.srv
   which_pallet_srv.srv
 )


## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   sensor_msgs
 )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES palletdetector
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
include_directories(
# include
${catkin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME} src/Pallet_Unloader.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_executable(palletsubscriber src/palletsubscriber.cpp)
target_link_libraries(palletsubscriber ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
