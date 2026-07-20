cmake_minimum_required(VERSION 3.0.2)
project(validator_inference)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  morai_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  validator_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    morai_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    tf2_ros
    validator_msgs
)
