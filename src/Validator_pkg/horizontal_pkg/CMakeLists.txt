cmake_minimum_required(VERSION 3.0.2)
project(horizontal_pkg)

find_package(catkin REQUIRED COMPONENTS
  rosbag
  sensor_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rosbag sensor_msgs
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_imu_orientation_analyzer.py)
  catkin_add_nosetests(test/test_imu_filtering.py)
endif()
