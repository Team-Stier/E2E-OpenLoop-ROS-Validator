cmake_minimum_required(VERSION 3.0.2)
project(validator_inference)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  morai_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_msgs
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
    tf2_msgs
    tf2_ros
    validator_msgs
)

catkin_install_python(PROGRAMS scripts/opt_feeder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/recorder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/fetch_model_assets.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/inference_engine_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../config/profile.json
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
