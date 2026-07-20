FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-catkin-tools python3-numpy python3-pip python3-pytest \
    python3-rospkg python3-rosdep ros-noetic-tf ros-noetic-tf2-ros git && \
    ln -s /usr/bin/pytest-3 /usr/local/bin/pytest && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/E2E-OpenLoop-ROS-Validator

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
