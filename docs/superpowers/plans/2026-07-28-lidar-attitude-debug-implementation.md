# LiDAR Attitude Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not commit or push; the user requires local-only changes.

**Goal:** Add an optional ROS Noetic diagnostic node and RViz view that color `base_link` points near `z=0` red and compare TF world roll/pitch with LiDAR road-relative roll/pitch.

**Architecture:** Keep the existing preprocessing and inference nodes unchanged. A separate `lidar_attitude_debugger.py` subscribes to `/lidar3D` and `/tf`, performs timestamp-aware attitude estimation, and publishes a 2D debug image, RGB PointCloud2, and MarkerArray.

**Tech Stack:** ROS1 Noetic, rospy, sensor_msgs, tf2_msgs, visualization_msgs, OpenCV, NumPy, catkin/nosetests.

## Global Constraints

- Work only on local branch `feature/lidar-attitude-debug`; do not commit or push.
- Preserve all existing topic and model input contracts.
- Use `base_link` and `abs(z) <= 0.10 m` for the red ground band.
- Label TF attitude as world-relative and LiDAR attitude as road-relative.
- Do not claim absolute attitude accuracy without IMU or another reference sensor.
- Do not modify the source rosbag or `0716demo.json`.

---

### Task 1: Pure geometry and confidence-gated plane estimation

**Files:**
- Create: `src/Validator_pkg/validator_pipeline/scripts/lidar_attitude_debugger.py`
- Create: `src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py`

**Interfaces:**
- Produces: `quaternion_to_rpy_degrees(quaternion) -> np.ndarray`
- Produces: `classify_ground_band(points, half_width) -> np.ndarray[bool]`
- Produces: `fit_ground_plane_ransac(points, config, rng) -> PlaneEstimate`
- Produces: `normal_to_road_rpy_degrees(normal) -> np.ndarray`
- Produces: `render_ground_bev(points, ground_mask, geometry) -> np.ndarray`

- [ ] Write tests for horizontal and known-angle synthetic planes, outlier rejection, invalid confidence cases, quaternion conversion, red-band boundaries, and BEV color.
- [ ] Run `/usr/bin/nosetests3 -v src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py` and verify failure because the new module/functions do not exist.
- [ ] Implement the minimum pure geometry, RANSAC, confidence gating, and rendering functions.
- [ ] Re-run the focused test and verify all cases pass.

### Task 2: Timestamp-aware ROS node and diagnostic publications

**Files:**
- Modify: `src/Validator_pkg/validator_pipeline/scripts/lidar_attitude_debugger.py`
- Modify: `src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py`
- Create: `src/Validator_pkg/validator_pipeline/config/lidar_attitude_debug.yaml`

**Interfaces:**
- Consumes: `/lidar3D` (`sensor_msgs/PointCloud2`)
- Consumes: `/tf` (`tf2_msgs/TFMessage`)
- Produces: `/lidar/debug/ground_bev` (`sensor_msgs/Image`, `bgr8`)
- Produces: `/lidar/debug/colored_points` (`sensor_msgs/PointCloud2`, XYZRGB)
- Produces: `/lidar/debug/markers` (`visualization_msgs/MarkerArray`)

- [ ] Add failing tests for causal TF selection, stale TF rejection, time-reversal reset, median filtering, and marker status selection.
- [ ] Run the focused test and verify the new cases fail.
- [ ] Implement TF history, point decoding/static transform, ROS publishers, text/plane/normal markers, and invalid-state behavior.
- [ ] Re-run the focused test and verify all cases pass.

### Task 3: Launch, RViz, package, and install integration

**Files:**
- Create: `src/Validator_pkg/validator_pipeline/launch/lidar_attitude_debug.launch`
- Create: `src/Validator_pkg/validator_pipeline/rviz/lidar_attitude_debug.rviz`
- Modify: `src/Validator_pkg/validator_pipeline/CMakeLists.txt`
- Modify: `src/Validator_pkg/validator_pipeline/package.xml`
- Modify: `src/Validator_pkg/validator_inference/test/test_package_layout.py`

**Interfaces:**
- Produces: `roslaunch Validator_pkg lidar_attitude_debug.launch`

- [ ] Extend package-layout/install tests to require the script, YAML, launch, RViz file, `visualization_msgs`, and catkin test registration.
- [ ] Run the relevant package tests and verify they fail before integration files are added.
- [ ] Add launch/RViz configuration, install the script, register pipeline nosetests, and declare `visualization_msgs`.
- [ ] Re-run focused integration tests and verify they pass.

### Task 4: Build and rosbag runtime verification

**Files:**
- Modify only if a verified failure requires a scoped correction.

**Interfaces:**
- Verifies all three diagnostic topics and the RViz launch contract.

- [ ] Run Python syntax checks for the new node.
- [ ] Run focused unit and package-layout tests with `/usr/bin/python3`.
- [ ] Run `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3`.
- [ ] Source `devel/setup.bash`, launch the diagnostic node/RViz, and replay `2026-07-16-17-13-14.bag` with `--clock`.
- [ ] Confirm `/lidar/debug/ground_bev`, `/lidar/debug/colored_points`, and `/lidar/debug/markers` publish at LiDAR cadence.
- [ ] Inspect logs for invalid-state reasons and verify that rosbag loop time reversal resets history without crashing.
- [ ] Report numerical and visual validation boundaries without claiming absolute attitude accuracy.
