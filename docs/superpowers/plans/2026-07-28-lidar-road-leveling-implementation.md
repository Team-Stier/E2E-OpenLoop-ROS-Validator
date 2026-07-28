# LiDAR Road-Leveling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a confidence-gated `/lidar3D/leveled` point cloud whose estimated road surface is `z=0`, without changing `/lidar3D` or any inference input.

**Architecture:** Extend the existing `lidar_attitude_debugger.py` diagnostic node. A pure NumPy transform builds an orthonormal road frame from the accepted RANSAC plane, while a separate ROS publisher emits XYZ `PointCloud2` messages only for valid fits.

**Tech Stack:** ROS1 Noetic, `rospy`, `sensor_msgs/PointCloud2`, NumPy, catkin, Python `unittest`/nosetests.

## Global Constraints

- Work on the existing `feature/lidar-attitude-debug` branch and preserve all user-owned uncommitted changes.
- Do not commit, push, merge, reset, or alter Git history.
- Keep `/lidar3D` and the inference model contracts unchanged.
- Publish the new diagnostic topic as `/lidar3D/leveled`.
- Do not use invalid RANSAC estimates for compensation.
- Do not claim intra-scan motion deskew; this feature is frame-level road-plane compensation.

---

### Task 1: Road-frame geometry

**Files:**
- Modify: `src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py`
- Modify: `src/Validator_pkg/validator_pipeline/scripts/lidar_attitude_debugger.py`

**Interfaces:**
- Consumes: `points: numpy.ndarray[N, 3]`, `normal: numpy.ndarray[3]`, `offset: float`.
- Produces: `level_points_to_plane(points, normal, offset) -> numpy.ndarray[N, 3]`.

- [ ] **Step 1: Write failing geometry tests**

Add tests which construct a hand-derived tilted plane basis. Put points on the
plane and at `1.2 m` along its normal, then require output coordinates
`[3, 2, 0]` and `[3, 2, 1.2]`. Add horizontal identity and zero-normal
rejection cases.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest \
  src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py
```

Expected: failure because `level_points_to_plane` does not exist.

- [ ] **Step 3: Implement the minimal transform**

Normalize the upward plane equation, project the base forward axis onto the
plane, build `y_road = cross(z_road, x_road)`, and compute:

```python
basis = np.column_stack([x_axis, y_axis, normal])
leveled = points @ basis
leveled[:, 2] += offset
```

Return a contiguous `float32` array and reject a zero normal or non-finite
offset.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all geometry tests pass.

### Task 2: Confidence-gated PointCloud2 output

**Files:**
- Modify: `src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py`
- Modify: `src/Validator_pkg/validator_pipeline/scripts/lidar_attitude_debugger.py`
- Modify: `src/Validator_pkg/validator_pipeline/config/lidar_attitude_debug.yaml`

**Interfaces:**
- Consumes: valid `PlaneEstimate`, base-frame points, source ROS header.
- Produces: `/lidar3D/leveled`, `sensor_msgs/PointCloud2`, XYZ float fields,
  source timestamp/sequence, `frame_id=base_link`.

- [ ] **Step 1: Write a failing publisher test**

Create a node object without starting ROS, attach a collecting publisher, call
`publish_leveled_cloud`, and assert the real generated message preserves the
header and contains the supplied XYZ values.

- [ ] **Step 2: Run the focused publisher test and verify RED**

Run the test module from Task 1. Expected: failure because
`publish_leveled_cloud` does not exist.

- [ ] **Step 3: Add the publisher and valid-fit callback path**

Read `~leveled_points_topic` with default `/lidar3D/leveled`, create a
queue-size-one `PointCloud2` publisher, and after a valid RANSAC result call:

```python
leveled_points = level_points_to_plane(
    points,
    plane.normal,
    plane.offset,
)
self.publish_leveled_cloud(leveled_points, message.header)
```

Do not publish this topic from the invalid branch.

- [ ] **Step 4: Run the test module and verify GREEN**

Run the test module from Task 1. Expected: all tests pass.

### Task 3: RViz and runtime contract

**Files:**
- Modify: `src/Validator_pkg/validator_pipeline/test/test_lidar_attitude_debugger.py`
- Modify: `src/Validator_pkg/validator_pipeline/rviz/lidar_attitude_debug.rviz`
- Modify: `src/Validator_pkg/validator_pipeline/scripts/lidar_attitude_debugger.py`

**Interfaces:**
- Consumes: `/lidar3D/leveled`.
- Produces: an independently toggleable RViz `PointCloud2` display.

- [ ] **Step 1: Extend the package integration test**

Require the RViz configuration to include `/lidar3D/leveled` while retaining
the existing raw-derived colored cloud and marker topics.

- [ ] **Step 2: Run the integration test and verify RED**

Run the test module from Task 1. Expected: failure because the RViz topic is
absent.

- [ ] **Step 3: Add the RViz display and document node I/O**

Add a `Road-Leveled Points` display using `/lidar3D/leveled`, a distinct flat
color, and `base_link` as the fixed frame. Add the topic to the script module
docstring.

- [ ] **Step 4: Run all package tests**

Run:

```bash
source /opt/ros/noetic/setup.bash
/usr/bin/python3 -m unittest discover \
  -s src/Validator_pkg/validator_pipeline/test -p 'test_*.py'
```

Expected: zero failures.

### Task 4: Build and rosbag verification

**Files:**
- No production file changes unless verification reveals a defect.

**Interfaces:**
- Verifies: package install rules, ROS type/frame/frequency, ground-level
  residual, and unchanged raw input contract.

- [ ] **Step 1: Build with the ROS system Python**

Run:

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Expected: exit code `0`.

- [ ] **Step 2: Restart the local BEV diagnostic**

Run the existing local `scripts/run_lidar_bev_debug.sh`, which launches the
attitude node, RViz, and the detected local rosbag.

- [ ] **Step 3: Verify the live ROS contract**

Check:

```bash
rostopic type /lidar3D/leveled
rostopic hz /lidar3D/leveled
rostopic echo -n 1 /lidar3D/leveled/header
```

Expected: `sensor_msgs/PointCloud2`, a nonzero rate while the bag plays, and
`frame_id: base_link`.

- [ ] **Step 4: Measure leveled ground residual**

Sample the output cloud and confirm points used by the accepted ground fit are
centered around `z=0` within the configured RANSAC/ground-band tolerances.
Report the observed value; do not invent a result if live sampling is blocked.
