# Offline LiDAR Horizontalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an offline ROS1 bag transformer that adds completely roll/pitch-leveled LiDAR clouds and align the repository's shared LiDAR extrinsic with the July 31 bag TF.

**Architecture:** A single `lidar_horizontal.py` module provides pure geometry/time/PointCloud functions plus an atomic bag CLI. It reuses the existing offline IMU conversion and filtering modules, derives sensor-relative rotation from bag TF, copies all source records, and adds `/lidar3D/horizontal` only when header-time alignment is valid.

**Tech Stack:** ROS1 Noetic system Python 3, rosbag, sensor_msgs, tf2_msgs, NumPy, SciPy, Catkin nosetests.

## Global Constraints

- Runtime file: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`.
- Output topic/frame: `/lidar3D/horizontal`, `Lidar3D-4-horizontal`.
- Filter: existing second-order 4.0 Hz zero-phase Butterworth roll/pitch filter.
- Timestamp authority: sensor `header.stamp`; no extrapolation; maximum bracket age/span `0.10 s`.
- Output bag copies every original record and atomically adds valid derived clouds with LZ4 compression.
- Preserve PointCloud2 layout and all non-XYZ bytes; mutate only FLOAT32 x/y/z.
- Shared `T_base_from_lidar` translation becomes `[1.5, 0.0, 1.25]` with identity rotation.
- Do not modify the input bag, original sensor/map/route files, BEV wiring, model-input selection, launch files, or controller behavior.

---

### Task 1: Geometry and IMU Time Alignment

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`
- Create: `src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py`

**Interfaces:**
- Produces `HorizontalizationError`.
- Produces `rotation_matrix_xyz(roll_rad, pitch_rad, yaw_rad=0.0) -> ndarray(3,3)`.
- Produces `level_rotation(roll_rad, pitch_rad, R_imu_from_lidar) -> ndarray(3,3)` implementing `Ry(pitch) @ Rx(roll) @ R_imu_from_lidar`.
- Produces `AttitudeTimeline(stamps_ns, roll_rad, pitch_rad)` and `interpolate(stamp_ns, max_gap_s) -> (roll_rad, pitch_rad) | None`.

- [ ] **Step 1: Write failing geometry and alignment tests**

Use literal hand-derived vectors for +10 degree roll, -8 degree pitch, combined
order, and one fixed 90 degree sensor-relative rotation. Add exact stamp,
midpoint, outside-range, excessive bracket-span, excessive one-sided age, and
non-increasing timeline tests.

- [ ] **Step 2: Run tests and verify RED**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 -m unittest -v \
  src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py
```

Expected: import failure because `horizontal_pkg.lidar_horizontal` is absent.

- [ ] **Step 3: Implement minimal geometry and alignment**

Implement explicit Rx/Ry/Rz matrices, finite/shape/orthonormal checks, and
binary-search interpolation over strictly increasing integer nanosecond stamps.
Exact matches bypass bracket interpolation. Return `None` for a rejected time,
not a stale angle.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py \
  src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py
git commit -m "feat: define LiDAR leveling geometry"
```

---

### Task 2: TF Extraction, IMU Preparation, and PointCloud2 Transformation

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py`

**Interfaces:**
- Produces `RigidTransform(translation_m, quaternion_xyzw)` and `Extrinsics(T_base_from_imu, T_base_from_lidar, R_imu_from_lidar)`.
- Produces `extract_extrinsics(tf_messages, base_frame, imu_frame, lidar_frame) -> Extrinsics`.
- Produces `prepare_attitude(imu_messages, cutoff_hz) -> (AttitudeTimeline, ImuStats)`.
- Produces `horizontalize_cloud(message, rotation, output_frame) -> PointCloud2`.

- [ ] **Step 1: Write failing TF, IMU, and binary cloud tests**

Build complete fake TransformStamped, Imu, PointField, and PointCloud2-shaped
objects. Assert median bag transforms yield `[0,0,0.75]`, invalid frames and
unstable/nonfinite transforms fail, duplicate IMU stamps keep the last valid
sample, invalid quaternions are counted, and 4 Hz filtering is used. Construct
an organized FLOAT32 XYZI payload and assert hand-derived XYZ values while
header stamp, dimensions, field descriptors, intensity bytes, padding bytes,
and NaNs remain unchanged. Add malformed datatype/offset/row-step/payload tests.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 1 command. Expected: failures for missing Task 2 interfaces.

- [ ] **Step 3: Implement TF, IMU, and cloud behavior**

Normalize quaternions, canonicalize signs, use component medians and reject
translation spread above `0.005 m` or rotation deviation above `0.1 deg`.
Reuse `imu_orientation_analyzer.quaternion_to_roll_pitch` and
`imu_filtering.filter_orientation`. Transform copied PointCloud payloads using
NumPy strided FLOAT32 views with explicit endian handling and row padding.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 command. Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py \
  src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py
git commit -m "feat: horizontalize LiDAR point clouds"
```

---

### Task 3: Atomic Bag Pipeline and CLI

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py`
- Modify: `src/Validator_pkg/horizontal_pkg/CMakeLists.txt`
- Modify: `src/Validator_pkg/horizontal_pkg/package.xml`
- Modify: `README.md`

**Interfaces:**
- Produces `process_bag(input_bag, output_bag, ..., bag_factory=None) -> ProcessingStats`.
- Produces `main(argv=None) -> int` with the approved CLI defaults.

- [ ] **Step 1: Write failing bag and CLI tests**

Use an injected in-memory bag factory that mirrors read/write/context-manager
behavior. Assert the pipeline makes a preparation pass for IMU/TF, copies every
record in original order, inserts the derived cloud immediately after its raw
cloud at the same record time, skips unaligned clouds, refuses missing topics
and zero outputs, uses LZ4, and atomically replaces only after close. Add input
equals output, temporary creation, write, replace, CLI success, and CLI expected
error tests.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 1 command. Expected: failures for missing bag and CLI interfaces.

- [ ] **Step 3: Implement bag processing and CLI**

Read IMU and TF on pass one, then copy/derive on pass two. Import `rosbag`
lazily, create a destination-local temporary `.bag`, use LZ4 compression, close
both bags before `os.replace`, remove temporary output on every failure, print
complete counters/extrinsics/runtime, and return `2` for expected errors.

- [ ] **Step 4: Wire Catkin and document execution**

Register `test/test_lidar_horizontal.py` with `catkin_add_nosetests`, retain the
existing ROS/system dependencies, and document the exact command, derived topic
and frame, complete-leveling semantics, and BEV/model exclusion in `README.md`.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 -m unittest -v \
  src/Validator_pkg/horizontal_pkg/test/test_lidar_horizontal.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_filtering.py \
  src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py
```

Expected: all horizontal package tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add README.md src/Validator_pkg/horizontal_pkg
git commit -m "feat: add horizontalized LiDAR bag topic"
```

---

### Task 4: Shared Extrinsic Migration

**Files:**
- Modify: `src/Validator_pkg/config/profile.json`
- Modify: direct shared-profile assertions under `src/Validator_pkg/validator_inference/test/`
- Modify: `README.md`
- Modify: `docs/model_io_contracts.md`
- Modify: `docs/inference_engine_design.md`
- Modify: `src/Validator_pkg/validator_pipeline/README_BEV_VISUALIZATION_SETUP.md`

**Interfaces:**
- Produces one shared `T_base_from_lidar` contract with translation `[1.5, 0.0, 1.25]` and identity quaternion.

- [ ] **Step 1: Change profile-consumer tests to the bag value and verify RED**

Update only tests that read/assert the shared profile. Do not change independent
transform fixtures that intentionally test arbitrary translations. Run affected
profile, preprocessing, integration, and smoke tests. Expected: failures while
the profile remains at the old value.

- [ ] **Step 2: Update all canonical profile declarations**

Change the three `[2.0, 0.0, 1.5]` declarations in `profile.json` to
`[1.5, 0.0, 1.25]`. Update prose that directly states the shared transform and
adjust numeric waypoint examples from base x `7/17/32` to `6.5/16.5/31.5`.

- [ ] **Step 3: Run affected tests and verify GREEN**

Run the affected validator tests through their existing Catkin target. Expected:
all shared-profile consumers pass with the bag-derived value.

- [ ] **Step 4: Commit Task 4**

```bash
git add README.md docs src/Validator_pkg/config/profile.json \
  src/Validator_pkg/validator_inference/test \
  src/Validator_pkg/validator_pipeline/README_BEV_VISUALIZATION_SETUP.md
git commit -m "fix: align LiDAR extrinsic with bag TF"
```

---

### Task 5: Actual Bag and Full Verification

**Files:**
- Runtime artifact only, ignored: `/home/stier/Rosbag/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T181003KST_horizontal.bag`

**Interfaces:**
- Consumes the implemented CLI and current July 31 bag.
- Produces validation counts, runtime, size, and plane metrics; no checked-in data artifact.

- [ ] **Step 1: Check disk space and run the actual conversion**

Verify enough free space for an additional compressed bag, then run the approved
CLI with default topics, frame, cutoff, and gap. Do not overwrite the input bag.

- [ ] **Step 2: Validate the output bag contract**

Use `rosbag info` and direct reads to confirm every source topic count is
preserved, `/lidar3D/horizontal` exists, output count equals the reported
written count, original and derived bag record/header stamps match, output frame
and fields are correct, and sampled intensity/non-XYZ bytes are preserved.

- [ ] **Step 3: Measure geometry and runtime**

Fit ground planes to timestamp-matched raw and horizontal clouds in identical
ROIs. Report median/p95 normal tilt and plane residual for both, the number of
evaluated frames, conversion wall time, output bag size, and skipped reasons.

- [ ] **Step 4: Run full verification**

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make run_tests
catkin_test_results --all build/test_results
/usr/bin/python3 -m py_compile \
  src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py
git diff --check
```

Expected: build succeeds, all tests report zero failures, compilation succeeds,
and the diff check is silent.

- [ ] **Step 5: Request independent code review and resolve findings**

Review geometry signs/order, timestamp causality, binary payload handling,
atomic bag behavior, profile migration scope, and actual validation evidence.
Fix Critical/Important findings through a new RED/GREEN cycle.
