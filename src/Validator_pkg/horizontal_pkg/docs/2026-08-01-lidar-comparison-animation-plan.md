# LiDAR Comparison Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a high-contrast MP4 comparing paired raw and horizontalized LiDAR point clouds over the complete recorded drive.

**Architecture:** A focused offline module reads paired PointCloud2 messages, deterministically selects frames and point indices, renders fixed-limit side/front Matplotlib panels, and streams RGB frames to ffmpeg. The final MP4 is created through a temporary file and atomically moved into `horizontal_pkg/docs` after validation.

**Tech Stack:** ROS1 Noetic `rosbag`, `sensor_msgs/PointCloud2`, Python 3, NumPy, Matplotlib Agg, ffmpeg/ffprobe, catkin nosetests.

## Global Constraints

- All implementation, tests, plans, and generated assets stay under `src/Validator_pkg/horizontal_pkg`.
- Raw color is `#FF5A36`; horizontalized color is `#16E0E0`; background is dark.
- Input bags are read-only and the final output is `docs/lidar_horizontal_comparison.mp4`.
- Pair messages by identical header timestamp and compare identical point indices.
- Use fixed X-Z and Y-Z view limits throughout the video.
- A failed render must not leave a partial final MP4.

---

### Task 1: Deterministic PointCloud Pairing and Sampling

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_comparison_animation.py`
- Create: `src/Validator_pkg/horizontal_pkg/test/test_lidar_comparison_animation.py`
- Modify: `src/Validator_pkg/horizontal_pkg/CMakeLists.txt`

**Interfaces:**
- Consumes: rosbag records from `/lidar3D` and `/lidar3D/horizontal`.
- Produces: `extract_xyz(message) -> numpy.ndarray`, `pair_clouds(records) -> iterator`, and `uniform_indices(total, target) -> tuple[int, ...]`.

- [ ] **Step 1: Write failing tests** covering XYZ extraction, exact-stamp pairing, unmatched-record accounting, incompatible layouts, and deterministic uniform indices including first/last frames.
- [ ] **Step 2: Run** `/usr/bin/nosetests3 -P src/Validator_pkg/horizontal_pkg/test/test_lidar_comparison_animation.py` and confirm missing-module failure.
- [ ] **Step 3: Implement** strict FLOAT32 XYZ views, layout signature comparison, bounded pending-message pairing, and integer uniform selection without randomness.
- [ ] **Step 4: Register** the new nosetest in `CMakeLists.txt` and run the focused test to green.
- [ ] **Step 5: Commit** with `feat: pair LiDAR clouds for visualization`.

### Task 2: High-Contrast Frame Rendering and Atomic Encoding

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_comparison_animation.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_lidar_comparison_animation.py`

**Interfaces:**
- Consumes: paired XYZ arrays from Task 1.
- Produces: `render_frame(raw_xyz, horizontal_xyz, metadata, config) -> numpy.ndarray` and `encode_animation(input_bag, output_mp4, ...) -> AnimationStats`.

- [ ] **Step 1: Write failing tests** asserting the exact two data colors, dark background, fixed axis limits, identical downsample indices, RGB output dimensions, ffmpeg argument construction, atomic replacement, and temporary cleanup after a mocked encoder failure.
- [ ] **Step 2: Run** the focused nosetest and confirm failures for missing render/encode interfaces.
- [ ] **Step 3: Implement** an Agg two-panel figure with X-Z and Y-Z overlays, opaque legend swatches, timestamp/frame/point text, fixed limits, and deterministic stride sampling.
- [ ] **Step 4: Implement** raw RGB streaming to ffmpeg using H.264 `libx264`, `yuv420p`, 10 FPS, a temporary MP4 beside the destination, ffprobe validation, and atomic `os.replace`.
- [ ] **Step 5: Run** the focused test and then `catkin_make run_tests_horizontal_pkg`; require zero failures.
- [ ] **Step 6: Commit** with `feat: render LiDAR comparison animation`.

### Task 3: Generate and Inspect the Documentation Asset

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/docs/lidar_horizontal_comparison.mp4`
- Modify: `src/Validator_pkg/horizontal_pkg/docs/2026-08-01-lidar-comparison-animation-design.md` only if actual verified encoder metadata differs from the design.

**Interfaces:**
- Consumes: `/home/stier/Rosbag/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T181003KST_horizontal.bag`.
- Produces: final MP4 and printed pairing/render statistics.

- [ ] **Step 1: Run** the module with the approved bag, output path, 10 FPS, fixed side/front limits, and deterministic point cap.
- [ ] **Step 2: Verify** with `ffprobe` that codec is H.264, pixel format is yuv420p, frame count and duration are positive, and file size is nonzero.
- [ ] **Step 3: Extract** first/middle/last preview frames into a temporary directory and visually inspect color separation, fixed axes, labels, and nonblank clouds.
- [ ] **Step 4: Run** `git diff --check`, package tests, and `catkin_test_results --verbose`.
- [ ] **Step 5: Commit** the generated MP4 with `docs: add LiDAR comparison animation`.
