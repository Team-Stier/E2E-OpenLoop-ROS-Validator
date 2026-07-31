# IMU Orientation Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `horizontal_pkg` with one runtime Python module that reads `/imu` orientation quaternions from a ROS1 bag and exports chronological roll/pitch CSV data.

**Architecture:** The source module contains pure quaternion and sample-consolidation functions plus a thin rosbag/CLI boundary. It preserves original acquisition stamps, resolves duplicate stamps deterministically, writes CSV atomically, and does not run ROS topics or alter LiDAR/model inputs.

**Tech Stack:** Ubuntu, ROS1 Noetic, Catkin, Python 3 standard-library `unittest`, `rosbag`, `sensor_msgs`, Catkin nosetests.

## Global Constraints

- Runtime logic exists only in `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_analyzer.py`.
- Input is a user-supplied indexed ROS1 bag and configurable IMU topic, default `/imu`.
- Original `sensor_msgs/Imu.header.stamp` is the measurement time; receipt-time normalization is prohibited.
- Runtime output is terminal summary plus CSV; no PNG, ROS node, launch file, LiDAR, Kalman filter, or PointCloud rotation.
- Runtime dependencies are limited to ROS Noetic `rosbag`, `sensor_msgs`, and the Python standard library.
- Finite nonzero quaternions are normalized; norm below `1e-6`, non-finite components, and zero source stamps are rejected.
- Norm deviation greater than `1e-3` is counted as a warning without rejecting the normalized sample.
- Equal source stamps use last-message-wins; record-order reversals are counted; CSV rows are strictly chronological and unique.
- Generated CSV data is not committed.
- Work stays local on `feature/imu-orientation-analyzer`; do not push, merge, or open a PR.

---

## File Map

| File | Responsibility |
| --- | --- |
| `src/Validator_pkg/horizontal_pkg/package.xml` | Catkin and ROS dependency contract. |
| `src/Validator_pkg/horizontal_pkg/CMakeLists.txt` | Python setup and Catkin-native test target. |
| `src/Validator_pkg/horizontal_pkg/setup.py` | Install `horizontal_pkg` from `src/`. |
| `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/__init__.py` | Empty import-package marker. |
| `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_analyzer.py` | All runtime calculation and CLI behavior. |
| `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py` | Pure unit and fake-bag tests. |
| `.gitignore` | Ignore generated horizontal-analysis CSV files. |
| `README.md` | Document the offline command and limits. |

---

### Task 1: Catkin Package and Quaternion Math

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/package.xml`
- Create: `src/Validator_pkg/horizontal_pkg/CMakeLists.txt`
- Create: `src/Validator_pkg/horizontal_pkg/setup.py`
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/__init__.py`
- Create: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_analyzer.py`
- Create: `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py`

**Interfaces:**
- Consumes: quaternion components in ROS order `(x, y, z, w)`.
- Produces: `OrientationAngles(quaternion_norm, roll_rad, pitch_rad, normalization_warning)`.
- Produces: `quaternion_to_roll_pitch(x, y, z, w) -> OrientationAngles`.
- Raises: `InvalidQuaternionError` for non-finite and near-zero inputs.

- [ ] **Step 1: Create package metadata and failing tests**

Use this Catkin structure:

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(horizontal_pkg)
find_package(catkin REQUIRED COMPONENTS rosbag sensor_msgs)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS rosbag sensor_msgs)
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_imu_orientation_analyzer.py)
endif()
```

Use package format 2 with `catkin` as build tool, `rosbag` and `sensor_msgs`
as build/build-export/exec dependencies, and `python3-nose` as a test
dependency. Use `generate_distutils_setup(packages=["horizontal_pkg"],
package_dir={"": "src"})` in `setup.py`.

Write tests asserting identity quaternion gives zero roll/pitch, positive
10-degree roll is recovered, negative 10-degree pitch is recovered, `q` and
`-q` give equal angles, `(0,0,0,2)` is normalized with a warning, and zero or
NaN input raises `InvalidQuaternionError`.

- [ ] **Step 2: Run tests to prove the interface is missing**

Run:

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py
```

Expected: import or missing-symbol failure.

- [ ] **Step 3: Implement the minimal math API**

Define `AnalyzerError`, `InvalidQuaternionError`, and frozen
`OrientationAngles`. `quaternion_to_roll_pitch` must implement:

```python
values = tuple(float(value) for value in (x, y, z, w))
if not all(math.isfinite(value) for value in values):
    raise InvalidQuaternionError("quaternion contains a non-finite component")
norm = math.sqrt(sum(value * value for value in values))
if norm < 1e-6:
    raise InvalidQuaternionError("quaternion norm is smaller than 1e-6")
x, y, z, w = (value / norm for value in values)
roll = math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
sin_pitch = max(-1.0, min(1.0, 2.0 * (w*y - z*x)))
pitch = math.asin(sin_pitch)
return OrientationAngles(norm, roll, pitch, abs(norm - 1.0) > 1e-3)
```

Keep `__init__.py` empty.

- [ ] **Step 4: Run tests and Catkin build**

Run the Task 1 `unittest` command, then:

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Expected: quaternion tests pass and Catkin builds `horizontal_pkg`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/Validator_pkg/horizontal_pkg
git commit -m "feat: add IMU quaternion conversion package"
```

---

### Task 2: Timestamp Consolidation and Atomic CSV

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_analyzer.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py`

**Interfaces:**
- Consumes: `RawOrientationSample` values in bag record order.
- Produces: `consolidate_samples(samples) -> AnalysisResult`.
- Produces: `write_csv_atomic(rows, output_path) -> None`.

- [ ] **Step 1: Add failing sample-policy tests**

Define tests with source stamps `[20, 10, 20, 0, 30]`: the second stamp
counts one reversal, the final valid stamp 20 replaces the first stamp 20,
zero stamp is omitted, and stamp 30 with zero quaternion is omitted. Assert
output stamps `[10, 20]`, replacement sequence number preservation, total
count 5, duplicate count 1, reversal count 1, zero count 1, invalid count 1,
and sorted frame IDs.

Assert the CSV header exactly equals:

```text
header_stamp_ns,elapsed_s,bag_record_stamp_ns,seq,frame_id,qx,qy,qz,qw,quaternion_norm,roll_rad,pitch_rad,roll_deg,pitch_deg
```

Assert elapsed time starts at zero, stamps strictly increase, a successful
write replaces an old destination, and a forced serialization failure keeps
the previous destination unchanged without leaving a temporary file.

- [ ] **Step 2: Run the selected tests and confirm failure**

Run the complete package `unittest` file. The existing quaternion cases pass
while the new consolidation and CSV cases fail.

Expected: missing dataclass/function failures.

- [ ] **Step 3: Implement immutable sample contracts**

Add frozen `RawOrientationSample`, `OrientationRow`, `AnalysisStats`, and
`AnalysisResult` dataclasses with the fields specified in the design. In
`consolidate_samples`, track the previous record-order source stamp, validate
each quaternion, place valid rows into a dictionary keyed by source stamp,
and return a stamp-sorted tuple. Count duplicate replacement only for a valid
overwrite. Raise `AnalyzerError("no valid IMU orientation samples")` when no
row remains.

- [ ] **Step 4: Implement atomic CSV output**

Create the parent directory, then use `tempfile.NamedTemporaryFile` in that
directory, `csv.DictWriter`, `flush`, `os.fsync`, and finally `os.replace`.
On any exception, unlink the temporary path and re-raise. Compute `elapsed_s`
from integer nanoseconds and serialize floating values with `repr`.

- [ ] **Step 5: Run all package tests**

Run the Task 1 `unittest` command. Expected: all math, sample-policy, and CSV tests
pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/Validator_pkg/horizontal_pkg
git commit -m "feat: export chronological IMU attitude CSV"
```

---

### Task 3: Rosbag CLI and Actual Recording Verification

**Files:**
- Modify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/imu_orientation_analyzer.py`
- Modify: `src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Produces: `analyze_bag(bag_path, imu_topic, bag_factory=None) -> AnalysisResult`.
- Produces: `format_summary(result, bag_path, imu_topic, output_path) -> str`.
- Produces: `main(argv=None) -> int` for module execution.

- [ ] **Step 1: Add failing fake-bag and CLI tests**

Build fake time/header/orientation/IMU values and a fake bag context manager.
Assert two fake IMU messages become two rows. Assert an empty topic raises
`AnalyzerError("topic /imu was not found in input.bag")`; an incompatible
`_type` error names `sensor_msgs/Imu`; a missing bag returns exit code 2; and
a successful injected-bag CLI writes a CSV and returns 0.

- [ ] **Step 2: Run selected tests and confirm failure**

Run the complete package `unittest` file. The existing cases pass while the
new bag, CLI, and summary cases fail.

Expected: missing bag/CLI functions.

- [ ] **Step 3: Implement bag reading and CLI**

Import `rosbag` lazily when no fake factory is injected. Check `Path.is_file`
before opening a real bag. Call `read_messages(topics=[imu_topic])`, require
`message._type == "sensor_msgs/Imu"`, convert both times with `to_nsec()`,
and construct `RawOrientationSample`. Distinguish missing topic from an
existing topic with no valid samples.

Use `argparse` with required `--bag`, required `--output-csv`, and optional
`--imu-topic` default `/imu`. Catch expected analyzer, bag, and filesystem
errors, print `error: ...` to stderr, return 2, and return 0 only after atomic
CSV completion. Use `statistics.median` for terminal min/median/max output and
finish with `raise SystemExit(main())`.

- [ ] **Step 4: Document and ignore generated output**

Add `records/horizontal/*.csv` to `.gitignore`. Add a README section showing:

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
python3 -m horizontal_pkg.imu_orientation_analyzer \
  --bag /home/stier/Rosbag/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T181003KST.bag \
  --output-csv records/horizontal/imu_roll_pitch.csv
```

State that the tool is offline, publishes no topics, rotates no PointCloud,
and does not yet validate the angles as compensation inputs.

- [ ] **Step 5: Run package and Catkin verification**

```bash
source /opt/ros/noetic/setup.bash
PYTHONPATH=src/Validator_pkg/horizontal_pkg/src \
  /usr/bin/python3 src/Validator_pkg/horizontal_pkg/test/test_imu_orientation_analyzer.py
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make run_tests_horizontal_pkg -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_test_results --verbose
```

Expected: all package tests and Catkin build/test targets pass.

- [ ] **Step 6: Run the inspected 2.1 GB bag smoke test**

Execute the documented module command with output
`/tmp/imu_roll_pitch.csv`. Expected terminal evidence includes 14,417 total
messages, one record-order reversal, finite roll/pitch, and a nonempty unique
chronological CSV. Verify with a short read-only Python assertion that stamps
equal `sorted(set(stamps))` and every angle is finite. Record actual duplicate
and unique-row counts rather than predicting them. Remove the temporary CSV.

- [ ] **Step 7: Run relevant repository regressions**

Run the existing validator-inference package-layout/install-contract tests
with the repository's Noetic-compatible environment. Expected: sibling Catkin
package discovery remains valid and no preprocessing/inference contract
changes.

- [ ] **Step 8: Commit Task 3**

```bash
git add .gitignore README.md src/Validator_pkg/horizontal_pkg
git commit -m "feat: analyze IMU roll and pitch from rosbag"
```

---

## Completion Evidence

Before reporting completion, capture the clean feature branch, the local
commits after `origin/main`, unit/Catkin/regression totals, system-Python
build result, actual bag summary and CSV row count, confirmation that no ROS
topic/LiDAR/model input changed, and the remaining unverified quaternion
reference-frame assumption.
