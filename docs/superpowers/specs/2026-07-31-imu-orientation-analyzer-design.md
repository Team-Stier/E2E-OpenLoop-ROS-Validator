# IMU Orientation Analyzer Design

## Goal

Create a new ROS Noetic Catkin package named `horizontal_pkg` whose only
runtime logic reads `sensor_msgs/Imu.orientation` from a rosbag and exports
roll and pitch. This first milestone establishes the IMU-only baseline before
any PointCloud rotation, LiDAR ground fitting, Kalman fusion, ROS publisher,
or model-input change.

## Scope

The analyzer will:

- open a user-supplied ROS1 bag;
- read only a configurable IMU topic, defaulting to `/imu`;
- use each message's original `header.stamp` as its measurement time;
- validate and normalize the orientation quaternion;
- convert the quaternion to ROS-convention roll and pitch;
- write one chronological CSV row for each unique valid measurement stamp;
- print a deterministic terminal summary of input quality and angle ranges.

The analyzer will not:

- subscribe to or publish ROS topics;
- call `rosbag play`;
- replace any `header.stamp` with receipt time;
- read LiDAR or camera messages;
- estimate a ground plane;
- run a complementary or Kalman filter;
- rotate a PointCloud;
- remap an existing preprocessing or inference input;
- generate a PNG in this milestone.

## Package Layout

The package will live beside the existing Catkin packages:

```text
src/Validator_pkg/horizontal_pkg/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── src/
│   └── horizontal_pkg/
│       ├── __init__.py
│       └── imu_orientation_analyzer.py
└── test/
    └── test_imu_orientation_analyzer.py
```

`imu_orientation_analyzer.py` is the only runtime Python implementation.
`__init__.py` only marks the import package, `setup.py` installs it into the
Catkin Python path, and the test file is not installed as runtime code.

## Command-Line Contract

After a Catkin build and overlay source, the analyzer will run as a Python
module:

```bash
python3 -m horizontal_pkg.imu_orientation_analyzer \
  --bag /absolute/path/to/input.bag \
  --output-csv /absolute/path/to/imu_roll_pitch.csv
```

Supported arguments:

- `--bag`: required path to a readable, indexed ROS1 bag;
- `--output-csv`: required output path; the parent directory is created when
  absent;
- `--imu-topic`: optional input topic, default `/imu`.

The command returns zero only after the requested topic was found, at least
one valid unique sample was exported, and the CSV was closed successfully.
Missing bags, unreadable output paths, absent topics, incompatible message
types, or no valid samples return a nonzero exit code with a direct error.

## Quaternion Semantics and Conversion

The four orientation fields are a unit quaternion in ROS order
`(x, y, z, w)`. They are not individual Euler angles. A sample is invalid if
its source stamp is zero, any quaternion component is non-finite, or the
quaternion norm is smaller than `1e-6`.

Every finite nonzero quaternion is normalized before conversion. A norm whose
absolute deviation from one exceeds `1e-3` is counted as a normalization
warning but remains usable after normalization.

For normalized `(x, y, z, w)`, the analyzer computes:

```text
sin_roll = 2 * (w*x + y*z)
cos_roll = 1 - 2 * (x*x + y*y)
roll = atan2(sin_roll, cos_roll)

sin_pitch = clamp(2 * (w*y - z*x), -1, 1)
pitch = asin(sin_pitch)
```

Radians are the internal values. CSV degree columns use `180 / pi` and do not
alter the radian results.

## Timestamp and Duplicate Policy

The bag is read in record order, while the CSV is ordered by original
`header.stamp` nanoseconds. Bag record time is retained only for diagnostics;
it never replaces the sensor stamp.

- Equal source stamps represent duplicate measurements. The last valid
  message encountered in bag record order replaces earlier messages at the
  same source stamp.
- A source stamp smaller than the preceding record-order IMU stamp increments
  the reversal counter. The valid measurement is retained and later placed in
  chronological CSV order rather than being relabeled or silently deleted.
- Zero source stamps and invalid quaternions are counted and omitted.

This policy produces one deterministic row per valid unique acquisition time
without pretending bag receipt order is sensor time.

## CSV Contract

The CSV header is fixed:

```text
header_stamp_ns,elapsed_s,bag_record_stamp_ns,seq,frame_id,
qx,qy,qz,qw,quaternion_norm,roll_rad,pitch_rad,roll_deg,pitch_deg
```

`elapsed_s` is measured from the first exported source stamp. Quaternion
columns contain the original message values; angle columns are computed from
the normalized quaternion. Floating-point output uses enough precision to
round-trip Python `float` values.

## Terminal Summary

The successful command prints:

- bag path, IMU topic, and output path;
- total IMU messages read;
- valid unique rows written;
- duplicate replacements;
- record-order timestamp reversals;
- zero-stamp and invalid-quaternion counts;
- quaternion normalization-warning count;
- observed `frame_id` values;
- exported acquisition-time duration;
- minimum, median, and maximum roll and pitch in degrees.

No summary number will be described as sensor accuracy or ground-truth error.

## Dependencies

Runtime dependencies are limited to ROS Noetic `rosbag`, `sensor_msgs`, and
the Python standard library. NumPy, SciPy, OpenCV, Matplotlib, LiDAR libraries,
and new non-ROS packages are excluded.

The Catkin package uses `catkin_python_setup()` and declares its ROS runtime
dependencies in `package.xml`. It does not generate messages or install a ROS
node executable.

## Error Handling

The module will expose pure helpers for quaternion validation/conversion and
sample consolidation. The CLI boundary catches expected bag, filesystem, and
contract errors, writes a concise message to stderr, and returns nonzero.
Unexpected exceptions are not converted into a successful partial export.

CSV creation uses an atomic temporary file in the requested output directory.
The completed file replaces the destination only after every row is written;
an interrupted or failed run removes the temporary file and preserves any
previous destination.

## Verification

Unit tests will cover:

- identity quaternion produces zero roll and pitch;
- known positive and negative roll/pitch rotations;
- `q` and `-q` produce equal angles;
- non-unit finite quaternion normalization;
- zero-norm and non-finite quaternion rejection;
- pitch input clamping at the inverse-sine boundary;
- duplicate source stamp last-message-wins behavior;
- record-order reversal counting and chronological output;
- missing topic and empty-valid-sample failures;
- exact CSV header and source-stamp preservation.

Repository verification will use the Noetic system interpreter and Catkin
test path. A manual smoke run will use:

```text
/home/stier/Rosbag/
s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T181003KST.bag
```

The smoke acceptance requires 14,417 `/imu` messages to be observed, a
monotonically increasing unique-stamp CSV, finite roll/pitch values, and a
nonzero exit only if the current bag no longer matches that inspected
recording. Generated CSV data remains untracked.

## Follow-up Boundary

PointCloud horizontalization is a separate milestone. It may consume these
same quaternion conversion helpers after the IMU coordinate convention and
rotation sign are validated. LiDAR ground-plane estimation or IMU-LiDAR
fusion will be added only if an IMU-only compensation baseline demonstrates a
measurable need.
