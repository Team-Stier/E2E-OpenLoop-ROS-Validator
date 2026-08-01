# Offline IMU-Based LiDAR Horizontalization Design

## Goal

Add an offline `horizontal_pkg` tool that reads one ROS1 bag, filters the bag's
IMU roll/pitch, completely levels every valid `/lidar3D` cloud, and writes a new
bag containing all original messages plus `/lidar3D/horizontal`. Align the
repository's shared LiDAR extrinsic with the current bag TF.

## Scope

The runtime module is:

`src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`

It is an offline transformation tool. It does not start a ROS node, publish a
live topic, alter the input bag, connect the derived cloud to BEV, or replace
any model input. The output bag is a complete copy of the input bag with one
additional PointCloud2 topic.

## Source Topics and Frames

The default source contracts are:

- IMU: `/imu`, `sensor_msgs/Imu`, frame `IMU-1`;
- LiDAR: `/lidar3D`, `sensor_msgs/PointCloud2`, frame `Lidar3D-4`;
- transforms: `/tf`, `tf2_msgs/TFMessage`;
- output: `/lidar3D/horizontal`, `sensor_msgs/PointCloud2`, frame
  `Lidar3D-4-horizontal`.

The tool obtains both sensor extrinsics from the bag's `/tf`. For the current
bag, the expected median transforms are:

```text
T_base_from_imu:
  parent: /base_link
  child: IMU-1
  translation: [1.5, 0.0, 0.5] m
  quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]

T_base_from_lidar:
  parent: /base_link
  child: Lidar3D-4
  translation: [1.5, 0.0, 1.25] m
  quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
```

Therefore `T_imu_from_lidar` has translation `[0.0, 0.0, 0.75] m` and identity
rotation. The runtime must derive this relation from `/tf`, not hard-code it.
It validates parent/child names, finite nonzero quaternions, translation
stability, and relative rotation. The current bag's measured values are the
acceptance reference, while small simulator floating-point variation is
allowed.

## Shared Profile Update

Update every canonical `T_base_from_lidar` declaration in
`src/Validator_pkg/config/profile.json` from translation `[2.0, 0.0, 1.5]` to
`[1.5, 0.0, 1.25]`, retaining identity rotation. Update direct documentation
and test expectations that describe or assert the shared transform. Do not
modify original MORAI sensor, map, route, or bag files.

This profile change affects TFv6 preprocessing, model target conversion,
waypoint export, and visualization consumers of the shared transform. It does
not itself enable horizontalized clouds as model inputs.

## IMU Filtering and Time Alignment

Read all `/imu` messages and consolidate them by original nonzero
`header.stamp`: reject invalid quaternions and keep the last message for a
duplicate stamp, matching `imu_orientation_analyzer.py`. Convert each valid
quaternion to roll, pitch, and yaw using the ROS xyzw convention.

Reuse `imu_filtering.filter_orientation` to apply the default second-order,
4.0 Hz, zero-phase Butterworth filter to roll and pitch. Yaw is not filtered
because leveling removes yaw from the compensation transform.

For each LiDAR `header.stamp`, linearly interpolate filtered roll and pitch
between bracketing IMU stamps. Do not extrapolate beyond IMU coverage. Reject a
pair when either bracket is more than `0.10 s` from the LiDAR stamp or the
bracket span exceeds `0.10 s`. Exact timestamp matches are accepted. Bag record
time is used only to preserve output message ordering; sensor `header.stamp` is
the measurement-time authority.

## Horizontalization Geometry

Use the ROS IMU orientation convention:

```text
R_world_from_imu = Rz(yaw) * Ry(pitch) * Rx(roll)
```

Define a level frame that shares the current yaw with the IMU but has zero roll
and pitch:

```text
R_world_from_level = Rz(yaw)
```

The point rotation from raw LiDAR coordinates into the horizontal output is:

```text
R_level_from_lidar =
    inverse(R_world_from_level)
    * R_world_from_imu_filtered
    * R_imu_from_lidar

p_horizontal = R_level_from_lidar * p_lidar
```

For the current identity sensor-relative rotation this reduces to
`Ry(filtered_pitch) * Rx(filtered_roll)`. The transform rotates vectors about
the LiDAR origin, so the 0.75 m IMU-to-LiDAR translation is not applied to raw
point coordinates. That translation remains required for extrinsic validation
and any later common-frame or BEV integration.

Complete leveling intentionally removes both vehicle attitude and genuine road
slope. The original `/lidar3D` remains present so downstream code can choose
the correct representation.

## PointCloud2 Contract

Accept clouds only when `x`, `y`, and `z` are finite-offset, count-one
`FLOAT32` fields, `point_step` and `row_step` are consistent, and the payload
length matches the declared layout. Preserve:

- `header.stamp`, `height`, `width`, fields, endianness, point/row steps,
  density, and every non-XYZ byte;
- intensity and any additional fields without reinterpretation;
- NaN XYZ points as NaN.

Copy the binary payload and update only x/y/z values. Reject unsupported or
malformed layouts instead of silently rebuilding a different schema.

## Output Bag and Failure Behavior

The input and output paths must differ. Write the output with LZ4 compression
into a temporary file in the destination directory and atomically replace the
requested output only after the entire bag closes successfully.

For every input record, first write the original message with its original
topic and bag record time. When a valid `/lidar3D` message is processed, write
the derived `/lidar3D/horizontal` message at the same bag record time and retain
the original sensor `header.stamp`. A cloud without valid IMU alignment is
copied only as the original and counted as skipped. Missing required topics,
invalid TF, unsupported PointCloud fields, output write failures, and zero
successfully horizontalized clouds cause a nonzero CLI exit without replacing
an existing destination.

The derived `Lidar3D-4-horizontal` frame is intentionally not inserted into
`/tf` in this work. It names the transformed coordinate contract honestly;
later BEV/RViz integration will define how that derived frame is consumed.

## Command-Line Interface

```bash
/usr/bin/python3 -m horizontal_pkg.lidar_horizontal \
  --input-bag /absolute/path/input.bag \
  --output-bag /absolute/path/input_horizontal.bag \
  --imu-topic /imu \
  --lidar-topic /lidar3D \
  --tf-topic /tf \
  --output-topic /lidar3D/horizontal \
  --output-frame Lidar3D-4-horizontal \
  --cutoff-hz 4.0 \
  --max-imu-gap-s 0.10
```

The topic, frame, cutoff, and maximum gap arguments have the displayed
defaults. Successful output reports IMU input/valid/duplicate counts, LiDAR
input/written/skipped counts, extracted extrinsics, and elapsed processing
time. Expected validation and filesystem errors print `error: ...` and return
exit code `2`.

## Validation

Tests must cover:

- roll-only, pitch-only, and combined synthetic planes becoming horizontal;
- correct sign and `Ry * Rx` order with hand-derived point expectations;
- non-identity IMU-to-LiDAR rotation composition;
- exact and bracketed timestamp alignment, gap rejection, and no
  extrapolation;
- duplicate/invalid IMU handling;
- extraction and validation of bag TF extrinsics;
- binary PointCloud2 XYZ transformation with all non-XYZ bytes unchanged;
- malformed field/layout rejection and NaN preservation;
- original-message copy plus derived-topic ordering;
- missing topic, zero-output, temporary-write, and atomic-replace failures;
- CLI success and expected-error exit codes;
- shared profile, documentation, and consumer test consistency.

After unit and Catkin tests pass, run the tool on the current July 31 bag.
Verify topic counts with `rosbag info`, verify output headers and fields, and
calculate ground-plane normal angle and residual before and after leveling on
timestamp-aligned samples. Report actual runtime and output size. A flatter
plane demonstrates geometric leveling only; it does not establish improved E2E
training or closed-loop performance.
