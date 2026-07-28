# LiDAR Nominal and Current Axes Design

## Goal

Compare the nominal LiDAR XY axes against XY-direction lines tilted by the
current LiDAR road-relative roll and pitch. All lines must share the configured
LiDAR sensor origin so angular differences are visually meaningful.

## Coordinate Contract

The common origin is the LiDAR translation loaded from `0716demo.json`.
For `/lidar3D`, the current configuration gives:

```text
O_lidar = (2.0, 0.0, 1.5) in base_link
```

The nominal yellow axes are:

```text
nominal X: O_lidar -> O_lidar + 25 * (1, 0, 0)
nominal Y: O_lidar - 15 * (0, 1, 0)
           through O_lidar
           to O_lidar + 15 * (0, 1, 0)
```

The marker frame remains `base_link`; “LiDAR-based” means the axes use the
LiDAR calibration origin rather than the `base_link` origin. The configured
LiDAR mount rotation is currently `(roll=0, pitch=0, yaw=0)`, so its nominal
XY directions match the `base_link` XY directions.

## Current LiDAR-Derived Directions

Use the current confidence-gated RANSAC plane estimate directly. Do not use TF
roll/pitch for these markers.

For the upward estimated normal `n = (a, b, c)`, construct plane-parallel
directions:

```text
current X direction = normalize((c, 0, -a))
current Y direction = normalize((0, c, -b))
```

Both directions are translated to `O_lidar`. This preserves a common origin:

```text
current X: O_lidar -> O_lidar + 25 * current_X_direction
current Y: O_lidar - 15 * current_Y_direction
           through O_lidar
           to O_lidar + 15 * current_Y_direction
```

Pitch therefore appears primarily as separation between nominal and current X
lines; roll appears primarily as separation between nominal and current Y
lines.

The current axes use only the fitted plane direction and are translated to
`O_lidar`. The fitted offset is not visualized.

## Marker Contract

Publish all markers through the existing `/lidar/debug/markers`
`visualization_msgs/MarkerArray`:

- Marker ID `5`: nominal X and Y lines, one `LINE_LIST`, yellow
  RGBA `(1.0, 1.0, 0.0, 1.0)`, width `0.15 m`.
- Marker ID `6`: current X and Y lines, one `LINE_LIST`, red
  RGBA `(1.0, 0.0, 0.0, 1.0)`, width `0.12 m`.

Use the incoming LiDAR timestamp and `base_link` frame. Existing marker IDs
`2` through `4`, topics, messages, launch files, and RViz displays remain
unchanged. The former translucent blue marker ID `1` is removed.

## Invalid Estimate Behavior

Always publish the nominal yellow axes. Publish marker ID `6` only when the
RANSAC plane is valid. The existing `DELETEALL` marker removes
current geometry immediately when the estimate becomes invalid, preventing
stale attitude visualization.

## Verification

- Add pure unit tests for horizontal and known-angle direction construction.
- Update the marker test to assert the calibrated common origin, endpoints,
  widths, colors, and marker IDs.
- Add an invalid-estimate test confirming only the nominal axes remain.
- Verify test-first failures before modifying production code.
- Run the full catkin build and test suite.
- Restart the local `bev` session and inspect live marker IDs `5` and `6`.
- Confirm translucent blue marker ID `1` and translucent red marker ID `7`
  are absent.

## Scope

This change affects visualization only. It does not alter point-cloud
coordinates, plane fitting, roll/pitch estimation, red z-band classification,
BEV images, rosbag playback, or any existing ROS interface. Work remains local
and uncommitted.
