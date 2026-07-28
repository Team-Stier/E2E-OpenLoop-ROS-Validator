# LiDAR Road-Leveling Design

## Goal

Publish a diagnostic point cloud whose fitted road plane is `z=0`, so vehicle
roll and pitch move the road-relative coordinate frame instead of changing the
apparent height of the same obstacle.

## Scope

- Keep the rosbag input `/lidar3D` unchanged.
- Keep all model and inference subscriptions unchanged.
- Add `/lidar3D/leveled` as a diagnostic `sensor_msgs/PointCloud2` output.
- Reuse the existing confidence-gated RANSAC plane estimate in
  `lidar_attitude_debugger.py`.
- Show the new cloud in the existing LiDAR attitude RViz configuration.
- Do not commit or push these local changes.

This is frame-level road-plane compensation. It does not deskew motion that
occurs inside one LiDAR scan.

## Coordinate Transform

The accepted RANSAC plane in `base_link` is

```text
n dot p + d = 0
```

where `n` is normalized and points upward. The leveled road axes are:

```text
z_road = n
x_road = normalize([1, 0, 0] - dot([1, 0, 0], n) * n)
y_road = normalize(cross(z_road, x_road))
```

For every point `p` already converted from the LiDAR sensor frame to
`base_link`, publish:

```text
p_leveled = [
  dot(x_road, p),
  dot(y_road, p),
  dot(z_road, p) + d
]
```

The projection of the existing `base_link` origin onto the fitted plane is
therefore `(x=0, y=0, z=0)` in the output. All fitted ground points have
`z approximately 0`, and points above the road retain their signed
road-relative height.

## Validity and Failure Behavior

The output is published only when the current plane passes the existing gates:

- minimum candidate count;
- minimum inlier ratio;
- maximum tilt;
- maximum inlier RMS.

An invalid or empty frame publishes no `/lidar3D/leveled` message. This avoids
turning a poor plane estimate into a plausible but incorrect obstacle cloud.
Existing diagnostic markers continue to report the rejection reason.

## ROS Contract

| Direction | Topic | Type | Frame |
| --- | --- | --- | --- |
| Input | `/lidar3D` | `sensor_msgs/PointCloud2` | rosbag sensor frame |
| Existing diagnostic | `/lidar/debug/colored_points` | `sensor_msgs/PointCloud2` | `base_link` |
| New diagnostic | `/lidar3D/leveled` | `sensor_msgs/PointCloud2` | `base_link` |

The new message contains finite `x`, `y`, and `z` float fields. It represents a
virtual road-leveled point set in the diagnostic `base_link` display frame; it
must not replace the raw model input without a separately reviewed model
preprocessing change.

## Verification

- Unit-test the transform with a tilted plane and known obstacle height.
- Unit-test horizontal-plane identity and malformed-normal rejection.
- Unit-test the generated `PointCloud2` header and XYZ values.
- Run the package test suite and a catkin build with `/usr/bin/python3`.
- Replay the local rosbag and verify `/lidar3D/leveled` type, frame, frequency,
  and fitted-ground residual in live ROS.
