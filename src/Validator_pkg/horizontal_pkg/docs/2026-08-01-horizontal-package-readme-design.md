# horizontal_pkg README Design

## Goal

Add a concise package-level README that lets a ROS1 user understand what
`horizontal_pkg` does, which bag topics and frames it consumes or creates, and
how to reproduce the offline LiDAR horizontalization and comparison animation.

## Location and scope

Create `src/Validator_pkg/horizontal_pkg/README.md`. Do not change source code,
ROS interfaces, configuration, generated media, or documentation outside
`horizontal_pkg`.

## Content

The README will contain:

1. A short package-purpose section covering IMU roll/pitch analysis, 4 Hz
   low-pass filtering, complete LiDAR roll/pitch leveling, and comparison-video
   generation.
2. A compact topic table for `/imu`, `/tf`, `/lidar3D`, and
   `/lidar3D/horizontal`, including direction, message type, and role.
3. A frame table for `base_link`, `IMU-1`, `Lidar3D-4`, and
   `Lidar3D-4-horizontal`.
4. A minimal offline data-flow diagram.
5. Copy-paste commands for creating the horizontalized bag and comparison MP4
   using `/usr/bin/python3` and the ROS Noetic environment.
6. Relative links to the existing raw/filtered IMU plots and LiDAR comparison
   MP4 under `docs/`.
7. Short operational cautions: this is offline bag processing, the correction
   rotates XYZ rather than removing range noise, complete leveling also removes
   real road grade, and unmatched timestamps are skipped and reported.

## Verification

- Confirm every documented topic, frame, default cutoff, alignment gap, FPS,
  frame count, and point cap against the current Python CLI definitions.
- Check all relative asset links exist.
- Run `git diff --check` and the existing package-layout test.
