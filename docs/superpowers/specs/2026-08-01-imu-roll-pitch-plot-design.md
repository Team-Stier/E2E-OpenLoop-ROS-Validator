# IMU Roll/Pitch Plot Design

## Goal

Generate a reproducible PNG that shows how the offline IMU orientation analyzer's
roll and pitch results change over the recording. Store the image and its usage
notes under `horizontal_pkg/docs`.

The plot documents raw roll and pitch converted from `/imu` orientation
quaternions. It does not claim that LiDAR points have already been
horizontalized or that sensor-axis and compensation signs are validated.

## Inputs and outputs

- Input CSV: output of `horizontal_pkg.imu_orientation_analyzer`
- Required columns: `elapsed_s`, `roll_deg`, and `pitch_deg`
- Plot module: `horizontal_pkg.imu_orientation_plotter`
- Documentation image:
  `src/Validator_pkg/horizontal_pkg/docs/imu_roll_pitch.png`
- Documentation notes:
  `src/Validator_pkg/horizontal_pkg/docs/README.md`

The plotting command accepts explicit input and output paths so another bag's
CSV can regenerate the same presentation without changing source code.

## Plot layout

The PNG contains two vertically stacked time-series axes with a shared elapsed
time axis:

1. Roll in degrees
2. Pitch in degrees

Each axis includes a zero-degree reference line, grid, unit label, and legend.
The title identifies the result as raw IMU orientation. A compact summary shows
sample count, duration, and the minimum, median, and maximum of each angle.
Separate axes preserve pitch detail when roll and pitch have different ranges.

## Validation and error handling

The CSV loader rejects:

- a missing or empty input file;
- missing required columns;
- non-finite or non-numeric required values;
- decreasing elapsed timestamps.

The output parent directory is created when necessary. The command returns a
non-zero exit code with a concise error message for rejected input.

Unit tests cover valid CSV loading, schema rejection, timestamp rejection, and
PNG creation. The package test target and Catkin build are run with ROS Noetic's
`/usr/bin/python3`. The generated PNG is opened after creation to confirm that
both plots, labels, and statistics are readable.

## Dependencies and scope

Add the ROS Noetic runtime dependency `python3-matplotlib`. The analyzer's CSV
schema and quaternion conversion remain unchanged. No ROS node, topic,
PointCloud, BEV output, model input, or launch file changes in this work.

The checked-in PNG is generated from
`records/horizontal/imu_roll_pitch.csv`, currently containing 10,882 samples
over 307.153 seconds.
