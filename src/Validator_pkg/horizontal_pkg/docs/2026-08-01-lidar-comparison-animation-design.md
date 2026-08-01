# LiDAR Horizontalization Comparison Animation Design

## Goal

Create a reproducible offline MP4 that makes the geometric difference between
`/lidar3D` and `/lidar3D/horizontal` easy to inspect. All implementation,
tests, and generated documentation assets remain inside `horizontal_pkg`.

## Input and synchronization

- Read the completed horizontalized rosbag offline.
- Pair `/lidar3D` and `/lidar3D/horizontal` using identical message header
  timestamps.
- Reject incompatible PointCloud2 layouts and skip unmatched frames with an
  explicit summary.
- Use the matching point indices from each pair so the overlay compares the
  same physical returns.

## Visual layout

The video uses a dark background and two synchronized panels:

1. Side view (X-Z), which emphasizes pitch compensation.
2. Front view (Y-Z), which emphasizes roll compensation.

Raw points use high-luminance orange-red (`#FF5A36`) and horizontalized points
use bright cyan (`#16E0E0`). Points are small and semi-transparent; the
horizontalized layer is rendered after the raw layer. A compact legend uses
opaque color samples. Axes, grids, and text use neutral gray/white so neither
data color is confused with the background.

Each frame shows the source timestamp, frame number, and the number of plotted
point pairs. Axis limits remain fixed for the entire animation to avoid a false
impression of motion caused by auto-scaling.

## Sampling and output

- Process the complete 308-second drive.
- Encode at 10 FPS and select input frames uniformly across the drive, producing
  a practical overview rather than a real-time-duration diagnostic recording.
- Apply deterministic point downsampling for readable rendering and bounded
  generation time.
- Write the final artifact to
  `horizontal_pkg/docs/lidar_horizontal_comparison.mp4`.
- Keep temporary frames outside the repository and remove them after successful
  encoding.

## Implementation boundary

Add one offline module under `horizontal_pkg/src/horizontal_pkg` and its tests
under `horizontal_pkg/test`. Do not change the validator pipeline, BEV nodes,
other ROS packages, or either input rosbag.

## Failure handling and verification

- Fail clearly when the bag, required topics, PointCloud2 XYZ fields, or video
  encoder are unavailable.
- Never leave a partial final MP4; generate through a temporary path and replace
  the destination only after successful encoding.
- Unit-test timestamp pairing, deterministic sampling, color constants, fixed
  view limits, and cleanup on encoder failure.
- Verify the generated MP4 with `ffprobe`, including codec, resolution, frame
  count, duration, and nonzero size; inspect representative frames visually.
