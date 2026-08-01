#!/usr/bin/env python3

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from horizontal_pkg import lidar_comparison_animation as animation


def cloud(stamp_ns=1, frame="Lidar3D-4", delta_z=0.0):
    message = PointCloud2()
    message.header.stamp.secs = stamp_ns // 1_000_000_000
    message.header.stamp.nsecs = stamp_ns % 1_000_000_000
    message.header.frame_id = frame
    message.height = 1
    message.width = 4
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 64
    message.is_dense = False
    values = [
        (1.0, -1.0, -1.0 + delta_z, 10.0),
        (5.0, 0.0, -1.1 + delta_z, 20.0),
        (10.0, 2.0, -0.9 + delta_z, 30.0),
        (math.nan, 0.0, 0.0, 40.0),
    ]
    payload = bytearray(message.row_step)
    for index, value in enumerate(values):
        struct.pack_into("<ffff", payload, index * message.point_step, *value)
    message.data = bytes(payload)
    return message


class SamplingTest(unittest.TestCase):
    def test_extract_xyz_preserves_point_order_and_nan(self):
        result = animation.extract_xyz(cloud())
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_allclose(result[:3], [[1, -1, -1], [5, 0, -1.1], [10, 2, -0.9]])
        self.assertTrue(math.isnan(result[3, 0]))

    def test_uniform_indices_include_first_and_last(self):
        self.assertEqual(animation.uniform_indices(10, 4), (0, 3, 6, 9))
        self.assertEqual(animation.uniform_indices(3, 10), (0, 1, 2))

    def test_matching_indices_are_deterministic_and_require_finite_pairs(self):
        raw = animation.extract_xyz(cloud())
        horizontal = animation.extract_xyz(cloud(delta_z=0.1))
        self.assertEqual(
            tuple(animation.matching_point_indices(raw, horizontal, 2)),
            (0, 2),
        )

    def test_incompatible_cloud_layout_is_rejected(self):
        changed = cloud(frame="Lidar3D-4-horizontal")
        changed.point_step = 20
        with self.assertRaisesRegex(animation.AnimationError, "layout"):
            animation.validate_pair(cloud(), changed)


class RenderingTest(unittest.TestCase):
    def test_approved_colors_and_dark_background_are_fixed(self):
        self.assertEqual(animation.RAW_COLOR, "#FF5A36")
        self.assertEqual(animation.HORIZONTAL_COLOR, "#16E0E0")
        self.assertEqual(animation.BACKGROUND_COLOR, "#101318")

    def test_render_frame_returns_requested_rgb_shape(self):
        raw = animation.extract_xyz(cloud())
        horizontal = animation.extract_xyz(cloud(delta_z=0.1))
        config = animation.RenderConfig(width_px=640, height_px=360, dpi=100)
        frame = animation.render_frame(raw, horizontal, "frame 1", config)
        self.assertEqual(frame.shape, (360, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)


class AtomicOutputTest(unittest.TestCase):
    def test_ffmpeg_command_requests_h264_yuv420p_and_faststart(self):
        command = animation._ffmpeg_command(
            Path("output.mp4"),
            animation.RenderConfig(width_px=640, height_px=360),
            10,
        )
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertEqual(command[command.index("-movflags") + 1], "+faststart")

    def test_encoder_failure_preserves_destination_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.mp4"
            output.write_bytes(b"existing")
            before = set(Path(directory).iterdir())
            with mock.patch.object(
                animation,
                "_encode_selected_pairs",
                side_effect=animation.AnimationError("encoder failed"),
            ):
                with self.assertRaisesRegex(animation.AnimationError, "encoder failed"):
                    animation.encode_animation(
                        "/virtual/input.bag",
                        output,
                        pair_stamps_ns=(1,),
                    )
            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(set(Path(directory).iterdir()), before)


if __name__ == "__main__":
    unittest.main()
