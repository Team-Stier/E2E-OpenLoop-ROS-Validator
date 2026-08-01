#!/usr/bin/env python3

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import json
import io
from contextlib import redirect_stdout

import numpy as np
import rosbag
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

    def test_pair_summary_counts_unmatched_and_rejects_duplicates(self):
        summary = animation.summarize_topic_stamps((1, 2, 3), (2, 3, 4))
        self.assertEqual(summary.paired_stamps_ns, (2, 3))
        self.assertEqual(summary.raw_unmatched_count, 1)
        self.assertEqual(summary.horizontal_unmatched_count, 1)
        with self.assertRaisesRegex(animation.AnimationError, "duplicate"):
            animation.summarize_topic_stamps((1, 1), (1,))


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

    def test_legend_handles_are_forced_opaque(self):
        handles = [mock.Mock(), mock.Mock()]
        legend = mock.Mock(legend_handles=handles)
        legend.get_texts.return_value = []
        animation._style_legend(legend)
        for handle in handles:
            handle.set_alpha.assert_called_once_with(1.0)
            handle.set_sizes.assert_called_once_with([32.0])


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

    def test_success_atomically_replaces_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.mp4"
            output.write_bytes(b"existing")
            def fake_encode(_input, temporary, *_args):
                temporary.write_bytes(b"video")
                return 1
            with mock.patch.object(animation, "_encode_selected_pairs", side_effect=fake_encode), mock.patch.object(animation, "_validate_video"):
                stats = animation.encode_animation("/virtual/input.bag", output, pair_stamps_ns=(1,))
            self.assertEqual(output.read_bytes(), b"video")
            self.assertEqual(stats.rendered_frames, 1)
            self.assertEqual(stats.raw_unmatched_count, 0)
            self.assertEqual(stats.horizontal_unmatched_count, 0)

    def test_cli_prints_unmatched_pairing_summary(self):
        stats = animation.AnimationStats(
            raw_frames=10,
            horizontal_frames=9,
            paired_frames=8,
            raw_unmatched_count=2,
            horizontal_unmatched_count=1,
            rendered_frames=8,
            fps=10,
            duration_s=0.8,
            output_bytes=123,
        )
        output = io.StringIO()
        with mock.patch.object(animation, "encode_animation", return_value=stats), redirect_stdout(output):
            result = animation.main(["--input-bag", "in.bag", "--output-mp4", "out.mp4"])
        self.assertEqual(result, 0)
        self.assertIn("raw_unmatched=2", output.getvalue())
        self.assertIn("horizontal_unmatched=1", output.getvalue())

    def test_ffprobe_checks_dimensions_frames_and_duration(self):
        valid = {"streams": [{"codec_name": "h264", "pix_fmt": "yuv420p", "width": 640, "height": 360, "nb_frames": "3", "duration": "0.3"}]}
        completed = mock.Mock(returncode=0, stdout=json.dumps(valid), stderr="")
        with mock.patch.object(animation.subprocess, "run", return_value=completed):
            animation._validate_video(Path("x.mp4"), animation.RenderConfig(width_px=640, height_px=360), 3)
        invalid = dict(valid)
        invalid["streams"] = [dict(valid["streams"][0], width=1280)]
        completed.stdout = json.dumps(invalid)
        with mock.patch.object(animation.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(animation.AnimationError, "metadata"):
                animation._validate_video(Path("x.mp4"), animation.RenderConfig(width_px=640, height_px=360), 3)

    def test_missing_ffprobe_is_reported_as_animation_error(self):
        with mock.patch.object(animation.subprocess, "run", side_effect=FileNotFoundError("ffprobe")):
            with self.assertRaisesRegex(animation.AnimationError, "ffprobe.*not available"):
                animation._validate_video(Path("x.mp4"), animation.RenderConfig(), 1)

    def test_missing_ffmpeg_is_reported_as_animation_error(self):
        with mock.patch.object(animation.subprocess, "Popen", side_effect=FileNotFoundError("ffmpeg")):
            with self.assertRaisesRegex(animation.AnimationError, "ffmpeg.*not available"):
                animation._encode_selected_pairs(
                    "/virtual/input.bag", Path("x.mp4"), (1,),
                    "/lidar3D", "/lidar3D/horizontal", 10,
                    animation.RenderConfig(),
                )

    def test_broken_ffmpeg_pipe_is_cleaned_up(self):
        process = mock.Mock()
        process.stdin.closed = False
        process.stdin.write.side_effect = BrokenPipeError("closed")
        process.poll.return_value = None
        bag = mock.MagicMock()
        bag.__enter__.return_value = bag
        bag.read_messages.return_value = [
            ("/lidar3D", cloud(1), None),
            ("/lidar3D/horizontal", cloud(1, "Lidar3D-4-horizontal", 0.1), None),
        ]
        with mock.patch.object(animation.subprocess, "Popen", return_value=process), mock.patch("rosbag.Bag", return_value=bag):
            with self.assertRaisesRegex(animation.AnimationError, "closed its input"):
                animation._encode_selected_pairs(
                    "/virtual/input.bag", Path("x.mp4"), (1,),
                    "/lidar3D", "/lidar3D/horizontal", 10,
                    animation.RenderConfig(),
                )
        process.kill.assert_called_once()
        process.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
