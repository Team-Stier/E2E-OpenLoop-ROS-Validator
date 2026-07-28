#!/usr/bin/env python3
"""Unit tests for LiDAR ground-plane and attitude diagnostics."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree
import math
import struct
import unittest

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "lidar_attitude_debugger.py"
)
SPEC = spec_from_file_location("lidar_attitude_debugger_tested", SCRIPT_PATH)
DEBUGGER = module_from_spec(SPEC)
SPEC.loader.exec_module(DEBUGGER)


def normal_from_roll_pitch(roll_degrees, pitch_degrees):
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    return np.asarray(
        [
            -math.sin(pitch),
            math.cos(pitch) * math.sin(roll),
            math.cos(pitch) * math.cos(roll),
        ],
        dtype=np.float64,
    )


def synthetic_plane(
    roll_degrees,
    pitch_degrees,
    count=800,
    noise_std=0.01,
    seed=7,
):
    rng = np.random.default_rng(seed)
    normal = normal_from_roll_pitch(roll_degrees, pitch_degrees)
    x = rng.uniform(2.0, 25.0, count)
    y = rng.uniform(-10.0, 10.0, count)
    z = -(normal[0] * x + normal[1] * y) / normal[2]
    z += rng.normal(0.0, noise_std, count)
    return np.column_stack([x, y, z]).astype(np.float32)


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.fit_config = DEBUGGER.PlaneFitConfig(
            iterations=160,
            distance_threshold=0.08,
            min_points=100,
            min_inlier_ratio=0.60,
            max_tilt_degrees=15.0,
            max_rms=0.05,
        )

    def test_ground_band_includes_boundaries_only(self):
        points = np.asarray(
            [
                [1.0, 0.0, -0.1000],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.1000],
                [4.0, 0.0, -0.1001],
                [5.0, 0.0, 0.1001],
            ],
            dtype=np.float32,
        )

        mask = DEBUGGER.classify_ground_band(points, 0.10)

        self.assertEqual(mask.tolist(), [True, True, True, False, False])

    def test_horizontal_plane_recovers_zero_attitude_with_outliers(self):
        plane = synthetic_plane(0.0, 0.0)
        rng = np.random.default_rng(11)
        outliers = rng.uniform(
            low=[2.0, -10.0, 1.0],
            high=[25.0, 10.0, 4.0],
            size=(200, 3),
        ).astype(np.float32)

        estimate = DEBUGGER.fit_ground_plane_ransac(
            np.vstack([plane, outliers]),
            self.fit_config,
            np.random.default_rng(13),
        )

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertAlmostEqual(estimate.roll_degrees, 0.0, delta=0.15)
        self.assertAlmostEqual(estimate.pitch_degrees, 0.0, delta=0.15)
        self.assertGreater(estimate.inlier_ratio, 0.70)
        self.assertLess(estimate.rms, 0.03)

    def test_known_plane_recovers_road_relative_roll_and_pitch(self):
        points = synthetic_plane(3.0, -2.0, count=1000)

        estimate = DEBUGGER.fit_ground_plane_ransac(
            points,
            self.fit_config,
            np.random.default_rng(17),
        )

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertAlmostEqual(estimate.roll_degrees, 3.0, delta=0.2)
        self.assertAlmostEqual(estimate.pitch_degrees, -2.0, delta=0.2)

    def test_plane_is_invalid_when_candidate_count_is_too_small(self):
        estimate = DEBUGGER.fit_ground_plane_ransac(
            synthetic_plane(0.0, 0.0, count=30),
            self.fit_config,
            np.random.default_rng(19),
        )

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "insufficient_points")

    def test_plane_is_invalid_when_inlier_ratio_is_too_low(self):
        plane = synthetic_plane(0.0, 0.0, count=120)
        rng = np.random.default_rng(31)
        outliers = rng.uniform(
            low=[2.0, -10.0, 1.0],
            high=[25.0, 10.0, 5.0],
            size=(280, 3),
        ).astype(np.float32)
        config = DEBUGGER.PlaneFitConfig(
            iterations=160,
            distance_threshold=0.08,
            min_points=100,
            min_inlier_ratio=0.70,
            max_tilt_degrees=15.0,
            max_rms=0.05,
        )

        estimate = DEBUGGER.fit_ground_plane_ransac(
            np.vstack([plane, outliers]),
            config,
            np.random.default_rng(37),
        )

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "low_inlier_ratio")

    def test_plane_is_invalid_when_tilt_exceeds_limit(self):
        config = DEBUGGER.PlaneFitConfig(
            iterations=160,
            distance_threshold=0.08,
            min_points=100,
            min_inlier_ratio=0.60,
            max_tilt_degrees=10.0,
            max_rms=0.05,
        )

        estimate = DEBUGGER.fit_ground_plane_ransac(
            synthetic_plane(12.0, 0.0, count=800),
            config,
            np.random.default_rng(41),
        )

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "tilt_exceeded")

    def test_plane_is_invalid_when_inlier_rms_exceeds_limit(self):
        config = DEBUGGER.PlaneFitConfig(
            iterations=160,
            distance_threshold=0.15,
            min_points=100,
            min_inlier_ratio=0.60,
            max_tilt_degrees=15.0,
            max_rms=0.01,
        )

        estimate = DEBUGGER.fit_ground_plane_ransac(
            synthetic_plane(0.0, 0.0, count=800, noise_std=0.04),
            config,
            np.random.default_rng(43),
        )

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "high_rms")

    def test_quaternion_conversion_uses_ros_roll_pitch_convention(self):
        roll = math.radians(4.0)
        pitch = math.radians(-3.0)
        yaw = math.radians(20.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        quaternion = SimpleNamespace(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )

        result = DEBUGGER.quaternion_to_rpy_degrees(quaternion)

        np.testing.assert_allclose(result, [4.0, -3.0, 20.0], atol=1.0e-6)

    def test_ground_bev_renders_ground_red_and_other_returns_gray(self):
        geometry = DEBUGGER.BevGeometry(
            forward_range=10.0,
            lateral_range=10.0,
            resolution=1.0,
        )
        points = np.asarray(
            [
                [5.0, 0.0, 0.02],
                [4.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        image = DEBUGGER.render_ground_bev(
            points,
            np.asarray([True, False]),
            geometry,
        )

        self.assertEqual(image.shape, (10, 10, 3))
        self.assertTrue(np.any(np.all(image == [0, 0, 255], axis=2)))
        self.assertTrue(np.any(np.all(image == [150, 150, 150], axis=2)))

    def test_float32_xyz_cloud_uses_fast_decode_and_filters_invalid_points(self):
        message = PointCloud2()
        message.height = 1
        message.width = 4
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 64
        message.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        message.data = b"".join(
            struct.pack("<ffff", *point)
            for point in (
                (1.0, 2.0, 3.0, 0.5),
                (0.0, 0.0, 0.0, 0.0),
                (4.0, 5.0, float("nan"), 0.5),
                (-1.0, -2.0, -3.0, 0.5),
            )
        )

        with patch.object(
            DEBUGGER.point_cloud2,
            "read_points",
            side_effect=AssertionError("slow generator path used"),
        ):
            decoded = DEBUGGER.decode_point_cloud(message)

        np.testing.assert_allclose(
            decoded,
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        )

    def test_plane_axis_directions_are_unit_and_parallel_to_plane(self):
        self.assertTrue(hasattr(DEBUGGER, "plane_axis_directions"))
        normal = normal_from_roll_pitch(3.0, -2.0)

        x_direction, y_direction = DEBUGGER.plane_axis_directions(normal)

        self.assertAlmostEqual(np.linalg.norm(x_direction), 1.0)
        self.assertAlmostEqual(np.linalg.norm(y_direction), 1.0)
        self.assertAlmostEqual(np.dot(normal, x_direction), 0.0)
        self.assertAlmostEqual(np.dot(normal, y_direction), 0.0)
        self.assertAlmostEqual(x_direction[1], 0.0)
        self.assertAlmostEqual(y_direction[0], 0.0)

    def test_level_points_maps_tilted_road_to_zero_and_preserves_height(self):
        normal = np.asarray([0.0, 0.6, 0.8], dtype=np.float64)
        points = np.asarray(
            [
                [3.0, 1.84, -0.88],
                [3.0, 2.56, 0.08],
            ],
            dtype=np.float32,
        )

        leveled = DEBUGGER.level_points_to_plane(
            points,
            normal,
            offset=-0.4,
        )

        np.testing.assert_allclose(
            leveled,
            [
                [3.0, 2.0, 0.0],
                [3.0, 2.0, 1.2],
            ],
            atol=1.0e-6,
        )
        self.assertEqual(leveled.dtype, np.float32)
        self.assertTrue(leveled.flags.c_contiguous)

    def test_level_points_is_identity_for_base_z_zero_plane(self):
        points = np.asarray(
            [
                [1.0, -2.0, 0.0],
                [4.0, 3.0, 1.5],
            ],
            dtype=np.float32,
        )

        leveled = DEBUGGER.level_points_to_plane(
            points,
            normal=np.asarray([0.0, 0.0, 1.0]),
            offset=0.0,
        )

        np.testing.assert_array_equal(leveled, points)

    def test_level_points_rejects_zero_normal(self):
        with self.assertRaisesRegex(ValueError, "normal"):
            DEBUGGER.level_points_to_plane(
                np.zeros((1, 3), dtype=np.float32),
                normal=np.zeros(3),
                offset=0.0,
            )

    def test_publish_leveled_cloud_preserves_header_and_xyz_values(self):
        debugger = DEBUGGER.LidarAttitudeDebugger.__new__(
            DEBUGGER.LidarAttitudeDebugger
        )
        debugger.base_frame = "base_link"
        published = []
        debugger.leveled_cloud_publisher = SimpleNamespace(
            publish=published.append
        )
        source_header = DEBUGGER.Header(
            seq=17,
            stamp=DEBUGGER.rospy.Time.from_sec(12.5),
            frame_id="lidar_sensor",
        )
        points = np.asarray(
            [
                [1.0, -2.0, 0.0],
                [4.0, 3.0, 1.2],
            ],
            dtype=np.float32,
        )

        debugger.publish_leveled_cloud(points, source_header)

        self.assertEqual(len(published), 1)
        message = published[0]
        self.assertEqual(message.header.seq, 17)
        self.assertAlmostEqual(message.header.stamp.to_sec(), 12.5)
        self.assertEqual(message.header.frame_id, "base_link")
        self.assertEqual(
            [field.name for field in message.fields],
            ["x", "y", "z"],
        )
        decoded = list(
            DEBUGGER.point_cloud2.read_points(
                message,
                field_names=("x", "y", "z"),
                skip_nans=False,
            )
        )
        np.testing.assert_allclose(decoded, points, atol=1.0e-6)

    def test_invalid_plane_keeps_only_nominal_yellow_lidar_axes(self):
        debugger = DEBUGGER.LidarAttitudeDebugger.__new__(
            DEBUGGER.LidarAttitudeDebugger
        )
        debugger.geometry = DEBUGGER.BevGeometry(
            forward_range=25.0,
            lateral_range=30.0,
            resolution=0.05,
        )
        debugger.roi = DEBUGGER.GroundRoi()
        debugger.calibration = {
            "translation": np.asarray([2.0, 0.0, 1.5]),
        }
        published = []
        debugger.marker_publisher = SimpleNamespace(publish=published.append)
        header = DEBUGGER.Header()
        plane = DEBUGGER._invalid_plane("test", 0)

        debugger.publish_markers(header, None, None, plane, None)

        markers_by_id = {
            marker.id: marker for marker in published[0].markers
        }
        self.assertIn(5, markers_by_id)
        reference_axes = markers_by_id[5]
        self.assertEqual(reference_axes.type, DEBUGGER.Marker.LINE_LIST)
        self.assertEqual(reference_axes.header.frame_id, "base_link")
        self.assertAlmostEqual(reference_axes.scale.x, 0.15)
        self.assertEqual(
            [
                (point.x, point.y, point.z)
                for point in reference_axes.points
            ],
            [
                (2.0, 0.0, 1.5),
                (27.0, 0.0, 1.5),
                (2.0, -15.0, 1.5),
                (2.0, 15.0, 1.5),
            ],
        )
        self.assertEqual(
            (
                reference_axes.color.r,
                reference_axes.color.g,
                reference_axes.color.b,
                reference_axes.color.a,
            ),
            (1.0, 1.0, 0.0, 1.0),
        )
        self.assertTrue({1, 6, 7, 8, 9}.isdisjoint(markers_by_id))

    def test_valid_plane_publishes_axes_without_faint_floor_planes(self):
        debugger = DEBUGGER.LidarAttitudeDebugger.__new__(
            DEBUGGER.LidarAttitudeDebugger
        )
        debugger.geometry = DEBUGGER.BevGeometry(
            forward_range=25.0,
            lateral_range=30.0,
            resolution=0.05,
        )
        debugger.roi = DEBUGGER.GroundRoi()
        debugger.calibration = {
            "translation": np.asarray([2.0, 0.0, 1.5]),
        }
        published = []
        debugger.marker_publisher = SimpleNamespace(publish=published.append)
        header = DEBUGGER.Header()
        plane = DEBUGGER.PlaneEstimate(
            valid=True,
            reason="valid",
            normal=np.asarray([0.0, 0.0, 1.0]),
            offset=0.0,
            roll_degrees=0.0,
            pitch_degrees=0.0,
            inlier_count=800,
            candidate_count=1000,
            inlier_ratio=0.8,
            rms=0.03,
        )

        debugger.publish_markers(
            header,
            None,
            None,
            plane,
            np.asarray([0.0, 0.0]),
        )

        markers_by_id = {
            marker.id: marker for marker in published[0].markers
        }
        self.assertTrue({5, 6}.issubset(markers_by_id))
        self.assertTrue({1, 7, 8, 9}.isdisjoint(markers_by_id))

        current_axes = markers_by_id[6]
        self.assertEqual(current_axes.type, DEBUGGER.Marker.LINE_LIST)
        self.assertAlmostEqual(current_axes.scale.x, 0.12)
        self.assertEqual(
            (
                current_axes.color.r,
                current_axes.color.g,
                current_axes.color.b,
                current_axes.color.a,
            ),
            (1.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(
            [
                (point.x, point.y, point.z)
                for point in current_axes.points
            ],
            [
                (2.0, 0.0, 1.5),
                (27.0, 0.0, 1.5),
                (2.0, -15.0, 1.5),
                (2.0, 15.0, 1.5),
            ],
        )



class TemporalAndStatusTest(unittest.TestCase):
    @staticmethod
    def quaternion(roll_degrees=0.0, pitch_degrees=0.0, yaw_degrees=0.0):
        roll = math.radians(roll_degrees)
        pitch = math.radians(pitch_degrees)
        yaw = math.radians(yaw_degrees)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        return SimpleNamespace(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )

    def test_tf_history_selects_latest_causal_sample_and_rejects_stale(self):
        history = DEBUGGER.CausalTransformHistory(maxlen=5)
        history.add(1.0, self.quaternion(roll_degrees=1.0))
        history.add(2.0, self.quaternion(roll_degrees=2.0))

        selected = history.select(1.5, max_age=0.6)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.stamp, 1.0)
        self.assertAlmostEqual(selected.rpy_degrees[0], 1.0)
        self.assertIsNone(history.select(0.5, max_age=1.0))
        self.assertIsNone(history.select(3.0, max_age=0.5))

    def test_tf_history_clears_old_loop_when_time_moves_backwards(self):
        history = DEBUGGER.CausalTransformHistory(maxlen=5)
        self.assertFalse(history.add(10.0, self.quaternion(roll_degrees=4.0)))

        did_reset = history.add(1.0, self.quaternion(roll_degrees=-2.0))

        self.assertTrue(did_reset)
        selected = history.select(1.0, max_age=0.1)
        self.assertAlmostEqual(selected.rpy_degrees[0], -2.0)
        self.assertEqual(len(history), 1)

    def test_median_filter_rejects_one_frame_angle_spike_and_can_reset(self):
        attitude_filter = DEBUGGER.AttitudeMedianFilter(window_size=3)
        attitude_filter.update(1.0, 2.0)
        attitude_filter.update(20.0, -20.0)
        filtered = attitude_filter.update(1.2, 2.2)

        np.testing.assert_allclose(filtered, [1.2, 2.0])

        attitude_filter.reset()
        np.testing.assert_allclose(
            attitude_filter.update(-3.0, 4.0),
            [-3.0, 4.0],
        )

    def test_status_lines_keep_world_and_road_relative_labels_distinct(self):
        tf_sample = DEBUGGER.TransformSample(
            stamp=10.0,
            quaternion=self.quaternion(roll_degrees=2.0, pitch_degrees=-1.0),
            rpy_degrees=np.asarray([2.0, -1.0, 0.0]),
        )
        plane = DEBUGGER.PlaneEstimate(
            valid=True,
            reason="valid",
            normal=normal_from_roll_pitch(0.5, -0.25),
            offset=0.0,
            roll_degrees=0.5,
            pitch_degrees=-0.25,
            inlier_count=800,
            candidate_count=1000,
            inlier_ratio=0.8,
            rms=0.03,
        )

        lines = DEBUGGER.format_status_lines(
            tf_sample=tf_sample,
            tf_age=0.02,
            plane=plane,
            filtered_lidar_rpy=np.asarray([0.5, -0.25]),
        )

        text = "\n".join(lines)
        self.assertIn("TF world", text)
        self.assertIn("LiDAR road-relative", text)
        self.assertIn("VALID", text)
        self.assertIn("inliers=800/1000", text)
        self.assertIn("RMS=0.030m", text)

    def test_status_lines_report_independent_invalid_reasons(self):
        plane = DEBUGGER.PlaneEstimate(
            valid=False,
            reason="low_inlier_ratio",
            normal=np.asarray([0.0, 0.0, 1.0]),
            offset=0.0,
            roll_degrees=float("nan"),
            pitch_degrees=float("nan"),
            inlier_count=20,
            candidate_count=200,
            inlier_ratio=0.1,
            rms=0.04,
        )

        lines = DEBUGGER.format_status_lines(
            tf_sample=None,
            tf_age=None,
            plane=plane,
            filtered_lidar_rpy=None,
        )

        text = "\n".join(lines)
        self.assertIn("TF world: INVALID", text)
        self.assertIn("LiDAR road-relative: INVALID (low_inlier_ratio)", text)


class PackageIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.package_root = Path(__file__).resolve().parents[1]

    def test_diagnostic_runtime_files_and_topics_are_installed(self):
        launch_path = (
            self.package_root / "launch" / "lidar_attitude_debug.launch"
        )
        rviz_path = (
            self.package_root / "rviz" / "lidar_attitude_debug.rviz"
        )
        self.assertTrue(launch_path.is_file())
        self.assertTrue(rviz_path.is_file())

        launch_root = ElementTree.parse(launch_path).getroot()
        nodes = {
            (node.get("pkg"), node.get("type"))
            for node in launch_root.findall("node")
        }
        self.assertIn(
            ("Validator_pkg", "lidar_attitude_debugger.py"),
            nodes,
        )
        self.assertIn(("rviz", "rviz"), nodes)

        rviz_text = rviz_path.read_text(encoding="utf-8")
        self.assertIn("/lidar/debug/ground_bev", rviz_text)
        self.assertIn("/lidar/debug/colored_points", rviz_text)
        self.assertIn("/lidar/debug/markers", rviz_text)
        self.assertIn("/lidar3D/leveled", rviz_text)
        self.assertIn("Fixed Frame: base_link", rviz_text)

    def test_manifest_and_cmake_declare_diagnostic_contract(self):
        package_root = ElementTree.parse(
            self.package_root / "package.xml"
        ).getroot()
        dependencies = {
            element.text
            for tag in ("build_depend", "build_export_depend", "exec_depend")
            for element in package_root.findall(tag)
        }
        self.assertIn("visualization_msgs", dependencies)

        cmake_text = (
            self.package_root / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("visualization_msgs", cmake_text)
        self.assertIn("scripts/lidar_attitude_debugger.py", cmake_text)
        self.assertIn("catkin_add_nosetests(test)", cmake_text)


if __name__ == "__main__":
    unittest.main()
