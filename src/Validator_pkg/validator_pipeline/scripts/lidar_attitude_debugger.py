#!/usr/bin/env python3
"""Visualize LiDAR z=0 motion and compare TF/LiDAR roll and pitch.

sub:
    /lidar3D
    /tf

pub:
    /lidar/debug/ground_bev
    /lidar/debug/colored_points
    /lidar/debug/markers
    /lidar3D/leveled
"""

from collections import deque
from dataclasses import dataclass
import json
import math
import os
import threading

import cv2
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class BevGeometry:
    """Metric geometry for a forward-facing diagnostic image."""

    forward_range: float = 25.0
    lateral_range: float = 30.0
    resolution: float = 0.05

    @property
    def height(self):
        return int(round(self.forward_range / self.resolution))

    @property
    def width(self):
        return int(round(self.lateral_range / self.resolution))


@dataclass(frozen=True)
class PlaneFitConfig:
    """Confidence and iteration settings for one RANSAC plane fit."""

    iterations: int = 120
    distance_threshold: float = 0.15
    min_points: int = 100
    min_inlier_ratio: float = 0.20
    max_tilt_degrees: float = 15.0
    max_rms: float = 0.15


@dataclass(frozen=True)
class PlaneEstimate:
    """Result of a confidence-gated road-plane fit."""

    valid: bool
    reason: str
    normal: np.ndarray
    offset: float
    roll_degrees: float
    pitch_degrees: float
    inlier_count: int
    candidate_count: int
    inlier_ratio: float
    rms: float


@dataclass(frozen=True)
class GroundRoi:
    """Vehicle-frame region used to find local road returns."""

    x_min: float = 2.0
    x_max: float = 25.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = -0.75
    z_max: float = 0.75


@dataclass(frozen=True)
class TransformSample:
    """One timestamped map-to-base_link attitude sample."""

    stamp: float
    quaternion: object
    rpy_degrees: np.ndarray


class CausalTransformHistory:
    """Select only transforms at or before a LiDAR reference stamp."""

    def __init__(self, maxlen=200):
        self.samples = deque(maxlen=int(maxlen))
        self.last_stamp = None

    def __len__(self):
        return len(self.samples)

    def clear(self):
        self.samples.clear()
        self.last_stamp = None

    def add(self, stamp, quaternion):
        stamp = float(stamp)
        did_reset = self.last_stamp is not None and stamp < self.last_stamp
        if did_reset:
            self.clear()
        self.samples.append(
            TransformSample(
                stamp=stamp,
                quaternion=quaternion,
                rpy_degrees=quaternion_to_rpy_degrees(quaternion),
            )
        )
        self.last_stamp = stamp
        return did_reset

    def select(self, reference_stamp, max_age):
        reference_stamp = float(reference_stamp)
        max_age = float(max_age)
        for sample in reversed(self.samples):
            if sample.stamp > reference_stamp:
                continue
            if reference_stamp - sample.stamp <= max_age:
                return sample
            return None
        return None


class AttitudeMedianFilter:
    """Short median filter for valid LiDAR road-relative estimates."""

    def __init__(self, window_size=5):
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        self.values = deque(maxlen=int(window_size))

    def reset(self):
        self.values.clear()

    def update(self, roll_degrees, pitch_degrees):
        self.values.append((float(roll_degrees), float(pitch_degrees)))
        return np.median(np.asarray(self.values, dtype=np.float64), axis=0)


def quaternion_to_rpy_degrees(quaternion):
    """Convert a ROS-style quaternion object into roll, pitch, yaw degrees."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return np.degrees([roll, pitch, yaw])


def quaternion_rotation_matrix(quaternion):
    """Return the base-to-world rotation represented by a ROS quaternion."""
    values = np.asarray(
        [
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(values)
    if norm <= 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def world_up_in_base(quaternion):
    """Express the map-frame up vector in base_link coordinates."""
    rotation_world_from_base = quaternion_rotation_matrix(quaternion)
    return rotation_world_from_base.T @ np.asarray([0.0, 0.0, 1.0])


def format_status_lines(tf_sample, tf_age, plane, filtered_lidar_rpy):
    """Create unambiguous status text for both attitude references."""
    if tf_sample is None:
        tf_line = "TF world: INVALID (missing/stale)"
    else:
        tf_line = (
            "TF world: VALID  roll={:+.2f} deg pitch={:+.2f} deg "
            "age={:.3f}s"
        ).format(
            tf_sample.rpy_degrees[0],
            tf_sample.rpy_degrees[1],
            float(tf_age),
        )

    if plane.valid and filtered_lidar_rpy is not None:
        lidar_line = (
            "LiDAR road-relative: VALID  roll={:+.2f} deg "
            "pitch={:+.2f} deg"
        ).format(
            filtered_lidar_rpy[0],
            filtered_lidar_rpy[1],
        )
    else:
        lidar_line = "LiDAR road-relative: INVALID ({})".format(plane.reason)

    quality_line = (
        "plane inliers={}/{} ({:.1f}%) RMS={}m"
    ).format(
        plane.inlier_count,
        plane.candidate_count,
        100.0 * plane.inlier_ratio,
        "{:.3f}".format(plane.rms) if np.isfinite(plane.rms) else "n/a",
    )
    return [tf_line, lidar_line, quality_line]


def classify_ground_band(points, half_width):
    """Return the points that lie inside a base_link z=0 tolerance band."""
    points = np.asarray(points)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.abs(points[:, 2]) <= float(half_width)


def normal_to_road_rpy_degrees(normal):
    """Convert an upward road normal in base_link into relative roll/pitch."""
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm <= 1.0e-12:
        raise ValueError("ground normal must be non-zero")
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal

    roll = math.atan2(normal[1], normal[2])
    pitch = math.atan2(
        -normal[0],
        math.sqrt(normal[1] ** 2 + normal[2] ** 2),
    )
    return np.degrees([roll, pitch])


def _invalid_plane(reason, candidate_count):
    return PlaneEstimate(
        valid=False,
        reason=reason,
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        offset=0.0,
        roll_degrees=float("nan"),
        pitch_degrees=float("nan"),
        inlier_count=0,
        candidate_count=int(candidate_count),
        inlier_ratio=0.0,
        rms=float("nan"),
    )


def _refine_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def fit_ground_plane_ransac(points, config, rng=None):
    """Fit one upward plane and reject results that fail confidence gates."""
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    candidate_count = len(points)
    if candidate_count < config.min_points:
        return _invalid_plane("insufficient_points", candidate_count)

    if rng is None:
        rng = np.random.default_rng()

    sample_indices = np.asarray(
        [
            rng.choice(candidate_count, size=3, replace=False)
            for _ in range(config.iterations)
        ],
        dtype=np.int64,
    )
    first = points[sample_indices[:, 0]]
    second = points[sample_indices[:, 1]]
    third = points[sample_indices[:, 2]]
    normals = np.cross(second - first, third - first)
    normal_lengths = np.linalg.norm(normals, axis=1)
    usable = normal_lengths > 1.0e-9
    normals = normals[usable]
    first = first[usable]
    normal_lengths = normal_lengths[usable]
    if len(normals) == 0:
        return _invalid_plane("no_plane", candidate_count)

    normals /= normal_lengths[:, None]
    offsets = -np.einsum("ij,ij->i", normals, first)
    all_distances = np.abs(points @ normals.T + offsets[None, :])
    all_masks = all_distances <= config.distance_threshold
    counts = np.count_nonzero(all_masks, axis=0)
    distance_sums = np.sum(
        np.where(all_masks, all_distances, 0.0),
        axis=0,
    )
    mean_distances = np.divide(
        distance_sums,
        counts,
        out=np.full_like(distance_sums, float("inf")),
        where=counts > 0,
    )
    ordering = np.lexsort((mean_distances, -counts))
    best_index = int(ordering[0])
    best_mask = all_masks[:, best_index]
    best_count = int(counts[best_index])
    if best_count < 3:
        return _invalid_plane("no_plane", candidate_count)

    normal, offset = _refine_plane(points[best_mask])
    distances = np.abs(points @ normal + offset)
    inlier_mask = distances <= config.distance_threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / float(candidate_count)
    rms = float(np.sqrt(np.mean(distances[inlier_mask] ** 2)))
    tilt_degrees = math.degrees(
        math.acos(max(-1.0, min(1.0, float(normal[2]))))
    )
    roll_degrees, pitch_degrees = normal_to_road_rpy_degrees(normal)

    reason = "valid"
    valid = True
    if inlier_ratio < config.min_inlier_ratio:
        reason = "low_inlier_ratio"
        valid = False
    elif tilt_degrees > config.max_tilt_degrees:
        reason = "tilt_exceeded"
        valid = False
    elif rms > config.max_rms:
        reason = "high_rms"
        valid = False

    return PlaneEstimate(
        valid=valid,
        reason=reason,
        normal=np.asarray(normal, dtype=np.float64),
        offset=float(offset),
        roll_degrees=float(roll_degrees),
        pitch_degrees=float(pitch_degrees),
        inlier_count=inlier_count,
        candidate_count=candidate_count,
        inlier_ratio=float(inlier_ratio),
        rms=rms,
    )


def render_ground_bev(points, ground_mask, geometry):
    """Render ordinary returns gray and base_link z=0 returns red."""
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    ground_mask = np.asarray(ground_mask, dtype=bool).reshape((-1,))
    if len(points) != len(ground_mask):
        raise ValueError("points and ground_mask must have the same length")

    image = np.zeros((geometry.height, geometry.width, 3), dtype=np.uint8)
    if len(points) == 0:
        return image

    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < geometry.forward_range)
        & (points[:, 1] >= -geometry.lateral_range / 2.0)
        & (points[:, 1] < geometry.lateral_range / 2.0)
    )
    rows = (
        geometry.height
        - 1
        - np.floor(points[:, 0] / geometry.resolution).astype(np.int32)
    )
    columns = np.floor(
        (geometry.lateral_range / 2.0 - points[:, 1])
        / geometry.resolution
    ).astype(np.int32)
    valid &= (
        (rows >= 0)
        & (rows < geometry.height)
        & (columns >= 0)
        & (columns < geometry.width)
    )

    image[rows[valid], columns[valid]] = (150, 150, 150)
    red = valid & ground_mask
    image[rows[red], columns[red]] = (0, 0, 255)
    return image


def select_ground_candidates(points, roi):
    """Select the configurable base_link ROI used for plane fitting."""
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return points
    mask = (
        (points[:, 0] >= roi.x_min)
        & (points[:, 0] <= roi.x_max)
        & (points[:, 1] >= roi.y_min)
        & (points[:, 1] <= roi.y_max)
        & (points[:, 2] >= roi.z_min)
        & (points[:, 2] <= roi.z_max)
    )
    return np.ascontiguousarray(points[mask])


def plane_axis_directions(normal):
    """Return normalized X/Y slice directions parallel to a fitted plane."""
    normal = np.asarray(normal, dtype=np.float64).reshape((3,))
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1.0e-12:
        raise ValueError("plane normal must be non-zero")
    normal = normal / normal_length
    if abs(normal[2]) <= 1.0e-9:
        raise ValueError("plane normal is too close to vertical")

    x_direction = np.asarray(
        [normal[2], 0.0, -normal[0]],
        dtype=np.float64,
    )
    y_direction = np.asarray(
        [0.0, normal[2], -normal[1]],
        dtype=np.float64,
    )
    x_direction /= np.linalg.norm(x_direction)
    y_direction /= np.linalg.norm(y_direction)
    return x_direction, y_direction


def level_points_to_plane(points, normal, offset):
    """Express base-frame points in a road frame whose fitted plane is z=0."""
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    normal = np.asarray(normal, dtype=np.float64).reshape((3,))
    normal_length = np.linalg.norm(normal)
    if not np.isfinite(normal).all() or normal_length <= 1.0e-12:
        raise ValueError("plane normal must be finite and non-zero")
    if not np.isfinite(offset):
        raise ValueError("plane offset must be finite")

    normal = normal / normal_length
    normalized_offset = float(offset) / normal_length
    if normal[2] < 0.0:
        normal = -normal
        normalized_offset = -normalized_offset

    base_forward = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = base_forward - np.dot(base_forward, normal) * normal
    x_length = np.linalg.norm(x_axis)
    if x_length <= 1.0e-12:
        raise ValueError("plane normal cannot be parallel to base forward")
    x_axis /= x_length
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    basis = np.column_stack([x_axis, y_axis, normal])
    leveled = points @ basis
    leveled[:, 2] += normalized_offset
    return np.ascontiguousarray(leveled, dtype=np.float32)


def rotation_matrix_from_rpy(roll_degrees, pitch_degrees, yaw_degrees):
    """Build a vehicle-frame XYZ rotation from MORAI degree values."""
    roll = math.radians(float(roll_degrees))
    pitch = math.radians(float(pitch_degrees))
    yaw = math.radians(float(yaw_degrees))
    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float32,
    )
    rotation_y = np.asarray(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float32,
    )
    rotation_z = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rotation_z @ rotation_y @ rotation_x


def load_lidar_calibration(config_path, input_topic):
    """Load the static LiDAR mount matching a MORAI ROS topic."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        sensor_config = json.load(config_file)

    for lidar in sensor_config.get("Lidar3DList", []):
        lidar_config = lidar.get("lc", {})
        ros_topic = lidar_config.get("rosConfig", {}).get("Topic", "")
        if ros_topic != input_topic:
            continue
        position = lidar["pos"]
        rotation = lidar["rot"]
        return {
            "translation": np.asarray(
                [
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                ],
                dtype=np.float32,
            ),
            "rotation": rotation_matrix_from_rpy(
                rotation["roll"],
                rotation["pitch"],
                rotation["yaw"],
            ),
        }
    raise ValueError(
        "No Lidar3DList calibration for topic '{}' in {}".format(
            input_topic,
            config_path,
        )
    )


def decode_point_cloud(message):
    """Decode finite, non-zero XYZ returns from PointCloud2."""
    fields_by_name = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"} - set(fields_by_name)
    if missing:
        raise ValueError(
            "PointCloud2 is missing field(s): {}".format(
                ", ".join(sorted(missing))
            )
        )

    xyz_fields = [fields_by_name[name] for name in ("x", "y", "z")]
    has_float32_xyz = all(
        field.datatype == PointField.FLOAT32 and field.count == 1
        for field in xyz_fields
    )
    if has_float32_xyz:
        endian = ">" if message.is_bigendian else "<"
        cloud_dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [endian + "f4"] * 3,
                "offsets": [field.offset for field in xyz_fields],
                "itemsize": message.point_step,
            }
        )
        if message.row_step == message.width * message.point_step:
            structured = np.frombuffer(
                message.data,
                dtype=cloud_dtype,
                count=message.width * message.height,
            )
        else:
            structured = np.ndarray(
                shape=(message.height, message.width),
                dtype=cloud_dtype,
                buffer=message.data,
                strides=(message.row_step, message.point_step),
            ).reshape((-1,))
        decoded = np.column_stack(
            [structured[name] for name in ("x", "y", "z")]
        ).astype(np.float32, copy=False)
    else:
        decoded = np.asarray(
            list(
                point_cloud2.read_points(
                    message,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ),
            dtype=np.float32,
        )
    if decoded.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    decoded = decoded.reshape((-1, 3))
    valid = np.isfinite(decoded).all(axis=1)
    valid &= np.any(np.abs(decoded) > 1.0e-6, axis=1)
    return np.ascontiguousarray(decoded[valid])


def sensor_to_base(points, calibration):
    """Apply the configured rigid LiDAR mount to base_link."""
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return points
    transformed = points @ calibration["rotation"].T
    transformed += calibration["translation"]
    return np.ascontiguousarray(transformed, dtype=np.float32)


class LidarAttitudeDebugger:
    """Publish isolated LiDAR attitude diagnostics without changing model I/O."""

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.input_topic = rospy.get_param("~input_topic", "/lidar3D")
        self.tf_topic = rospy.get_param("~tf_topic", "/tf")
        self.world_frame = rospy.get_param("~world_frame", "map").lstrip("/")
        self.base_frame = rospy.get_param(
            "~base_frame",
            "base_link",
        ).lstrip("/")
        self.ground_band_half_width = float(
            rospy.get_param("~ground_band_half_width", 0.10)
        )
        self.max_tf_age = float(rospy.get_param("~max_tf_age", 0.10))
        self.ransac_seed = int(rospy.get_param("~ransac_seed", 23))
        self.rng = np.random.default_rng(self.ransac_seed)

        self.geometry = BevGeometry(
            forward_range=float(rospy.get_param("~forward_range", 25.0)),
            lateral_range=float(rospy.get_param("~lateral_range", 30.0)),
            resolution=float(rospy.get_param("~resolution", 0.05)),
        )
        self.roi = GroundRoi(
            x_min=float(rospy.get_param("~roi/x_min", 2.0)),
            x_max=float(rospy.get_param("~roi/x_max", 25.0)),
            y_min=float(rospy.get_param("~roi/y_min", -10.0)),
            y_max=float(rospy.get_param("~roi/y_max", 10.0)),
            z_min=float(rospy.get_param("~roi/z_min", -0.75)),
            z_max=float(rospy.get_param("~roi/z_max", 0.75)),
        )
        self.fit_config = PlaneFitConfig(
            iterations=int(rospy.get_param("~ransac/iterations", 120)),
            distance_threshold=float(
                rospy.get_param("~ransac/distance_threshold", 0.15)
            ),
            min_points=int(rospy.get_param("~ransac/min_points", 100)),
            min_inlier_ratio=float(
                rospy.get_param("~ransac/min_inlier_ratio", 0.20)
            ),
            max_tilt_degrees=float(
                rospy.get_param("~ransac/max_tilt_degrees", 15.0)
            ),
            max_rms=float(rospy.get_param("~ransac/max_rms", 0.15)),
        )
        self._validate_parameters()

        package_path = rospkg.RosPack().get_path("Validator_pkg")
        default_config_path = os.path.join(
            package_path,
            "config",
            "0716demo.json",
        )
        self.config_path = rospy.get_param(
            "~config_path",
            default_config_path,
        )
        self.calibration = load_lidar_calibration(
            self.config_path,
            self.input_topic,
        )

        self.tf_history = CausalTransformHistory(
            maxlen=int(rospy.get_param("~tf_history_size", 200))
        )
        self.attitude_filter = AttitudeMedianFilter(
            window_size=int(rospy.get_param("~median_window", 5))
        )
        self.last_lidar_stamp = None

        self.image_publisher = rospy.Publisher(
            rospy.get_param(
                "~ground_bev_topic",
                "/lidar/debug/ground_bev",
            ),
            Image,
            queue_size=1,
        )
        self.cloud_publisher = rospy.Publisher(
            rospy.get_param(
                "~colored_points_topic",
                "/lidar/debug/colored_points",
            ),
            PointCloud2,
            queue_size=1,
        )
        self.leveled_cloud_publisher = rospy.Publisher(
            rospy.get_param(
                "~leveled_points_topic",
                "/lidar3D/leveled",
            ),
            PointCloud2,
            queue_size=1,
        )
        self.marker_publisher = rospy.Publisher(
            rospy.get_param(
                "~markers_topic",
                "/lidar/debug/markers",
            ),
            MarkerArray,
            queue_size=1,
        )
        self.tf_subscriber = rospy.Subscriber(
            self.tf_topic,
            TFMessage,
            self.tf_callback,
            queue_size=100,
        )
        self.lidar_subscriber = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.lidar_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.loginfo(
            "LiDAR attitude debugger: %s + %s -> z=0 BEV/cloud/markers + "
            "/lidar3D/leveled",
            self.input_topic,
            self.tf_topic,
        )

    def _validate_parameters(self):
        if self.ground_band_half_width <= 0.0:
            raise ValueError("~ground_band_half_width must be positive")
        if self.max_tf_age < 0.0:
            raise ValueError("~max_tf_age must be non-negative")
        if (
            self.geometry.forward_range <= 0.0
            or self.geometry.lateral_range <= 0.0
            or self.geometry.resolution <= 0.0
        ):
            raise ValueError("BEV geometry values must be positive")
        if not (
            self.roi.x_min < self.roi.x_max
            and self.roi.y_min < self.roi.y_max
            and self.roi.z_min < self.roi.z_max
        ):
            raise ValueError("ground ROI minimums must be below maximums")

    def tf_callback(self, message):
        with self.lock:
            for transform in message.transforms:
                parent = transform.header.frame_id.lstrip("/")
                child = transform.child_frame_id.lstrip("/")
                if parent != self.world_frame or child != self.base_frame:
                    continue
                did_reset = self.tf_history.add(
                    transform.header.stamp.to_sec(),
                    transform.transform.rotation,
                )
                if did_reset:
                    self.attitude_filter.reset()
                    self.rng = np.random.default_rng(self.ransac_seed)

    def lidar_callback(self, message):
        stamp = message.header.stamp.to_sec()
        if self.last_lidar_stamp is not None and stamp < self.last_lidar_stamp:
            with self.lock:
                self.attitude_filter.reset()
                self.rng = np.random.default_rng(self.ransac_seed)
        self.last_lidar_stamp = stamp

        try:
            points = sensor_to_base(
                decode_point_cloud(message),
                self.calibration,
            )
        except (TypeError, ValueError) as error:
            rospy.logerr_throttle(2.0, "LiDAR attitude decode failed: %s", error)
            return

        if len(points) == 0:
            plane = _invalid_plane("empty_cloud", 0)
            self.publish_markers(message.header, None, None, plane, None)
            return

        ground_mask = classify_ground_band(
            points,
            self.ground_band_half_width,
        )
        candidates = select_ground_candidates(points, self.roi)
        plane = fit_ground_plane_ransac(
            candidates,
            self.fit_config,
            self.rng,
        )
        if plane.valid:
            filtered_lidar_rpy = self.attitude_filter.update(
                plane.roll_degrees,
                plane.pitch_degrees,
            )
            leveled_points = level_points_to_plane(
                points,
                plane.normal,
                plane.offset,
            )
            self.publish_leveled_cloud(leveled_points, message.header)
        else:
            self.attitude_filter.reset()
            filtered_lidar_rpy = None

        with self.lock:
            tf_sample = self.tf_history.select(stamp, self.max_tf_age)
        tf_age = stamp - tf_sample.stamp if tf_sample is not None else None
        status_lines = format_status_lines(
            tf_sample,
            tf_age,
            plane,
            filtered_lidar_rpy,
        )

        image = render_ground_bev(points, ground_mask, self.geometry)
        self._draw_status(image, status_lines)
        self.publish_image(image, message.header)
        self.publish_colored_cloud(points, ground_mask, message.header)
        self.publish_markers(
            message.header,
            tf_sample,
            tf_age,
            plane,
            filtered_lidar_rpy,
        )
        rospy.loginfo_throttle(2.0, " | ".join(status_lines))

    @staticmethod
    def _draw_status(image, status_lines):
        for index, line in enumerate(status_lines):
            cv2.putText(
                image,
                line,
                (12, 25 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def publish_image(self, image, source_header):
        try:
            message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            message.header.seq = source_header.seq
            message.header.stamp = source_header.stamp
            message.header.frame_id = self.base_frame
            self.image_publisher.publish(message)
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "Ground BEV conversion failed: %s", error)

    def publish_colored_cloud(self, points, ground_mask, source_header):
        colors = np.full(len(points), 0x969696, dtype=np.uint32)
        colors[ground_mask] = 0xFF0000
        rgb_as_float = colors.view(np.float32)
        values = np.column_stack([points, rgb_as_float]).tolist()
        header = Header(
            seq=source_header.seq,
            stamp=source_header.stamp,
            frame_id=self.base_frame,
        )
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        self.cloud_publisher.publish(
            point_cloud2.create_cloud(header, fields, values)
        )

    def publish_leveled_cloud(self, points, source_header):
        header = Header(
            seq=source_header.seq,
            stamp=source_header.stamp,
            frame_id=self.base_frame,
        )
        self.leveled_cloud_publisher.publish(
            point_cloud2.create_cloud_xyz32(header, points)
        )

    @staticmethod
    def _marker(header, marker_id, marker_type):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "lidar_attitude_debug"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    def _arrow_marker(self, header, marker_id, start, direction, color):
        marker = self._marker(header, marker_id, Marker.ARROW)
        marker.points = [
            Point(*[float(value) for value in start]),
            Point(
                *[
                    float(value)
                    for value in np.asarray(start) + 2.0 * np.asarray(direction)
                ]
            ),
        ]
        marker.scale.x = 0.08
        marker.scale.y = 0.16
        marker.scale.z = 0.20
        self._set_color(marker, *color, 1.0)
        return marker

    def publish_markers(
        self,
        source_header,
        tf_sample,
        tf_age,
        plane,
        filtered_lidar_rpy,
    ):
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers = [delete_all]

        status_lines = format_status_lines(
            tf_sample,
            tf_age,
            plane,
            filtered_lidar_rpy,
        )
        text = self._marker(source_header, 2, Marker.TEXT_VIEW_FACING)
        text.pose.position.x = 4.0
        text.pose.position.z = 2.5
        text.scale.z = 0.38
        text.text = "\n".join(status_lines)
        self._set_color(text, 1.0, 1.0, 1.0, 1.0)
        markers.append(text)

        arrow_start = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
        if plane.valid:
            markers.append(
                self._arrow_marker(
                    source_header,
                    3,
                    arrow_start,
                    plane.normal,
                    (1.0, 0.85, 0.0),
                )
            )
        if tf_sample is not None:
            markers.append(
                self._arrow_marker(
                    source_header,
                    4,
                    arrow_start,
                    world_up_in_base(tf_sample.quaternion),
                    (0.0, 1.0, 1.0),
                )
            )

        lidar_origin = np.asarray(
            self.calibration["translation"],
            dtype=np.float64,
        )
        half_lateral_range = self.geometry.lateral_range / 2.0
        reference_axes = self._marker(source_header, 5, Marker.LINE_LIST)
        reference_axis_points = [
            lidar_origin,
            lidar_origin
            + np.asarray(
                [self.geometry.forward_range, 0.0, 0.0],
                dtype=np.float64,
            ),
            lidar_origin
            + np.asarray([0.0, -half_lateral_range, 0.0]),
            lidar_origin
            + np.asarray([0.0, half_lateral_range, 0.0]),
        ]
        reference_axes.points = [
            Point(*[float(value) for value in point])
            for point in reference_axis_points
        ]
        reference_axes.scale.x = 0.15
        self._set_color(reference_axes, 1.0, 1.0, 0.0, 1.0)
        markers.append(reference_axes)

        if plane.valid:
            x_direction, y_direction = plane_axis_directions(plane.normal)
            current_axes = self._marker(
                source_header,
                6,
                Marker.LINE_LIST,
            )
            current_axis_points = [
                lidar_origin,
                lidar_origin
                + self.geometry.forward_range * x_direction,
                lidar_origin - half_lateral_range * y_direction,
                lidar_origin + half_lateral_range * y_direction,
            ]
            current_axes.points = [
                Point(*[float(value) for value in point])
                for point in current_axis_points
            ]
            current_axes.scale.x = 0.12
            self._set_color(current_axes, 1.0, 0.0, 0.0, 1.0)
            markers.append(current_axes)

        self.marker_publisher.publish(MarkerArray(markers=markers))


def main():
    rospy.init_node("lidar_attitude_debugger")
    try:
        LidarAttitudeDebugger()
        rospy.spin()
    except (OSError, KeyError, TypeError, ValueError) as error:
        rospy.logfatal("LiDAR attitude debugger initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
