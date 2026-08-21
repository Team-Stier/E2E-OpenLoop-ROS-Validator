#!/usr/bin/env python3
"""Convert MORAI PointCloud2 data into visual and model LiDAR BEVs.

sub:
    /lidar3D                         sensor_msgs/PointCloud2
    /tf                              tf2_msgs/TFMessage (TFv6 only)

pub:
    /lidar3D/BEV                    mono8, 600x500 visualization BEV
    /lidar/tfv6/processed           32FC1, 384x320 density histogram
    /lidar/interfuser/processed     32FC3, 224x224 below/above/total
    /lidar/diffusiondrive_v2/processed
                                     32FC1, 256x256 above-ground histogram

Set ``~model_profiles`` to one profile, a comma-separated string, a list, or
``all``. The default ``raw`` publishes only the visualization BEV. SparseDrive
v2 is accepted as a profile but has no LiDAR output in its official sensor
configuration.

Float ROS images are stored as HWC. Inference code must convert them to CHW
and add a batch dimension before passing them to PyTorch models.
"""

import json
import math
import os
import threading
from collections import deque, namedtuple
from copy import copy

import cv2
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from tf2_msgs.msg import TFMessage


SUPPORTED_PROFILES = {
    "tfv6",
    "interfuser",
    "diffusiondrive_v2",
    "sparsedrive_v2",
}

PROFILE_ALIASES = {
    "transfuser_v6": "tfv6",
    "diffusiondrive": "diffusiondrive_v2",
    "diffusiondrivev2": "diffusiondrive_v2",
    "diffusion_drive_v2": "diffusiondrive_v2",
    "sparsedrive": "sparsedrive_v2",
    "sparsedrivev2": "sparsedrive_v2",
    "sparse_drive_v2": "sparsedrive_v2",
}

MODEL_OUTPUT_TOPICS = {
    "tfv6": "/lidar/tfv6/processed",
    "interfuser": "/lidar/interfuser/processed",
    "diffusiondrive_v2": "/lidar/diffusiondrive_v2/processed",
}

EgoPose = namedtuple("EgoPose", "stamp x y yaw")
LidarSweep = namedtuple("LidarSweep", "stamp world_xy z")


def parse_model_profiles(value):
    """Normalize a ROS parameter into a validated set of model profiles."""
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        requested = [str(item).strip() for item in value]
    else:
        raise ValueError(
            "~model_profiles must be a string or list, got {}".format(
                type(value).__name__,
            )
        )

    normalized = set()
    for profile in requested:
        profile = profile.lower()
        if not profile or profile in ("raw", "none", "legacy"):
            continue
        if profile == "all":
            normalized.update(SUPPORTED_PROFILES)
            continue
        profile = PROFILE_ALIASES.get(profile, profile)
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                "Unsupported model profile '{}'. Choose from: raw, {}".format(
                    profile,
                    ", ".join(sorted(SUPPORTED_PROFILES)),
                )
            )
        normalized.add(profile)

    return normalized


def rotation_matrix_from_rpy(roll_degrees, pitch_degrees, yaw_degrees):
    """Return a sensor-to-vehicle rotation matrix for MORAI RPY degrees."""
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)

    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float32,
    )
    rotation_y = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float32,
    )
    rotation_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rotation_z @ rotation_y @ rotation_x


def load_lidar_calibration(config_path, input_topic):
    """Load the LiDAR pose matching ``input_topic`` from a MORAI JSON file."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        sensor_config = json.load(config_file)

    for lidar in sensor_config.get("Lidar3DList", []):
        lidar_config = lidar.get("lc", {})
        ros_topic = lidar_config.get("rosConfig", {}).get("Topic", "")
        if ros_topic != input_topic:
            continue

        position = lidar["pos"]
        rotation = lidar["rot"]
        translation = np.array(
            [
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            ],
            dtype=np.float32,
        )
        rotation_matrix = rotation_matrix_from_rpy(
            float(rotation["roll"]),
            float(rotation["pitch"]),
            float(rotation["yaw"]),
        )
        return {
            "translation": translation,
            "rotation": rotation_matrix,
            "roll": float(rotation["roll"]),
            "pitch": float(rotation["pitch"]),
            "yaw": float(rotation["yaw"]),
        }

    raise ValueError(
        "No Lidar3DList calibration for topic '{}' in {}".format(
            input_topic,
            config_path,
        )
    )


def decode_point_cloud(message):
    """Decode finite, non-zero XYZ returns from a PointCloud2 message."""
    field_names = {field.name for field in message.fields}
    missing_fields = {"x", "y", "z"} - field_names
    if missing_fields:
        raise ValueError(
            "PointCloud2 is missing field(s): {}".format(
                ", ".join(sorted(missing_fields)),
            )
        )

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
    finite = np.isfinite(decoded).all(axis=1)
    non_zero = np.any(np.abs(decoded) > 1e-6, axis=1)
    return np.ascontiguousarray(decoded[finite & non_zero])


def sensor_to_vehicle(points, calibration):
    """Transform MORAI LiDAR sensor-frame XYZ into the vehicle frame."""
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    transformed = points @ calibration["rotation"].T
    transformed += calibration["translation"]
    return np.ascontiguousarray(transformed, dtype=np.float32)


def capped_histogram(points_xy, x_edges, y_edges, maximum_count=5.0):
    """Create a clipped point-count histogram normalized to [0, 1]."""
    if points_xy.size == 0:
        return np.zeros(
            (len(x_edges) - 1, len(y_edges) - 1),
            dtype=np.float32,
        )

    histogram = np.histogram2d(
        points_xy[:, 0],
        points_xy[:, 1],
        bins=(x_edges, y_edges),
    )[0]
    np.minimum(histogram, maximum_count, out=histogram)
    histogram /= maximum_count
    return np.ascontiguousarray(histogram, dtype=np.float32)


def prepare_visual_bev(
    points_vehicle,
    forward_range=25.0,
    lateral_range=30.0,
    resolution=0.05,
):
    """Build a camera-aligned forward-up, vehicle-left-left visualization."""
    output_height = int(round(forward_range / resolution))
    output_width = int(round(lateral_range / resolution))
    x_edges = np.linspace(
        0.0,
        forward_range,
        output_height + 1,
        dtype=np.float32,
    )
    y_edges = np.linspace(
        -lateral_range / 2.0,
        lateral_range / 2.0,
        output_width + 1,
        dtype=np.float32,
    )
    histogram = capped_histogram(
        points_vehicle[:, :2],
        x_edges,
        y_edges,
    )

    # Histogram axes increase from near-to-far and right-to-left. Reverse both
    # so the image displays far/forward at the top and vehicle-left on the left.
    visual = np.rint(histogram[::-1, ::-1] * 255.0).astype(np.uint8)

    # Grid lines are visualization-only and never enter a model tensor.
    grid_step_pixels = max(1, int(round(5.0 / resolution)))
    for pixel_y in range(0, visual.shape[0], grid_step_pixels):
        cv2.line(visual, (0, pixel_y), (visual.shape[1] - 1, pixel_y), 60, 1)
    for pixel_x in range(0, visual.shape[1], grid_step_pixels):
        cv2.line(visual, (pixel_x, 0), (pixel_x, visual.shape[0] - 1), 60, 1)

    return np.ascontiguousarray(visual)


def prepare_tfv6(points_vehicle):
    """Build TFv6's one-channel 384x320 LiDAR density pseudo-image."""
    if points_vehicle.size == 0:
        filtered = points_vehicle
    else:
        height_mask = (
            (points_vehicle[:, 2] >= -4.0)
            & (points_vehicle[:, 2] <= 10.0)
        )
        filtered = points_vehicle[height_mask].copy()

    # LEAD/TFv6 uses CARLA ego axes (x forward, y right). MORAI vehicle y is
    # left-positive, so flip it before applying the official raster layout.
    if filtered.size:
        filtered[:, 1] *= -1.0

    x_edges = np.linspace(-32.0, 64.0, 385, dtype=np.float32)
    y_edges = np.linspace(-40.0, 40.0, 321, dtype=np.float32)
    histogram = capped_histogram(filtered[:, :2], x_edges, y_edges)
    return np.ascontiguousarray(histogram.T, dtype=np.float32)


def remove_ground_plane_ransac(
    points_vehicle,
    distance_threshold=0.2,
    iterations=48,
):
    """Remove the dominant near-horizontal ground plane for TFv6.

    LEAD applies segmented RANSAC before accumulating LiDAR sweeps. MORAI's
    validation bag is locally flat, so a deterministic dominant-plane fit is
    used here without introducing a new runtime dependency.
    """
    if points_vehicle.shape[0] < 3:
        return points_vehicle

    candidate_mask = (
        (points_vehicle[:, 2] >= -2.0)
        & (points_vehicle[:, 2] <= 1.0)
        & (np.linalg.norm(points_vehicle[:, :2], axis=1) <= 50.0)
    )
    candidates = points_vehicle[candidate_mask]
    if candidates.shape[0] < 3:
        return points_vehicle

    random_generator = np.random.default_rng(0)
    best_normal = None
    best_offset = None
    best_inlier_count = 0

    for _ in range(iterations):
        sample_indices = random_generator.choice(
            candidates.shape[0],
            size=3,
            replace=False,
        )
        first, second, third = candidates[sample_indices]
        normal = np.cross(second - first, third - first)
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-6:
            continue
        normal /= normal_length
        if normal[2] < 0.0:
            normal *= -1.0
        if normal[2] < 0.85:
            continue

        offset = -float(np.dot(normal, first))
        distances = np.abs(candidates @ normal + offset)
        inlier_count = int(np.count_nonzero(distances < distance_threshold))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal.copy()
            best_offset = offset

    if best_normal is None or best_inlier_count < 3:
        return points_vehicle

    candidate_distances = np.abs(candidates @ best_normal + best_offset)
    plane_points = candidates[candidate_distances < distance_threshold]
    if plane_points.shape[0] >= 3:
        design = np.column_stack(
            [plane_points[:, 0], plane_points[:, 1], np.ones(len(plane_points))]
        )
        coefficients = np.linalg.lstsq(
            design,
            plane_points[:, 2],
            rcond=None,
        )[0]
        plane_height = (
            coefficients[0] * points_vehicle[:, 0]
            + coefficients[1] * points_vehicle[:, 1]
            + coefficients[2]
        )
        ground_mask = (
            points_vehicle[:, 2] - plane_height
        ) < distance_threshold
    else:
        ground_mask = np.abs(
            points_vehicle @ best_normal + best_offset
        ) < distance_threshold

    return np.ascontiguousarray(points_vehicle[~ground_mask])


def vehicle_to_interfuser_sensor(points_vehicle):
    """Map vehicle XYZ into InterFuser's yaw=-90, z=2.5 m LiDAR frame."""
    if points_vehicle.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    converted = np.empty_like(points_vehicle, dtype=np.float32)
    # Official CARLA sensor pose: x=1.3, y=0.0, z=2.5, yaw=-90 degrees.
    # After the agent's explicit y *= -1 operation, coordinates become:
    #   first axis: vehicle-left, second axis: negative vehicle-forward.
    converted[:, 0] = points_vehicle[:, 1]
    converted[:, 1] = 1.3 - points_vehicle[:, 0]
    converted[:, 2] = points_vehicle[:, 2] - 2.5
    return converted


def prepare_interfuser(points_vehicle):
    """Build InterFuser's normalized below/above/total 224x224 input."""
    points = vehicle_to_interfuser_sensor(points_vehicle)
    x_edges = np.linspace(-28.0, 29.0, 225, dtype=np.float32)
    y_edges = np.linspace(-28.0, 0.0, 225, dtype=np.float32)

    below = points[points[:, 2] <= -2.0]
    above = points[points[:, 2] > -2.0]
    below_histogram = capped_histogram(below[:, :2], x_edges, y_edges)
    above_histogram = capped_histogram(above[:, :2], x_edges, y_edges)
    total_histogram = below_histogram + above_histogram

    return np.ascontiguousarray(
        np.stack(
            [below_histogram, above_histogram, total_histogram],
            axis=-1,
        ),
        dtype=np.float32,
    )


def prepare_diffusiondrive_v2(points_vehicle, morai_domain=False):
    """Build the public default DiffusionDriveV2 1x256x256 LiDAR input."""
    if points_vehicle.size == 0:
        above = points_vehicle
    else:
        below_max_height = points_vehicle[:, 2] < 100.0
        above_split = points_vehicle[:, 2] > 0.2
        keep = below_max_height & above_split
        if morai_domain:
            inside_vehicle = (
                (0.0 <= points_vehicle[:, 0])
                & (points_vehicle[:, 0] <= 2.0)
                & (-1.0 <= points_vehicle[:, 1])
                & (points_vehicle[:, 1] <= 1.0)
                & (0.8 <= points_vehicle[:, 2])
                & (points_vehicle[:, 2] <= 1.6)
            )
            keep &= ~inside_vehicle
        above = points_vehicle[keep]

    x_edges = np.linspace(-32.0, 32.0, 257, dtype=np.float32)
    y_edges = np.linspace(-32.0, 32.0, 257, dtype=np.float32)
    return capped_histogram(above[:, :2], x_edges, y_edges)


def select_output_stamp(source_stamp, use_ros_time_stamp):
    """명시적으로 요청한 경우에만 bag replay 시각을 선택한다."""
    return rospy.Time.now() if use_ros_time_stamp else source_stamp


def vehicle_points_to_world(points_vehicle, pose):
    """Transform vehicle-frame XY into world XY using a MORAI ego pose."""
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float32,
    )
    return points_vehicle[:, :2] @ rotation.T + np.array(
        [pose.x, pose.y],
        dtype=np.float32,
    )


def world_points_to_vehicle(world_xy, pose):
    """Transform world XY into the current MORAI vehicle frame."""
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float32,
    )
    translated = world_xy - np.array([pose.x, pose.y], dtype=np.float32)
    return translated @ rotation


def quaternion_to_yaw(quaternion):
    """Return planar yaw in radians from a geometry_msgs Quaternion."""
    sine = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cosine = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(sine, cosine)


class LidarBEVProcessor:
    """Publish visual and model-specific LiDAR BEV representations."""

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.input_topic = rospy.get_param("~input_topic", "/lidar3D")
        self.visual_topic = rospy.get_param(
            "~visual_topic",
            "/lidar3D/BEV",
        )
        self.tf_topic = rospy.get_param("~tf_topic", "/tf")
        self.world_frame = rospy.get_param("~world_frame", "map").lstrip("/")
        self.base_frame = rospy.get_param(
            "~base_frame",
            "base_link",
        ).lstrip("/")
        self.output_frame_id = rospy.get_param("~frame_id", "base_link")
        self.show_gui = bool(rospy.get_param("~show_gui", False))
        self.use_ros_time_stamp = bool(
            rospy.get_param("~use_ros_time_stamp", False)
        )
        self.morai_ddv2_domain = bool(
            rospy.get_param("~morai_ddv2_domain", False)
        )
        self.forward_range = float(rospy.get_param("~forward_range", 25.0))
        self.lateral_range = float(rospy.get_param("~lateral_range", 30.0))
        self.resolution = float(rospy.get_param("~resolution", 0.05))
        self.ego_sync_slop = float(rospy.get_param("~ego_sync_slop", 0.1))
        self.tfv6_num_sweeps = int(rospy.get_param("~tfv6_num_sweeps", 5))
        self.tfv6_history_seconds = float(
            rospy.get_param("~tfv6_history_seconds", 0.5)
        )
        self.tfv6_remove_ground = bool(
            rospy.get_param("~tfv6_remove_ground", True)
        )
        self.tfv6_ground_distance = float(
            rospy.get_param("~tfv6_ground_distance", 0.2)
        )

        if self.forward_range <= 0.0:
            raise ValueError("~forward_range must be positive")
        if self.lateral_range <= 0.0:
            raise ValueError("~lateral_range must be positive")
        if self.resolution <= 0.0:
            raise ValueError("~resolution must be positive")
        if self.ego_sync_slop < 0.0:
            raise ValueError("~ego_sync_slop must be non-negative")
        if self.tfv6_num_sweeps <= 0:
            raise ValueError("~tfv6_num_sweeps must be positive")
        if self.tfv6_history_seconds < 0.0:
            raise ValueError("~tfv6_history_seconds must be non-negative")
        if self.tfv6_ground_distance <= 0.0:
            raise ValueError("~tfv6_ground_distance must be positive")

        profile_parameter = rospy.get_param(
            "~model_profiles",
            rospy.get_param("~model_profile", "raw"),
        )
        self.model_profiles = parse_model_profiles(profile_parameter)

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

        self.ego_poses = deque(maxlen=200)
        self.tfv6_sweeps = deque(maxlen=self.tfv6_num_sweeps)
        self.last_lidar_stamp = None

        self.visual_publisher = rospy.Publisher(
            self.visual_topic,
            Image,
            queue_size=1,
        )
        self.model_publishers = {
            profile: rospy.Publisher(topic, Image, queue_size=1)
            for profile, topic in MODEL_OUTPUT_TOPICS.items()
            if profile in self.model_profiles
        }

        self.tf_subscriber = None
        if "tfv6" in self.model_profiles:
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
        rospy.on_shutdown(self.on_shutdown)

        translation = self.calibration["translation"]
        rospy.loginfo("LiDAR BEV processor initialized")
        rospy.loginfo("Input: %s", self.input_topic)
        rospy.loginfo("Visualization output: %s", self.visual_topic)
        rospy.loginfo(
            "Visualization area: %.1f m forward x %.1f m lateral at "
            "%.2f m/pixel (%dx%d)",
            self.forward_range,
            self.lateral_range,
            self.resolution,
            int(round(self.lateral_range / self.resolution)),
            int(round(self.forward_range / self.resolution)),
        )
        rospy.loginfo(
            "Calibration %s: xyz=(%.2f, %.2f, %.2f), rpy=(%.1f, %.1f, %.1f)",
            self.config_path,
            translation[0],
            translation[1],
            translation[2],
            self.calibration["roll"],
            self.calibration["pitch"],
            self.calibration["yaw"],
        )
        if self.model_profiles:
            rospy.loginfo(
                "LiDAR model profiles enabled: %s",
                ", ".join(sorted(self.model_profiles)),
            )
        else:
            rospy.loginfo(
                "LiDAR model profiles disabled (set ~model_profiles to enable)",
            )
        if "sparsedrive_v2" in self.model_profiles:
            rospy.loginfo(
                "SparseDrive v2 requests no LiDAR input; no SparseDrive LiDAR topic will be published",
            )

    def tf_callback(self, message):
        """Cache map-to-base_link poses for TFv6 motion compensation."""
        for stamped_transform in message.transforms:
            parent = stamped_transform.header.frame_id.lstrip("/")
            child = stamped_transform.child_frame_id.lstrip("/")
            if parent != self.world_frame or child != self.base_frame:
                continue

            translation = stamped_transform.transform.translation
            pose = EgoPose(
                stamp=stamped_transform.header.stamp.to_sec(),
                x=float(translation.x),
                y=float(translation.y),
                yaw=quaternion_to_yaw(
                    stamped_transform.transform.rotation,
                ),
            )
            with self.lock:
                self.ego_poses.append(pose)
            break

    def closest_ego_pose(self, stamp):
        """Return the cached ego pose closest to ``stamp`` within the slop."""
        with self.lock:
            if not self.ego_poses:
                return None
            pose = min(
                self.ego_poses,
                key=lambda candidate: abs(candidate.stamp - stamp),
            )
        if abs(pose.stamp - stamp) > self.ego_sync_slop:
            return None
        return pose

    def reset_history_if_time_reversed(self, stamp):
        """Reset sweep accumulation when a looping rosbag jumps backward."""
        with self.lock:
            if self.last_lidar_stamp is not None and stamp < self.last_lidar_stamp:
                self.tfv6_sweeps.clear()
            self.last_lidar_stamp = stamp

    def accumulate_tfv6_points(self, points_vehicle, stamp):
        """Motion-compensate recent TFv6 sweeps into the current ego frame."""
        pose = self.closest_ego_pose(stamp)
        if pose is None:
            rospy.logwarn_throttle(
                2.0,
                "No %s %s->%s pose within %.3f s of LiDAR; TFv6 uses the current sweep only",
                self.tf_topic,
                self.world_frame,
                self.base_frame,
                self.ego_sync_slop,
            )
            return points_vehicle

        sweep = LidarSweep(
            stamp=stamp,
            world_xy=vehicle_points_to_world(points_vehicle, pose),
            z=points_vehicle[:, 2].copy(),
        )
        with self.lock:
            self.tfv6_sweeps.append(sweep)
            oldest_allowed = stamp - self.tfv6_history_seconds
            while (
                self.tfv6_sweeps
                and self.tfv6_sweeps[0].stamp < oldest_allowed
            ):
                self.tfv6_sweeps.popleft()
            sweeps = list(self.tfv6_sweeps)

        accumulated = []
        for history_sweep in sweeps:
            current_xy = world_points_to_vehicle(history_sweep.world_xy, pose)
            accumulated.append(
                np.column_stack([current_xy, history_sweep.z]).astype(
                    np.float32,
                    copy=False,
                )
            )

        if not accumulated:
            return points_vehicle
        return np.ascontiguousarray(np.concatenate(accumulated, axis=0))

    def publish_image(self, publisher, image, encoding, source_header, frame_id):
        """Publish a NumPy image with the source timestamp."""
        try:
            message = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            message.header.seq = source_header.seq
            message.header.stamp = source_header.stamp
            message.header.frame_id = frame_id
            publisher.publish(message)
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "LiDAR image publish failed: %s", error)

    def lidar_callback(self, message):
        """Decode one cloud and publish every selected LiDAR representation."""
        output_header = copy(message.header)
        output_header.stamp = select_output_stamp(
            message.header.stamp,
            self.use_ros_time_stamp,
        )
        try:
            points_sensor = decode_point_cloud(message)
            points_vehicle = sensor_to_vehicle(
                points_sensor,
                self.calibration,
            )
        except (ValueError, TypeError) as error:
            rospy.logerr_throttle(2.0, "LiDAR decode failed: %s", error)
            return

        stamp = message.header.stamp.to_sec()
        self.reset_history_if_time_reversed(stamp)

        visual = prepare_visual_bev(
            points_vehicle,
            self.forward_range,
            self.lateral_range,
            self.resolution,
        )
        self.publish_image(
            self.visual_publisher,
            visual,
            "mono8",
            output_header,
            self.output_frame_id,
        )

        if "tfv6" in self.model_profiles:
            tfv6_points = points_vehicle
            if self.tfv6_remove_ground:
                tfv6_points = remove_ground_plane_ransac(
                    tfv6_points,
                    distance_threshold=self.tfv6_ground_distance,
                )
            accumulated_points = self.accumulate_tfv6_points(
                tfv6_points,
                stamp,
            )
            self.publish_image(
                self.model_publishers["tfv6"],
                prepare_tfv6(accumulated_points),
                "32FC1",
                output_header,
                "tfv6_lidar_bev",
            )

        if "interfuser" in self.model_profiles:
            self.publish_image(
                self.model_publishers["interfuser"],
                prepare_interfuser(points_vehicle),
                "32FC3",
                output_header,
                "interfuser_lidar_bev",
            )

        if "diffusiondrive_v2" in self.model_profiles:
            self.publish_image(
                self.model_publishers["diffusiondrive_v2"],
                prepare_diffusiondrive_v2(
                    points_vehicle,
                    self.morai_ddv2_domain,
                ),
                "32FC1",
                output_header,
                "diffusiondrive_v2_lidar_bev",
            )

        if self.show_gui:
            cv2.imshow("LiDAR BEV", visual)
            cv2.waitKey(1)

        rospy.loginfo_throttle(
            2.0,
            "LiDAR processed: %d valid points, profiles=%s",
            points_vehicle.shape[0],
            ",".join(sorted(self.model_profiles)) or "raw",
        )

    def on_shutdown(self):
        """Close the optional OpenCV window."""
        if self.show_gui:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("lidar_bev_processor", anonymous=False)
    try:
        LidarBEVProcessor()
        rospy.spin()
    except (OSError, TypeError, ValueError, rospkg.ResourceNotFound) as error:
        rospy.logfatal("lidar_bev_processor initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
