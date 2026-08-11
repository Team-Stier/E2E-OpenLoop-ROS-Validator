#!/usr/bin/env python3
"""Decode MORAI cameras and optionally publish model-specific image inputs.

The legacy outputs are always published as BGR8 with the original header:
    /image/front/processed
    /image/left/processed
    /image/right/processed

Set ``~model_profiles`` to one name, a comma-separated string, or a list.
The default ``raw`` keeps only the legacy outputs. Supported profiles:

``tfv6`` (official three-camera profile)
    /image/tfv6/processed                 RGB8,   1152x384
    Camera order in the stitched image: left, front, right.

``interfuser``
    /image/interfuser/front               32FC3, 224x224
    /image/interfuser/left                32FC3, 128x128
    /image/interfuser/right               32FC3, 128x128
    /image/interfuser/center              32FC3, 128x128
    Values use ImageNet normalization in RGB order.

``diffusiondrive_v2``
    /image/diffusiondrive_v2/processed    32FC3, 1024x256
    Camera order is left, front, right; values are RGB in [0, 1].

``sparsedrive_v2``
    /image/sparsedrive_v2/left            32FC3, 512x256
    /image/sparsedrive_v2/front           32FC3, 512x256
    /image/sparsedrive_v2/right           32FC3, 512x256
    Values use ImageNet normalization in RGB order.

Set ``~publish_sparsedrive_metadata:=true`` only for preprocessing diagnostics.
Those FOV-derived arrays are not inference inputs; SparseDrive projection
conditions come from validated profile-inline calibration in DrivingFeature.

ROS Image stores float model inputs as HWC. Inference code must transpose them
to CHW and add the batch/view dimensions expected by the PyTorch model.
SparseDriveV2 geometry is derived from 0716demo.json. Its intrinsics and
LiDAR-to-image matrices include the exact resize/crop applied to the images.
"""

import json
import math
import os
import threading
from copy import copy

import cv2
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


CAMERA_NAMES = ("front", "left", "right")

INPUT_TOPICS = {
    "front": "/image_jpeg/front/compressed",
    "left": "/image_jpeg/left/compressed",
    "right": "/image_jpeg/right/compressed",
}

LEGACY_OUTPUT_TOPICS = {
    "front": "/image/front/processed",
    "left": "/image/left/processed",
    "right": "/image/right/processed",
}

SUPPORTED_PROFILES = {
    "tfv6",
    "interfuser",
    "diffusiondrive_v2",
    "sparsedrive_v2",
}

PROFILE_ALIASES = {
    "diffusiondrive": "diffusiondrive_v2",
    "diffusiondrivev2": "diffusiondrive_v2",
    "diffusion_drive_v2": "diffusiondrive_v2",
    "sparsedrive": "sparsedrive_v2",
    "sparsedrivev2": "sparsedrive_v2",
    "sparse_drive_v2": "sparsedrive_v2",
    "transfuser_v6": "tfv6",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPARSEDRIVE_CAMERA_ORDER = ("left", "front", "right")
SPARSEDRIVE_RESIZE_WIDTH = 512
SPARSEDRIVE_RESIZE_HEIGHT = 288
SPARSEDRIVE_CROP_LEFT = 0
SPARSEDRIVE_CROP_TOP = 32
SPARSEDRIVE_OUTPUT_WIDTH = 512
SPARSEDRIVE_OUTPUT_HEIGHT = 256

SPARSEDRIVE_METADATA_TOPICS = {
    "cam_intrinsic": "/image/sparsedrive_v2/cam_intrinsic",
    "lidar2cam": "/image/sparsedrive_v2/lidar2cam",
    "cam2lidar": "/image/sparsedrive_v2/cam2lidar",
    "projection_mat": "/image/sparsedrive_v2/projection_mat",
    "img_aug_matrix": "/image/sparsedrive_v2/img_aug_matrix",
    "image_wh": "/image/sparsedrive_v2/image_wh",
}


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


def resize_image(image, size):
    """Resize with area interpolation for shrinking and linear for growing."""
    target_width, target_height = size
    interpolation = (
        cv2.INTER_AREA
        if target_width < image.shape[1] or target_height < image.shape[0]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def center_crop(image, crop_width, crop_height):
    """Return a centered crop, raising if the requested crop cannot fit."""
    image_height, image_width = image.shape[:2]
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(
            "Cannot crop {}x{} from {}x{}".format(
                crop_width,
                crop_height,
                image_width,
                image_height,
            )
        )
    x0 = (image_width - crop_width) // 2
    y0 = (image_height - crop_height) // 2
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def bgr_to_rgb(image):
    """Convert an OpenCV BGR image to contiguous RGB."""
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def imagenet_normalize(rgb_image):
    """Convert RGB uint8 HWC to ImageNet-normalized float32 HWC."""
    normalized = rgb_image.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized, dtype=np.float32)


def unit_normalize(rgb_image):
    """Convert RGB uint8 HWC to contiguous float32 HWC in [0, 1]."""
    return np.ascontiguousarray(
        rgb_image.astype(np.float32) / 255.0,
        dtype=np.float32,
    )


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
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def camera_vehicle_to_optical_rotation(calibration):
    """Return MORAI vehicle-to-camera-optical rotation.

    MORAI vehicle coordinates are x-forward, y-left, z-up. Camera optical
    coordinates are x-right, y-down, z-forward. The pitch-then-yaw order is
    fixed by the recorded MORAI camera-frame transforms for this sensor rig.
    """
    roll = math.radians(calibration["roll"])
    pitch = math.radians(calibration["pitch"])
    yaw = math.radians(calibration["yaw"])

    rotation_y = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    optical_to_vehicle = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    optical_roll = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_vehicle = (
        rotation_y @ rotation_z @ optical_to_vehicle @ optical_roll
    )
    return camera_to_vehicle.T


def load_sparsedrive_calibration(config_path):
    """Load three cameras and the reference LiDAR from MORAI JSON."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        sensor_config = json.load(config_file)

    cameras = {}
    for camera in sensor_config.get("cameraList", []):
        camera_config = camera.get("cc", {})
        topic = camera_config.get("rosConfig", {}).get("Topic", "").lower()
        camera_name = next(
            (name for name in CAMERA_NAMES if name in topic),
            None,
        )
        if camera_name is None:
            continue

        position = camera["pos"]
        rotation = camera["rot"]
        principal_point = camera_config["principalPoint"]
        cameras[camera_name] = {
            "position": np.array(
                [
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                ],
                dtype=np.float64,
            ),
            "roll": float(rotation["roll"]),
            "pitch": float(rotation["pitch"]),
            "yaw": float(rotation["yaw"]),
            "config_width": int(camera_config["cameraResWidth"]),
            "config_height": int(camera_config["cameraResHeight"]),
            "horizontal_fov": float(camera_config["cameraFOV"]),
            "principal_x": float(principal_point["x"]),
            "principal_y": float(principal_point["y"]),
        }

    missing_cameras = [
        name for name in SPARSEDRIVE_CAMERA_ORDER if name not in cameras
    ]
    if missing_cameras:
        raise ValueError(
            "Missing SparseDrive camera calibration(s) in {}: {}".format(
                config_path,
                ", ".join(missing_cameras),
            )
        )

    lidar = None
    for candidate in sensor_config.get("Lidar3DList", []):
        lidar_config = candidate.get("lc", {})
        topic = lidar_config.get("rosConfig", {}).get("Topic", "")
        if topic != "/lidar3D":
            continue
        position = candidate["pos"]
        rotation = candidate["rot"]
        lidar = {
            "position": np.array(
                [
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                ],
                dtype=np.float64,
            ),
            "rotation": rotation_matrix_from_rpy(
                float(rotation["roll"]),
                float(rotation["pitch"]),
                float(rotation["yaw"]),
            ),
        }
        break

    if lidar is None:
        raise ValueError(
            "Missing /lidar3D calibration in {}".format(config_path)
        )

    return {"cameras": cameras, "lidar": lidar}


def camera_intrinsic_matrix(calibration, image_shape):
    """Build K from the actual image size and MORAI horizontal FOV."""
    image_height, image_width = image_shape[:2]
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Camera image dimensions must be positive")

    horizontal_fov = math.radians(calibration["horizontal_fov"])
    if not 0.0 < horizontal_fov < math.pi:
        raise ValueError("Camera horizontal FOV must be between 0 and 180")

    focal_x = (image_width / 2.0) / math.tan(horizontal_fov / 2.0)
    focal_y = focal_x
    scale_x = image_width / float(calibration["config_width"])
    scale_y = image_height / float(calibration["config_height"])
    principal_x = calibration["principal_x"] * scale_x
    principal_y = calibration["principal_y"] * scale_y

    return np.array(
        [
            [focal_x, 0.0, principal_x],
            [0.0, focal_y, principal_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def sparsedrive_image_augmentation_matrix(image_shape):
    """Return the exact 512x288 resize and top-32 crop transform."""
    image_height, image_width = image_shape[:2]
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Camera image dimensions must be positive")

    scale_x = SPARSEDRIVE_RESIZE_WIDTH / float(image_width)
    scale_y = SPARSEDRIVE_RESIZE_HEIGHT / float(image_height)
    transform = np.array(
        [
            [scale_x, 0.0, -float(SPARSEDRIVE_CROP_LEFT)],
            [0.0, scale_y, -float(SPARSEDRIVE_CROP_TOP)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extended = np.eye(4, dtype=np.float64)
    extended[:3, :3] = transform
    return extended


def build_sparsedrive_geometry(images, calibration):
    """Build official SparseDrive-style geometry for the processed views."""
    geometry = {
        "cam_intrinsic": [],
        "lidar2cam": [],
        "cam2lidar": [],
        "projection_mat": [],
        "img_aug_matrix": [],
        "image_wh": [],
    }

    lidar_position = calibration["lidar"]["position"]
    lidar_to_vehicle_rotation = calibration["lidar"]["rotation"]

    for camera_name in SPARSEDRIVE_CAMERA_ORDER:
        camera = calibration["cameras"][camera_name]
        intrinsic = camera_intrinsic_matrix(
            camera,
            images[camera_name].shape,
        )
        vehicle_to_camera = camera_vehicle_to_optical_rotation(camera)

        lidar2cam = np.eye(4, dtype=np.float64)
        lidar2cam[:3, :3] = (
            vehicle_to_camera @ lidar_to_vehicle_rotation
        )
        lidar2cam[:3, 3] = vehicle_to_camera @ (
            lidar_position - camera["position"]
        )
        cam2lidar = np.linalg.inv(lidar2cam)

        intrinsic_4x4 = np.eye(4, dtype=np.float64)
        intrinsic_4x4[:3, :3] = intrinsic
        augmentation = sparsedrive_image_augmentation_matrix(
            images[camera_name].shape
        )
        adjusted_intrinsic = augmentation @ intrinsic_4x4
        projection = adjusted_intrinsic @ lidar2cam

        geometry["cam_intrinsic"].append(adjusted_intrinsic)
        geometry["lidar2cam"].append(lidar2cam)
        geometry["cam2lidar"].append(cam2lidar)
        geometry["projection_mat"].append(projection)
        geometry["img_aug_matrix"].append(augmentation)
        geometry["image_wh"].append(
            [SPARSEDRIVE_OUTPUT_WIDTH, SPARSEDRIVE_OUTPUT_HEIGHT]
        )

    stacked = {
        key: np.ascontiguousarray(np.stack(value), dtype=np.float64)
        for key, value in geometry.items()
    }
    for key, value in stacked.items():
        if not np.isfinite(value).all():
            raise ValueError(
                "SparseDrive {} contains non-finite values".format(key)
            )

    identity = np.eye(4, dtype=np.float64)
    for camera_index in range(len(SPARSEDRIVE_CAMERA_ORDER)):
        product = (
            stacked["cam2lidar"][camera_index]
            @ stacked["lidar2cam"][camera_index]
        )
        if not np.allclose(product, identity, atol=1.0e-8):
            raise ValueError("SparseDrive camera extrinsics are not invertible")

    return stacked


def numpy_to_multiarray(array, labels):
    """Serialize a NumPy tensor into a row-major Float64MultiArray."""
    array = np.ascontiguousarray(array, dtype=np.float64)
    if len(labels) != array.ndim:
        raise ValueError("MultiArray labels must match array dimensions")

    message = Float64MultiArray()
    message.layout.dim = []
    for index, (label, size) in enumerate(zip(labels, array.shape)):
        stride = int(np.prod(array.shape[index:], dtype=np.int64))
        message.layout.dim.append(
            MultiArrayDimension(
                label=label,
                size=int(size),
                stride=stride,
            )
        )
    message.layout.data_offset = 0
    message.data = array.reshape(-1).tolist()
    return message


def prepare_tfv6(images):
    """Build TFv6's three 384x384 camera strip in left/front/right order."""
    views = []
    for camera_name in ("left", "front", "right"):
        rgb = bgr_to_rgb(images[camera_name])
        square_size = min(rgb.shape[:2])
        square = center_crop(rgb, square_size, square_size)
        views.append(resize_image(square, (384, 384)))
    return np.ascontiguousarray(np.concatenate(views, axis=1))


def prepare_interfuser(images):
    """Apply the official InterFuser inference resize/crop operations."""
    front_rgb = bgr_to_rgb(images["front"])
    left_rgb = bgr_to_rgb(images["left"])
    right_rgb = bgr_to_rgb(images["right"])

    front = center_crop(resize_image(front_rgb, (341, 256)), 224, 224)
    left = center_crop(resize_image(left_rgb, (195, 146)), 128, 128)
    right = center_crop(resize_image(right_rgb, (195, 146)), 128, 128)
    center = center_crop(front_rgb, 128, 128)

    return {
        "front": imagenet_normalize(front),
        "left": imagenet_normalize(left),
        "right": imagenet_normalize(right),
        "center": imagenet_normalize(center),
    }


def prepare_diffusiondrive_v2(images, morai_domain=False):
    """Build DiffusionDriveV2's 1024x256 left/front/right camera strip."""
    if morai_domain:
        resized = [
            cv2.resize(
                bgr_to_rgb(images[name]),
                size,
                interpolation=cv2.INTER_AREA,
            )
            for name, size in (
                ("left", (341, 256)),
                ("front", (455, 256)),
                ("right", (341, 256)),
            )
        ]
        return unit_normalize(np.concatenate(resized, axis=1)[:, 56:1080])

    canonical = {
        name: cv2.resize(
            bgr_to_rgb(images[name]),
            (1920, 1080),
            interpolation=cv2.INTER_LINEAR,
        )
        for name in CAMERA_NAMES
    }
    stitched = np.concatenate(
        [
            canonical["left"][28:-28, 416:-416],
            canonical["front"][28:-28],
            canonical["right"][28:-28, 416:-416],
        ],
        axis=1,
    )
    return unit_normalize(
        cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR)
    )


def select_output_stamp(source_stamp, use_ros_time_stamp):
    """명시적으로 요청한 경우에만 bag replay 시각을 선택한다."""
    return rospy.Time.now() if use_ros_time_stamp else source_stamp


def prepare_sparsedrive_v2(images):
    """Build SparseDriveV2's three normalized 512x256 RGB views."""
    prepared = {}
    for camera_name in SPARSEDRIVE_CAMERA_ORDER:
        rgb = bgr_to_rgb(images[camera_name])
        # Official test pipeline: resize 1920x1080 to 512x288, then retain
        # the bottom 256 rows. Applying the same target geometry also makes
        # the three MORAI source resolutions consistent for the checkpoint.
        resized = resize_image(
            rgb,
            (SPARSEDRIVE_RESIZE_WIDTH, SPARSEDRIVE_RESIZE_HEIGHT),
        )
        cropped = resized[
            SPARSEDRIVE_CROP_TOP:
            SPARSEDRIVE_CROP_TOP + SPARSEDRIVE_OUTPUT_HEIGHT,
            SPARSEDRIVE_CROP_LEFT:
            SPARSEDRIVE_CROP_LEFT + SPARSEDRIVE_OUTPUT_WIDTH,
        ]
        prepared[camera_name] = imagenet_normalize(cropped)
    return prepared


class ImageAdaptor:
    """Decode compressed cameras and publish legacy/model-specific images."""

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = {}
        self.generation = {name: 0 for name in CAMERA_NAMES}
        self.last_profile_generation = {name: 0 for name in CAMERA_NAMES}
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.1))
        self.use_ros_time_stamp = bool(
            rospy.get_param("~use_ros_time_stamp", False)
        )
        self.morai_ddv2_domain = bool(
            rospy.get_param("~morai_ddv2_domain", False)
        )

        if self.sync_slop < 0.0:
            raise ValueError("~sync_slop must be non-negative")

        profile_parameter = rospy.get_param(
            "~model_profiles",
            rospy.get_param("~model_profile", "raw"),
        )
        self.model_profiles = parse_model_profiles(profile_parameter)
        self.sparsedrive_calibration = None
        self.sparsedrive_metadata_publishers = {}
        self.sparsedrive_metadata_shape_key = None
        self.publish_sparsedrive_metadata_enabled = bool(
            rospy.get_param("~publish_sparsedrive_metadata", False)
        )

        self.legacy_publishers = {
            camera_name: rospy.Publisher(
                topic,
                Image,
                queue_size=1,
            )
            for camera_name, topic in LEGACY_OUTPUT_TOPICS.items()
        }
        self.profile_publishers = self.create_profile_publishers()

        if (
            "sparsedrive_v2" in self.model_profiles
            and self.publish_sparsedrive_metadata_enabled
        ):
            package_path = rospkg.RosPack().get_path("Validator_pkg")
            default_config_path = os.path.join(
                package_path,
                "config",
                "0716demo.json",
            )
            config_path = rospy.get_param(
                "~config_path",
                default_config_path,
            )
            self.sparsedrive_calibration = load_sparsedrive_calibration(
                config_path
            )
            self.sparsedrive_metadata_publishers = {
                name: rospy.Publisher(
                    topic,
                    Float64MultiArray,
                    queue_size=1,
                    latch=True,
                )
                for name, topic in SPARSEDRIVE_METADATA_TOPICS.items()
            }

        self.subscribers = {
            camera_name: rospy.Subscriber(
                INPUT_TOPICS[camera_name],
                CompressedImage,
                self.image_callback,
                callback_args=camera_name,
                queue_size=1,
                buff_size=2**24,
            )
            for camera_name in CAMERA_NAMES
        }

        rospy.loginfo("image_adaptor node initialized")
        rospy.loginfo(
            "Legacy BGR8 outputs enabled: %s",
            ", ".join(LEGACY_OUTPUT_TOPICS.values()),
        )
        if self.model_profiles:
            rospy.loginfo(
                "Model image profiles enabled: %s",
                ", ".join(sorted(self.model_profiles)),
            )
        else:
            rospy.loginfo(
                "Model image profiles disabled (set ~model_profiles to enable)",
            )

    def create_profile_publishers(self):
        """Create only the publishers needed by the selected profiles."""
        publishers = {}

        if "tfv6" in self.model_profiles:
            publishers["tfv6"] = rospy.Publisher(
                "/image/tfv6/processed",
                Image,
                queue_size=1,
            )

        if "interfuser" in self.model_profiles:
            publishers["interfuser"] = {
                name: rospy.Publisher(
                    "/image/interfuser/{}".format(name),
                    Image,
                    queue_size=1,
                )
                for name in ("front", "left", "right", "center")
            }

        if "diffusiondrive_v2" in self.model_profiles:
            publishers["diffusiondrive_v2"] = rospy.Publisher(
                "/image/diffusiondrive_v2/processed",
                Image,
                queue_size=1,
            )

        if "sparsedrive_v2" in self.model_profiles:
            publishers["sparsedrive_v2"] = {
                name: rospy.Publisher(
                    "/image/sparsedrive_v2/{}".format(name),
                    Image,
                    queue_size=1,
                )
                for name in ("left", "front", "right")
            }

        return publishers

    def publish_sparsedrive_metadata(self, images):
        """Publish geometry matching the current SparseDrive image sizes."""
        shape_key = tuple(
            (
                camera_name,
                images[camera_name].shape[1],
                images[camera_name].shape[0],
            )
            for camera_name in SPARSEDRIVE_CAMERA_ORDER
        )
        if shape_key == self.sparsedrive_metadata_shape_key:
            return

        geometry = build_sparsedrive_geometry(
            images,
            self.sparsedrive_calibration,
        )
        matrix_labels = (
            "camera[left,front,right]",
            "row",
            "column",
        )
        for name in (
            "cam_intrinsic",
            "lidar2cam",
            "cam2lidar",
            "projection_mat",
            "img_aug_matrix",
        ):
            self.sparsedrive_metadata_publishers[name].publish(
                numpy_to_multiarray(geometry[name], matrix_labels)
            )
        self.sparsedrive_metadata_publishers["image_wh"].publish(
            numpy_to_multiarray(
                geometry["image_wh"],
                (
                    "camera[left,front,right]",
                    "coordinate[width,height]",
                ),
            )
        )
        self.sparsedrive_metadata_shape_key = shape_key
        rospy.loginfo(
            "SparseDrive geometry published for source shapes: %s",
            ", ".join(
                "{}={}x{}".format(name, width, height)
                for name, width, height in shape_key
            ),
        )

    @staticmethod
    def decode_compressed_image(msg):
        """Decode one sensor_msgs/CompressedImage into OpenCV BGR."""
        np_array = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    def publish_image(self, publisher, image, encoding, header, frame_id=None):
        """Publish an OpenCV image while preserving the synchronized stamp."""
        try:
            output_msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            output_msg.header.seq = header.seq
            output_msg.header.stamp = header.stamp
            output_msg.header.frame_id = (
                header.frame_id if frame_id is None else frame_id
            )
            publisher.publish(output_msg)
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "Image publish failed: %s", error)

    def image_callback(self, msg, camera_name):
        """Publish the legacy image and trigger a synchronized model batch."""
        output_header = copy(msg.header)
        output_header.stamp = select_output_stamp(
            msg.header.stamp,
            self.use_ros_time_stamp,
        )
        image = self.decode_compressed_image(msg)
        if image is None:
            rospy.logwarn_throttle(
                2.0,
                "Failed to decode %s compressed image",
                camera_name,
            )
            return

        self.publish_image(
            self.legacy_publishers[camera_name],
            image,
            "bgr8",
            output_header,
        )

        snapshot = None
        if self.model_profiles:
            with self.lock:
                self.generation[camera_name] += 1
                self.latest[camera_name] = (
                    image,
                    msg.header,
                    self.generation[camera_name],
                    output_header,
                )
                snapshot = self.get_synchronized_snapshot()

        if snapshot is not None:
            self.publish_model_profiles(snapshot)

        rospy.loginfo_throttle(
            5.0,
            "%s camera processed: %dx%d",
            camera_name,
            image.shape[1],
            image.shape[0],
        )

    def get_synchronized_snapshot(self):
        """Return one fresh three-camera group when stamps are close enough."""
        if any(name not in self.latest for name in CAMERA_NAMES):
            return None

        if any(
            self.latest[name][2] <= self.last_profile_generation[name]
            for name in CAMERA_NAMES
        ):
            return None

        stamps = [
            self.latest[name][1].stamp.to_sec()
            for name in CAMERA_NAMES
        ]
        stamp_spread = max(stamps) - min(stamps)
        if stamp_spread > self.sync_slop:
            rospy.logwarn_throttle(
                2.0,
                "Camera stamps differ by %.3f s (limit %.3f s)",
                stamp_spread,
                self.sync_slop,
            )
            return None

        snapshot = {
            name: self.latest[name]
            for name in CAMERA_NAMES
        }
        self.last_profile_generation = {
            name: snapshot[name][2]
            for name in CAMERA_NAMES
        }
        return snapshot

    def publish_model_profiles(self, snapshot):
        """Prepare and publish every enabled model profile for one frame."""
        images = {name: snapshot[name][0] for name in CAMERA_NAMES}
        latest_name = max(
            CAMERA_NAMES,
            key=lambda name: snapshot[name][3].stamp.to_nsec(),
        )
        header = snapshot[latest_name][3]

        try:
            if "tfv6" in self.model_profiles:
                self.publish_image(
                    self.profile_publishers["tfv6"],
                    prepare_tfv6(images),
                    "rgb8",
                    header,
                    "tfv6_multi_camera",
                )

            if "interfuser" in self.model_profiles:
                prepared = prepare_interfuser(images)
                for name, image in prepared.items():
                    self.publish_image(
                        self.profile_publishers["interfuser"][name],
                        image,
                        "32FC3",
                        header,
                        "interfuser_{}".format(name),
                    )

            if "diffusiondrive_v2" in self.model_profiles:
                self.publish_image(
                    self.profile_publishers["diffusiondrive_v2"],
                    prepare_diffusiondrive_v2(
                        images,
                        self.morai_ddv2_domain,
                    ),
                    "32FC3",
                    header,
                    "diffusiondrive_v2_multi_camera",
                )

            if "sparsedrive_v2" in self.model_profiles:
                if self.publish_sparsedrive_metadata_enabled:
                    self.publish_sparsedrive_metadata(images)
                prepared = prepare_sparsedrive_v2(images)
                for name, image in prepared.items():
                    self.publish_image(
                        self.profile_publishers["sparsedrive_v2"][name],
                        image,
                        "32FC3",
                        header,
                        "sparsedrive_v2_{}".format(name),
                    )
        except (cv2.error, ValueError) as error:
            rospy.logerr_throttle(
                2.0,
                "Model image preprocessing failed: %s",
                error,
            )
            return

        rospy.loginfo_throttle(
            2.0,
            "Published synchronized model image profiles: %s",
            ", ".join(sorted(self.model_profiles)),
        )


def main():
    rospy.init_node("image_adaptor", anonymous=False)
    try:
        ImageAdaptor()
        rospy.spin()
    except (TypeError, ValueError) as error:
        rospy.logfatal("image_adaptor initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
