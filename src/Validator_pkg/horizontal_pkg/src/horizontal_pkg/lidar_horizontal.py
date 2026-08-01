"""Create completely roll/pitch-leveled LiDAR clouds in an offline ROS bag."""

import argparse
from bisect import bisect_left
import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from horizontal_pkg import imu_filtering
from horizontal_pkg import imu_orientation_analyzer


class HorizontalizationError(RuntimeError):
    """Raised when offline horizontalization cannot satisfy its contract."""


def rotation_matrix_xyz(roll_rad, pitch_rad, yaw_rad=0.0):
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    angles = tuple(float(value) for value in (roll_rad, pitch_rad, yaw_rad))
    if not all(math.isfinite(value) for value in angles):
        raise HorizontalizationError("rotation angles must be finite")
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _validated_rotation(matrix):
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise HorizontalizationError("sensor extrinsic must be a finite rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise HorizontalizationError("sensor extrinsic must be a rotation matrix")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise HorizontalizationError("sensor extrinsic must be a proper rotation")
    return rotation


def level_rotation(roll_rad, pitch_rad, R_imu_from_lidar):
    """Return the LiDAR-to-yaw-level rotation for filtered IMU attitude."""
    lidar_to_imu = _validated_rotation(R_imu_from_lidar)
    return rotation_matrix_xyz(roll_rad, pitch_rad) @ lidar_to_imu


@dataclass(frozen=True)
class AttitudeTimeline:
    """Filtered roll/pitch values indexed by strictly increasing sensor stamps."""

    stamps_ns: tuple
    roll_rad: tuple
    pitch_rad: tuple

    def __post_init__(self):
        if not (
            len(self.stamps_ns) == len(self.roll_rad) == len(self.pitch_rad)
        ):
            raise HorizontalizationError("attitude timeline lengths must match")
        if not self.stamps_ns:
            raise HorizontalizationError("attitude timeline must not be empty")
        if any(
            current <= previous
            for previous, current in zip(self.stamps_ns, self.stamps_ns[1:])
        ):
            raise HorizontalizationError(
                "attitude stamps must be strictly increasing"
            )
        values = tuple(self.roll_rad) + tuple(self.pitch_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise HorizontalizationError("attitude angles must be finite")

    def interpolate(self, stamp_ns, max_gap_s):
        """Interpolate angles at one stamp or return None when alignment is unsafe."""
        try:
            stamp = int(stamp_ns)
            max_gap = float(max_gap_s)
        except (TypeError, ValueError) as error:
            raise HorizontalizationError("timestamp and max gap must be numeric") from error
        if not math.isfinite(max_gap) or max_gap <= 0.0:
            raise HorizontalizationError("max IMU gap must be positive and finite")

        index = bisect_left(self.stamps_ns, stamp)
        if index < len(self.stamps_ns) and self.stamps_ns[index] == stamp:
            return float(self.roll_rad[index]), float(self.pitch_rad[index])
        if index == 0 or index == len(self.stamps_ns):
            return None

        before = int(self.stamps_ns[index - 1])
        after = int(self.stamps_ns[index])
        limit_ns = max_gap * 1e9
        if (
            after - before > limit_ns
            or stamp - before > limit_ns
            or after - stamp > limit_ns
        ):
            return None
        fraction = (stamp - before) / float(after - before)
        roll = float(self.roll_rad[index - 1]) + fraction * (
            float(self.roll_rad[index]) - float(self.roll_rad[index - 1])
        )
        pitch = float(self.pitch_rad[index - 1]) + fraction * (
            float(self.pitch_rad[index]) - float(self.pitch_rad[index - 1])
        )
        return roll, pitch


@dataclass(frozen=True)
class RigidTransform:
    """One validated sensor pose expressed in its parent frame."""

    translation_m: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class Extrinsics:
    """Bag-derived IMU/LiDAR extrinsics required for leveling."""

    T_base_from_imu: RigidTransform
    T_base_from_lidar: RigidTransform
    translation_imu_from_lidar_m: np.ndarray
    R_imu_from_lidar: np.ndarray


@dataclass(frozen=True)
class ImuStats:
    """Quality counters collected while preparing the attitude timeline."""

    total_messages: int
    valid_unique_messages: int
    duplicate_replacements: int
    zero_stamp_count: int
    invalid_quaternion_count: int
    normalization_warning_count: int


@dataclass(frozen=True)
class ProcessingStats:
    """Counters and geometry from one successful bag conversion."""

    imu_stats: ImuStats
    extrinsics: Extrinsics
    lidar_input_count: int
    lidar_written_count: int
    lidar_skipped_alignment_count: int
    elapsed_s: float


def _normalized_frame(frame):
    return str(frame).lstrip("/")


def _normalized_quaternion(values):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise HorizontalizationError("transform quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise HorizontalizationError("transform quaternion must be nonzero")
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _quaternion_matrix(quaternion_xyzw):
    x, y, z, w = _normalized_quaternion(quaternion_xyzw)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _summarize_transform(samples, child_frame):
    if not samples:
        raise HorizontalizationError(
            "bag TF is missing child frame {}".format(child_frame)
        )
    translations = np.asarray([sample[0] for sample in samples], dtype=float)
    if not np.all(np.isfinite(translations)):
        raise HorizontalizationError(
            "{} translation must be finite".format(child_frame)
        )
    quaternions = np.asarray(
        [_normalized_quaternion(sample[1]) for sample in samples],
        dtype=float,
    )
    reference = quaternions[0]
    quaternions = np.asarray(
        [value if np.dot(value, reference) >= 0.0 else -value for value in quaternions]
    )
    translation = np.median(translations, axis=0)
    quaternion = _normalized_quaternion(np.median(quaternions, axis=0))
    translation_spread = np.max(np.linalg.norm(translations - translation, axis=1))
    if translation_spread > 0.005:
        raise HorizontalizationError(
            "{} translation is unstable ({:.6f} m)".format(
                child_frame,
                translation_spread,
            )
        )
    dots = np.clip(np.abs(quaternions @ quaternion), 0.0, 1.0)
    angular_spread_deg = float(np.degrees(np.max(2.0 * np.arccos(dots))))
    if angular_spread_deg > 0.1:
        raise HorizontalizationError(
            "{} rotation is unstable ({:.6f} deg)".format(
                child_frame,
                angular_spread_deg,
            )
        )
    return RigidTransform(translation, quaternion)


def extract_extrinsics(tf_messages, base_frame, imu_frame, lidar_frame):
    """Extract stable median sensor transforms from bag TF messages."""
    base = _normalized_frame(base_frame)
    targets = {
        _normalized_frame(imu_frame): [],
        _normalized_frame(lidar_frame): [],
    }
    for message in tf_messages:
        for transform in message.transforms:
            parent = _normalized_frame(transform.header.frame_id)
            child = _normalized_frame(transform.child_frame_id)
            if parent != base or child not in targets:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            targets[child].append(
                (
                    (translation.x, translation.y, translation.z),
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                )
            )

    imu = _summarize_transform(targets[_normalized_frame(imu_frame)], imu_frame)
    lidar = _summarize_transform(
        targets[_normalized_frame(lidar_frame)],
        lidar_frame,
    )
    R_base_from_imu = _quaternion_matrix(imu.quaternion_xyzw)
    R_base_from_lidar = _quaternion_matrix(lidar.quaternion_xyzw)
    R_imu_from_lidar = R_base_from_imu.T @ R_base_from_lidar
    translation_imu_from_lidar = R_base_from_imu.T @ (
        lidar.translation_m - imu.translation_m
    )
    return Extrinsics(
        T_base_from_imu=imu,
        T_base_from_lidar=lidar,
        translation_imu_from_lidar_m=translation_imu_from_lidar,
        R_imu_from_lidar=R_imu_from_lidar,
    )


def prepare_attitude(
    imu_messages,
    cutoff_hz=4.0,
    max_imu_gap_s=0.10,
    expected_frame="IMU-1",
):
    """Validate, consolidate, and zero-phase filter bag IMU roll/pitch."""
    rows = {}
    total = 0
    duplicates = 0
    zero_stamps = 0
    invalid = 0
    normalization_warnings = 0
    for message in imu_messages:
        total += 1
        if _normalized_frame(message.header.frame_id) != _normalized_frame(
            expected_frame
        ):
            raise HorizontalizationError(
                "unexpected IMU frame {!r}; expected {!r}".format(
                    message.header.frame_id,
                    expected_frame,
                )
            )
        stamp = int(message.header.stamp.to_nsec())
        if stamp == 0:
            zero_stamps += 1
            continue
        orientation = message.orientation
        try:
            angles = imu_orientation_analyzer.quaternion_to_roll_pitch(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        except imu_orientation_analyzer.InvalidQuaternionError:
            invalid += 1
            continue
        if angles.normalization_warning:
            normalization_warnings += 1
        if stamp in rows:
            duplicates += 1
        rows[stamp] = (angles.roll_rad, angles.pitch_rad)

    stamps = sorted(rows)
    if not stamps:
        raise HorizontalizationError("no valid IMU orientation samples")
    if not math.isfinite(float(max_imu_gap_s)) or float(max_imu_gap_s) <= 0.0:
        raise HorizontalizationError("max IMU gap must be positive and finite")
    limit_ns = float(max_imu_gap_s) * 1e9
    segments = []
    start = 0
    for index in range(1, len(stamps)):
        if stamps[index] - stamps[index - 1] > limit_ns:
            segments.append(stamps[start:index])
            start = index
    segments.append(stamps[start:])
    filtered_stamps = []
    filtered_roll = []
    filtered_pitch = []
    for segment in segments:
        if len(segment) < 16:
            continue
        try:
            filtered = imu_filtering.filter_orientation(
                segment,
                [math.degrees(rows[stamp][0]) for stamp in segment],
                [math.degrees(rows[stamp][1]) for stamp in segment],
                cutoff_hz,
            )
        except imu_filtering.FilteringError as error:
            raise HorizontalizationError(
                "IMU filtering failed: {}".format(error)
            ) from error
        filtered_stamps.extend(segment)
        filtered_roll.extend(np.radians(filtered.roll_deg_filtered))
        filtered_pitch.extend(np.radians(filtered.pitch_deg_filtered))
    if not filtered_stamps:
        raise HorizontalizationError(
            "no IMU segment is long enough for zero-phase filtering"
        )
    timeline = AttitudeTimeline(
        stamps_ns=tuple(filtered_stamps),
        roll_rad=tuple(filtered_roll),
        pitch_rad=tuple(filtered_pitch),
    )
    return timeline, ImuStats(
        total_messages=total,
        valid_unique_messages=len(stamps),
        duplicate_replacements=duplicates,
        zero_stamp_count=zero_stamps,
        invalid_quaternion_count=invalid,
        normalization_warning_count=normalization_warnings,
    )


def _xyz_fields(message):
    datatype_sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
    fields = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if field.name in fields:
                raise HorizontalizationError("PointCloud2 XYZ fields must be unique")
            fields[field.name] = field
    if set(fields) != {"x", "y", "z"}:
        raise HorizontalizationError("PointCloud2 requires x, y, and z fields")
    for field in fields.values():
        if field.datatype != 7 or field.count != 1:
            raise HorizontalizationError("PointCloud2 XYZ fields must be count-one FLOAT32")
        if field.offset < 0 or field.offset + 4 > message.point_step:
            raise HorizontalizationError("PointCloud2 XYZ field offset is invalid")
    xyz_ranges = {
        name: range(field.offset, field.offset + 4)
        for name, field in fields.items()
    }
    for field in message.fields:
        size = datatype_sizes.get(field.datatype)
        if size is None or field.count < 1:
            continue
        field_bytes = set(range(field.offset, field.offset + size * field.count))
        for name, xyz_range in xyz_ranges.items():
            if field.name == name:
                continue
            if field_bytes.intersection(xyz_range):
                raise HorizontalizationError(
                    "PointCloud2 fields must not overlap XYZ bytes"
                )
    return fields


def horizontalize_cloud(message, rotation, output_frame):
    """Rotate PointCloud2 XYZ in-place on a copied binary payload."""
    transform = _validated_rotation(rotation)
    if not str(output_frame):
        raise HorizontalizationError("output frame must not be empty")
    if message.height < 1 or message.width < 1 or message.point_step < 1:
        raise HorizontalizationError("PointCloud2 dimensions and point_step are invalid")
    if message.row_step < message.point_step * message.width:
        raise HorizontalizationError("PointCloud2 row_step is inconsistent")
    if len(message.data) != message.row_step * message.height:
        raise HorizontalizationError("PointCloud2 payload length is inconsistent")
    fields = _xyz_fields(message)
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype(endian + "f4")
    payload = bytearray(message.data)
    for row in range(message.height):
        views = {}
        for name in ("x", "y", "z"):
            views[name] = np.ndarray(
                shape=(message.width,),
                dtype=dtype,
                buffer=payload,
                offset=row * message.row_step + fields[name].offset,
                strides=(message.point_step,),
            )
        points = np.column_stack(
            (views["x"].copy(), views["y"].copy(), views["z"].copy())
        )
        finite = np.all(np.isfinite(points), axis=1)
        rotated = np.full(points.shape, np.nan, dtype=float)
        rotated[finite] = points[finite] @ transform.T
        views["x"][:] = rotated[:, 0]
        views["y"][:] = rotated[:, 1]
        views["z"][:] = rotated[:, 2]

    output = copy.deepcopy(message)
    output.header.frame_id = str(output_frame)
    output.data = bytes(payload)
    return output


def _default_bag_factory(path, mode, compression=None):
    import rosbag

    if compression is None:
        return rosbag.Bag(str(path), mode)
    return rosbag.Bag(str(path), mode, compression=compression)


def _temporary_bag_path(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".{}-".format(destination.name),
        suffix=".bag",
        dir=str(destination.parent),
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def process_bag(
    input_bag,
    output_bag,
    imu_topic="/imu",
    lidar_topic="/lidar3D",
    tf_topic="/tf",
    output_topic="/lidar3D/horizontal",
    output_frame="Lidar3D-4-horizontal",
    cutoff_hz=4.0,
    max_imu_gap_s=0.10,
    base_frame="base_link",
    imu_frame="IMU-1",
    lidar_frame="Lidar3D-4",
    bag_factory=None,
):
    """Copy a ROS bag and add timestamp-aligned horizontal LiDAR clouds."""
    started = time.perf_counter()
    input_path = Path(input_bag).expanduser().resolve(strict=False)
    output_path = Path(output_bag).expanduser().resolve(strict=False)
    if input_path == output_path:
        raise HorizontalizationError("input and output bag paths must differ")
    if output_topic in {imu_topic, lidar_topic, tf_topic}:
        raise HorizontalizationError(
            "output topic must differ from IMU, LiDAR, and TF topics"
        )
    if bag_factory is None:
        if not input_path.is_file():
            raise HorizontalizationError(
                "input bag does not exist: {}".format(input_path)
            )
        bag_factory = _default_bag_factory

    imu_messages = []
    tf_messages = []
    output_topic_present = False
    try:
        with bag_factory(str(input_path), "r") as source:
            for topic, message, _ in source.read_messages(
                topics=[imu_topic, tf_topic, output_topic]
            ):
                if topic == imu_topic:
                    imu_messages.append(message)
                elif topic == tf_topic:
                    tf_messages.append(message)
                elif topic == output_topic:
                    output_topic_present = True
    except HorizontalizationError:
        raise
    except Exception as error:
        raise HorizontalizationError(
            "failed to read input bag {}: {}".format(input_path, error)
        ) from error

    if not imu_messages:
        raise HorizontalizationError(
            "input bag is missing IMU topic {}".format(imu_topic)
        )
    if not tf_messages:
        raise HorizontalizationError(
            "input bag is missing TF topic {}".format(tf_topic)
        )
    if output_topic_present:
        raise HorizontalizationError(
            "input bag already contains output topic {}".format(output_topic)
        )
    timeline, imu_stats = prepare_attitude(
        imu_messages,
        cutoff_hz,
        max_imu_gap_s=max_imu_gap_s,
        expected_frame=imu_frame,
    )
    extrinsics = extract_extrinsics(
        tf_messages,
        base_frame,
        imu_frame,
        lidar_frame,
    )

    temporary = None
    lidar_input_count = 0
    lidar_written_count = 0
    lidar_skipped_alignment_count = 0
    try:
        temporary = _temporary_bag_path(output_path)
        with bag_factory(str(temporary), "w", compression="lz4") as destination:
            with bag_factory(str(input_path), "r") as source:
                for topic, message, record_time, connection_header in source.read_messages(
                    return_connection_header=True
                ):
                    destination.write(
                        topic,
                        message,
                        record_time,
                        connection_header=connection_header,
                    )
                    if topic != lidar_topic:
                        continue
                    lidar_input_count += 1
                    if _normalized_frame(message.header.frame_id) != _normalized_frame(
                        lidar_frame
                    ):
                        raise HorizontalizationError(
                            "unexpected LiDAR frame {!r}; expected {!r}".format(
                                message.header.frame_id,
                                lidar_frame,
                            )
                        )
                    stamp_ns = int(message.header.stamp.to_nsec())
                    attitude = timeline.interpolate(stamp_ns, max_imu_gap_s)
                    if attitude is None:
                        lidar_skipped_alignment_count += 1
                        continue
                    roll_rad, pitch_rad = attitude
                    rotation = level_rotation(
                        roll_rad,
                        pitch_rad,
                        extrinsics.R_imu_from_lidar,
                    )
                    horizontal_cloud = horizontalize_cloud(
                        message,
                        rotation,
                        output_frame,
                    )
                    destination.write(output_topic, horizontal_cloud, record_time)
                    lidar_written_count += 1
        if lidar_written_count == 0:
            raise HorizontalizationError(
                "horizontalization produced zero output LiDAR clouds"
            )
        os.replace(str(temporary), str(output_path))
        temporary = None
    except HorizontalizationError:
        raise
    except Exception as error:
        raise HorizontalizationError(
            "failed to write output bag {}: {}".format(output_path, error)
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    return ProcessingStats(
        imu_stats=imu_stats,
        extrinsics=extrinsics,
        lidar_input_count=lidar_input_count,
        lidar_written_count=lidar_written_count,
        lidar_skipped_alignment_count=lidar_skipped_alignment_count,
        elapsed_s=time.perf_counter() - started,
    )


def build_argument_parser():
    """Build the offline bag horizontalization parser."""
    parser = argparse.ArgumentParser(
        description="Add IMU-leveled LiDAR PointCloud2 messages to a ROS1 bag."
    )
    parser.add_argument("--input-bag", required=True)
    parser.add_argument("--output-bag", required=True)
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--lidar-topic", default="/lidar3D")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--output-topic", default="/lidar3D/horizontal")
    parser.add_argument("--output-frame", default="Lidar3D-4-horizontal")
    parser.add_argument("--cutoff-hz", type=float, default=4.0)
    parser.add_argument("--max-imu-gap-s", type=float, default=0.10)
    return parser


def main(argv=None, bag_factory=None):
    """Run the offline horizontalization CLI."""
    arguments = build_argument_parser().parse_args(argv)
    try:
        stats = process_bag(
            arguments.input_bag,
            arguments.output_bag,
            imu_topic=arguments.imu_topic,
            lidar_topic=arguments.lidar_topic,
            tf_topic=arguments.tf_topic,
            output_topic=arguments.output_topic,
            output_frame=arguments.output_frame,
            cutoff_hz=arguments.cutoff_hz,
            max_imu_gap_s=arguments.max_imu_gap_s,
            bag_factory=bag_factory,
        )
    except HorizontalizationError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    print(
        "imu_total={} valid_unique={} duplicates={} invalid={} zero_stamp={}".format(
            stats.imu_stats.total_messages,
            stats.imu_stats.valid_unique_messages,
            stats.imu_stats.duplicate_replacements,
            stats.imu_stats.invalid_quaternion_count,
            stats.imu_stats.zero_stamp_count,
        )
    )
    print(
        "lidar_input={} written={} skipped_alignment={}".format(
            stats.lidar_input_count,
            stats.lidar_written_count,
            stats.lidar_skipped_alignment_count,
        )
    )
    print(
        "imu_from_lidar_translation_m={}".format(
            stats.extrinsics.translation_imu_from_lidar_m.tolist()
        )
    )
    print("elapsed_s={:.6f}".format(stats.elapsed_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
