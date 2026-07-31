"""Extract roll and pitch from IMU orientation quaternions in a ROS bag."""

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from dataclasses import dataclass


class AnalyzerError(RuntimeError):
    """Base error for rejected analyzer input or output."""


class InvalidQuaternionError(AnalyzerError):
    """Raised when an orientation quaternion cannot represent a rotation."""


@dataclass(frozen=True)
class OrientationAngles:
    """Roll and pitch obtained from one normalized quaternion."""

    quaternion_norm: float
    roll_rad: float
    pitch_rad: float
    normalization_warning: bool


@dataclass(frozen=True)
class RawOrientationSample:
    """Raw bag and IMU header data required for one exported sample."""

    header_stamp_ns: int
    bag_record_stamp_ns: int
    seq: int
    frame_id: str
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class OrientationRow:
    """Validated orientation values for one unique source timestamp."""

    raw: RawOrientationSample
    quaternion_norm: float
    roll_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class AnalysisStats:
    """Input-quality counters collected while consolidating IMU samples."""

    total_messages: int
    duplicate_replacements: int
    timestamp_reversals: int
    zero_stamp_count: int
    invalid_quaternion_count: int
    normalization_warning_count: int
    frame_ids: tuple


@dataclass(frozen=True)
class AnalysisResult:
    """Chronological IMU rows and their source-quality counters."""

    rows: tuple
    stats: AnalysisStats


CSV_FIELDS = (
    "header_stamp_ns",
    "elapsed_s",
    "bag_record_stamp_ns",
    "seq",
    "frame_id",
    "qx",
    "qy",
    "qz",
    "qw",
    "quaternion_norm",
    "roll_rad",
    "pitch_rad",
    "roll_deg",
    "pitch_deg",
)


def quaternion_to_roll_pitch(x, y, z, w):
    """Convert a ROS-order quaternion into roll and pitch radians."""
    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise InvalidQuaternionError(
            "quaternion contains a non-finite component"
        )

    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-6:
        raise InvalidQuaternionError(
            "quaternion norm is smaller than 1e-6"
        )

    x, y, z, w = (value / norm for value in values)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_pitch = max(
        -1.0,
        min(1.0, 2.0 * (w * y - z * x)),
    )
    pitch = math.asin(sin_pitch)

    return OrientationAngles(
        quaternion_norm=norm,
        roll_rad=roll,
        pitch_rad=pitch,
        normalization_warning=abs(norm - 1.0) > 1e-3,
    )


def consolidate_samples(samples):
    """Validate and consolidate IMU samples by original source timestamp."""
    rows_by_stamp = {}
    total_messages = 0
    duplicate_replacements = 0
    timestamp_reversals = 0
    zero_stamp_count = 0
    invalid_quaternion_count = 0
    normalization_warning_count = 0
    frame_ids = set()
    previous_nonzero_stamp = None

    for sample in samples:
        total_messages += 1
        stamp = int(sample.header_stamp_ns)
        frame_ids.add(str(sample.frame_id))

        if stamp == 0:
            zero_stamp_count += 1
            continue

        if (
            previous_nonzero_stamp is not None
            and stamp < previous_nonzero_stamp
        ):
            timestamp_reversals += 1
        previous_nonzero_stamp = stamp

        try:
            angles = quaternion_to_roll_pitch(
                sample.qx,
                sample.qy,
                sample.qz,
                sample.qw,
            )
        except InvalidQuaternionError:
            invalid_quaternion_count += 1
            continue

        if angles.normalization_warning:
            normalization_warning_count += 1
        if stamp in rows_by_stamp:
            duplicate_replacements += 1
        rows_by_stamp[stamp] = OrientationRow(
            raw=sample,
            quaternion_norm=angles.quaternion_norm,
            roll_rad=angles.roll_rad,
            pitch_rad=angles.pitch_rad,
        )

    rows = tuple(rows_by_stamp[stamp] for stamp in sorted(rows_by_stamp))
    if not rows:
        raise AnalyzerError("no valid IMU orientation samples")

    return AnalysisResult(
        rows=rows,
        stats=AnalysisStats(
            total_messages=total_messages,
            duplicate_replacements=duplicate_replacements,
            timestamp_reversals=timestamp_reversals,
            zero_stamp_count=zero_stamp_count,
            invalid_quaternion_count=invalid_quaternion_count,
            normalization_warning_count=normalization_warning_count,
            frame_ids=tuple(sorted(frame_ids)),
        ),
    )


def _csv_row(row, first_stamp_ns):
    """Return one stable CSV mapping without losing nanosecond precision."""
    raw = row.raw
    return {
        "header_stamp_ns": raw.header_stamp_ns,
        "elapsed_s": repr((raw.header_stamp_ns - first_stamp_ns) / 1e9),
        "bag_record_stamp_ns": raw.bag_record_stamp_ns,
        "seq": raw.seq,
        "frame_id": raw.frame_id,
        "qx": repr(float(raw.qx)),
        "qy": repr(float(raw.qy)),
        "qz": repr(float(raw.qz)),
        "qw": repr(float(raw.qw)),
        "quaternion_norm": repr(row.quaternion_norm),
        "roll_rad": repr(row.roll_rad),
        "pitch_rad": repr(row.pitch_rad),
        "roll_deg": repr(math.degrees(row.roll_rad)),
        "pitch_deg": repr(math.degrees(row.pitch_rad)),
    }


def write_csv_atomic(rows, output_path):
    """Atomically replace a CSV only after every orientation row is written."""
    rows = tuple(rows)
    if not rows:
        raise AnalyzerError("cannot write an empty orientation CSV")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=".{}-".format(destination.name),
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            first_stamp_ns = rows[0].raw.header_stamp_ns
            for row in rows:
                writer.writerow(_csv_row(row, first_stamp_ns))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(destination))
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def analyze_bag(bag_path, imu_topic="/imu", bag_factory=None):
    """Read one IMU topic and return consolidated orientation samples."""
    bag_path = Path(bag_path)
    if bag_factory is None:
        if not bag_path.is_file():
            raise AnalyzerError("bag file does not exist: {}".format(bag_path))
        try:
            import rosbag
        except ImportError as error:
            raise AnalyzerError(
                "ROS Noetic rosbag is not available in this Python environment"
            ) from error
        bag_factory = rosbag.Bag

    samples = []
    try:
        with bag_factory(str(bag_path), "r") as bag:
            for _topic, message, record_time in bag.read_messages(
                topics=[imu_topic]
            ):
                message_type = getattr(message, "_type", None)
                if message_type != "sensor_msgs/Imu":
                    raise AnalyzerError(
                        "topic {} must contain sensor_msgs/Imu, got {}".format(
                            imu_topic,
                            message_type or type(message).__name__,
                        )
                    )
                try:
                    header = message.header
                    orientation = message.orientation
                    samples.append(
                        RawOrientationSample(
                            header_stamp_ns=int(header.stamp.to_nsec()),
                            bag_record_stamp_ns=int(record_time.to_nsec()),
                            seq=int(header.seq),
                            frame_id=str(header.frame_id),
                            qx=float(orientation.x),
                            qy=float(orientation.y),
                            qz=float(orientation.z),
                            qw=float(orientation.w),
                        )
                    )
                except (AttributeError, TypeError, ValueError) as error:
                    raise AnalyzerError(
                        "topic {} message does not satisfy sensor_msgs/Imu contract"
                        .format(imu_topic)
                    ) from error
    except AnalyzerError:
        raise
    except Exception as error:
        raise AnalyzerError(
            "failed to read bag {}: {}".format(bag_path, error)
        ) from error

    if not samples:
        raise AnalyzerError(
            "topic {} was not found in {}".format(imu_topic, bag_path)
        )
    return consolidate_samples(samples)


def format_summary(result, bag_path, imu_topic, output_path):
    """Format deterministic counts and angle ranges for one completed run."""
    rows = result.rows
    stats = result.stats
    roll_degrees = [math.degrees(row.roll_rad) for row in rows]
    pitch_degrees = [math.degrees(row.pitch_rad) for row in rows]
    duration_seconds = (
        rows[-1].raw.header_stamp_ns - rows[0].raw.header_stamp_ns
    ) / 1e9

    lines = [
        "bag={}".format(Path(bag_path)),
        "imu_topic={}".format(imu_topic),
        "output_csv={}".format(Path(output_path)),
        "total_messages={}".format(stats.total_messages),
        "unique_rows={}".format(len(rows)),
        "duplicate_replacements={}".format(stats.duplicate_replacements),
        "timestamp_reversals={}".format(stats.timestamp_reversals),
        "zero_stamp_count={}".format(stats.zero_stamp_count),
        "invalid_quaternion_count={}".format(
            stats.invalid_quaternion_count
        ),
        "normalization_warning_count={}".format(
            stats.normalization_warning_count
        ),
        "frame_ids={}".format(",".join(stats.frame_ids)),
        "duration_s={:.6f}".format(duration_seconds),
        "roll_deg min={:.6f} median={:.6f} max={:.6f}".format(
            min(roll_degrees),
            statistics.median(roll_degrees),
            max(roll_degrees),
        ),
        "pitch_deg min={:.6f} median={:.6f} max={:.6f}".format(
            min(pitch_degrees),
            statistics.median(pitch_degrees),
            max(pitch_degrees),
        ),
    ]
    return "\n".join(lines)


def _argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract roll and pitch from sensor_msgs/Imu orientation in a "
            "ROS1 bag."
        )
    )
    parser.add_argument("--bag", required=True, help="input ROS1 bag path")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="destination CSV path",
    )
    parser.add_argument(
        "--imu-topic",
        default="/imu",
        help="sensor_msgs/Imu topic (default: /imu)",
    )
    return parser


def main(argv=None, bag_factory=None):
    """Run the offline analyzer and return a process exit code."""
    arguments = _argument_parser().parse_args(argv)
    bag_path = Path(arguments.bag)
    output_path = Path(arguments.output_csv)
    try:
        result = analyze_bag(
            bag_path,
            arguments.imu_topic,
            bag_factory=bag_factory,
        )
        write_csv_atomic(result.rows, output_path)
    except (AnalyzerError, OSError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    print(
        format_summary(
            result,
            bag_path,
            arguments.imu_topic,
            output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
