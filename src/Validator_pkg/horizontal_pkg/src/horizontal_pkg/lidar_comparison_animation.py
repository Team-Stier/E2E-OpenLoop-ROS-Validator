"""Render an offline raw-versus-horizontalized LiDAR comparison MP4."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_COLOR = "#FF5A36"
HORIZONTAL_COLOR = "#16E0E0"
BACKGROUND_COLOR = "#101318"
TEXT_COLOR = "#F2F4F8"
GRID_COLOR = "#596170"


class AnimationError(RuntimeError):
    """Raised when a comparison animation cannot satisfy its contract."""


@dataclass(frozen=True)
class RenderConfig:
    width_px: int = 1280
    height_px: int = 720
    dpi: int = 100
    side_x_limits: tuple = (-5.0, 30.0)
    side_z_limits: tuple = (-3.0, 3.0)
    front_y_limits: tuple = (-15.0, 15.0)
    front_z_limits: tuple = (-3.0, 3.0)
    max_points: int = 12000

    def __post_init__(self):
        if self.width_px < 1 or self.height_px < 1 or self.dpi < 1:
            raise AnimationError("render dimensions and DPI must be positive")
        if self.max_points < 1:
            raise AnimationError("max points must be positive")


@dataclass(frozen=True)
class AnimationStats:
    paired_frames: int
    rendered_frames: int
    fps: int
    duration_s: float
    output_bytes: int


@dataclass(frozen=True)
class PairingSummary:
    paired_stamps_ns: tuple
    raw_count: int
    horizontal_count: int
    raw_unmatched_count: int
    horizontal_unmatched_count: int


def _field_map(message):
    result = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if field.name in result:
                raise AnimationError("PointCloud2 XYZ fields must be unique")
            result[field.name] = field
    if set(result) != {"x", "y", "z"}:
        raise AnimationError("PointCloud2 requires x, y, and z fields")
    for field in result.values():
        if field.datatype != 7 or field.count != 1:
            raise AnimationError("PointCloud2 XYZ fields must be FLOAT32")
        if field.offset < 0 or field.offset + 4 > message.point_step:
            raise AnimationError("PointCloud2 XYZ offset is outside point_step")
    return result


def layout_signature(message):
    """Return the binary layout elements that paired clouds must share."""
    return (
        int(message.height),
        int(message.width),
        int(message.point_step),
        int(message.row_step),
        bool(message.is_bigendian),
        tuple(
            (field.name, field.offset, field.datatype, field.count)
            for field in message.fields
        ),
    )


def extract_xyz(message):
    """Copy ordered XYZ values from an organized or unorganized PointCloud2."""
    fields = _field_map(message)
    if message.height < 1 or message.width < 1:
        raise AnimationError("PointCloud2 dimensions must be positive")
    if message.row_step < message.width * message.point_step:
        raise AnimationError("PointCloud2 row_step is inconsistent")
    if len(message.data) != message.row_step * message.height:
        raise AnimationError("PointCloud2 payload length is inconsistent")
    dtype = np.dtype((">" if message.is_bigendian else "<") + "f4")
    rows = []
    for row in range(message.height):
        views = []
        for name in ("x", "y", "z"):
            views.append(
                np.ndarray(
                    (message.width,),
                    dtype=dtype,
                    buffer=message.data,
                    offset=row * message.row_step + fields[name].offset,
                    strides=(message.point_step,),
                )
            )
        rows.append(np.column_stack(views))
    return np.vstack(rows).astype(np.float32, copy=True)


def validate_pair(raw_message, horizontal_message):
    """Reject pairs that cannot be compared point-for-point."""
    if raw_message.header.stamp != horizontal_message.header.stamp:
        raise AnimationError("paired PointCloud2 timestamps differ")
    if layout_signature(raw_message) != layout_signature(horizontal_message):
        raise AnimationError("paired PointCloud2 layout differs")


def uniform_indices(total, target):
    """Return deterministic, approximately uniform indices including endpoints."""
    total = int(total)
    target = int(target)
    if total < 0 or target < 1:
        raise AnimationError("sample counts must be nonnegative and target positive")
    if total == 0:
        return ()
    if target >= total:
        return tuple(range(total))
    if target == 1:
        return (0,)
    return tuple(int(round(index * (total - 1) / float(target - 1))) for index in range(target))


def matching_point_indices(raw_xyz, horizontal_xyz, max_points):
    """Select the same finite point indices from both arrays."""
    raw = np.asarray(raw_xyz)
    horizontal = np.asarray(horizontal_xyz)
    if raw.shape != horizontal.shape or raw.ndim != 2 or raw.shape[1] != 3:
        raise AnimationError("paired XYZ arrays must have matching Nx3 shapes")
    finite = np.all(np.isfinite(raw), axis=1) & np.all(np.isfinite(horizontal), axis=1)
    candidates = np.flatnonzero(finite)
    selected = uniform_indices(len(candidates), min(int(max_points), len(candidates)))
    return candidates[np.asarray(selected, dtype=int)] if selected else np.asarray([], dtype=int)


def _style_axis(axis, xlabel, ylabel, x_limits, z_limits):
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*z_limits)
    axis.set_xlabel(xlabel, color=TEXT_COLOR)
    axis.set_ylabel(ylabel, color=TEXT_COLOR)
    axis.tick_params(colors=TEXT_COLOR)
    axis.grid(True, color=GRID_COLOR, alpha=0.28, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)


def _style_legend(legend):
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", ())
    for handle in handles:
        handle.set_alpha(1.0)


def render_frame(raw_xyz, horizontal_xyz, metadata, config=RenderConfig()):
    """Render one fixed-geometry side/front overlay as an RGB image."""
    indices = matching_point_indices(raw_xyz, horizontal_xyz, config.max_points)
    raw = np.asarray(raw_xyz)[indices]
    horizontal = np.asarray(horizontal_xyz)[indices]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(config.width_px / config.dpi, config.height_px / config.dpi),
        dpi=config.dpi,
        facecolor=BACKGROUND_COLOR,
    )
    side, front = axes
    _style_axis(side, "X forward [m]", "Z up [m]", config.side_x_limits, config.side_z_limits)
    _style_axis(front, "Y left [m]", "Z up [m]", config.front_y_limits, config.front_z_limits)
    side.set_title("Side view: pitch compensation", color=TEXT_COLOR)
    front.set_title("Front view: roll compensation", color=TEXT_COLOR)
    for axis, horizontal_index in ((side, 0), (front, 1)):
        axis.scatter(raw[:, horizontal_index], raw[:, 2], s=1.1, c=RAW_COLOR, alpha=0.48, linewidths=0, label="Raw /lidar3D")
        axis.scatter(horizontal[:, horizontal_index], horizontal[:, 2], s=1.1, c=HORIZONTAL_COLOR, alpha=0.58, linewidths=0, label="Horizontalized")
        legend = axis.legend(loc="upper right", facecolor="#202631", edgecolor=GRID_COLOR, framealpha=0.95)
        _style_legend(legend)
    figure.suptitle(str(metadata), color=TEXT_COLOR, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(figure)
    return rgb


def summarize_topic_stamps(raw_stamps, horizontal_stamps):
    """Validate timestamp uniqueness and summarize exact topic pairing."""
    raw = tuple(int(value) for value in raw_stamps)
    horizontal = tuple(int(value) for value in horizontal_stamps)
    if any(count > 1 for count in Counter(raw).values()):
        raise AnimationError("raw LiDAR topic contains duplicate header stamps")
    if any(count > 1 for count in Counter(horizontal).values()):
        raise AnimationError("horizontal LiDAR topic contains duplicate header stamps")
    raw_set = set(raw)
    horizontal_set = set(horizontal)
    paired = tuple(sorted(raw_set & horizontal_set))
    return PairingSummary(
        paired_stamps_ns=paired,
        raw_count=len(raw),
        horizontal_count=len(horizontal),
        raw_unmatched_count=len(raw_set - horizontal_set),
        horizontal_unmatched_count=len(horizontal_set - raw_set),
    )


def collect_pair_stamps(input_bag, raw_topic, horizontal_topic):
    """Collect and summarize exact header-stamp pairs from both topics."""
    try:
        import rosbag
        stamps = {raw_topic: [], horizontal_topic: []}
        with rosbag.Bag(str(input_bag), "r") as bag:
            for topic, message, _ in bag.read_messages(topics=list(stamps)):
                stamps[topic].append(int(message.header.stamp.to_nsec()))
    except Exception as error:
        raise AnimationError("failed to read input bag: {}".format(error)) from error
    if not stamps[raw_topic]:
        raise AnimationError("bag is missing raw LiDAR topic {}".format(raw_topic))
    if not stamps[horizontal_topic]:
        raise AnimationError(
            "bag is missing horizontal LiDAR topic {}".format(horizontal_topic)
        )
    summary = summarize_topic_stamps(stamps[raw_topic], stamps[horizontal_topic])
    if not summary.paired_stamps_ns:
        raise AnimationError("bag contains no timestamp-paired LiDAR frames")
    return summary


def _ffmpeg_command(output_path, config, fps):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "{}x{}".format(config.width_px, config.height_px),
        "-framerate", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]


def _encode_selected_pairs(input_bag, temporary, selected_stamps, raw_topic, horizontal_topic, fps, config):
    import rosbag

    selected = set(selected_stamps)
    pending = {}
    rendered = 0
    process = None
    with tempfile.TemporaryFile() as error_log:
        try:
            process = subprocess.Popen(
                _ffmpeg_command(temporary, config, fps),
                stdin=subprocess.PIPE,
                stderr=error_log,
            )
            with rosbag.Bag(str(input_bag), "r") as bag:
                for topic, message, _ in bag.read_messages(topics=[raw_topic, horizontal_topic]):
                    stamp = int(message.header.stamp.to_nsec())
                    if stamp not in selected:
                        continue
                    pair = pending.setdefault(stamp, {})
                    pair[topic] = message
                    if raw_topic not in pair or horizontal_topic not in pair:
                        continue
                    raw_message = pair[raw_topic]
                    horizontal_message = pair[horizontal_topic]
                    validate_pair(raw_message, horizontal_message)
                    raw_xyz = extract_xyz(raw_message)
                    horizontal_xyz = extract_xyz(horizontal_message)
                    metadata = "Frame {:04d}/{:04d}   stamp {:.3f} s   paired points {:,}".format(
                        rendered + 1, len(selected_stamps), stamp / 1e9,
                        len(matching_point_indices(raw_xyz, horizontal_xyz, config.max_points)),
                    )
                    frame = render_frame(raw_xyz, horizontal_xyz, metadata, config)
                    process.stdin.write(frame.tobytes())
                    rendered += 1
                    del pending[stamp]
            process.stdin.close()
            return_code = process.wait()
            error_log.seek(0)
            stderr = error_log.read().decode("utf-8", errors="replace")
            if return_code != 0:
                raise AnimationError("ffmpeg failed: {}".format(stderr.strip()))
        except FileNotFoundError as error:
            raise AnimationError("ffmpeg is not available") from error
        except BrokenPipeError as error:
            error_log.seek(0)
            stderr = error_log.read().decode("utf-8", errors="replace")
            raise AnimationError(
                "ffmpeg closed its input: {}".format(stderr.strip())
            ) from error
        finally:
            if process is not None and process.poll() is None:
                try:
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                except BrokenPipeError:
                    pass
                process.kill()
                process.wait()
    if rendered != len(selected_stamps):
        raise AnimationError("rendered {} of {} selected frame pairs".format(rendered, len(selected_stamps)))
    return rendered


def _validate_video(path, config, expected_frames):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames,duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as error:
        raise AnimationError("ffprobe is not available") from error
    if result.returncode != 0:
        raise AnimationError("ffprobe validation failed: {}".format(result.stderr.strip()))
    try:
        stream = json.loads(result.stdout)["streams"][0]
        valid = (
            stream["codec_name"] == "h264"
            and stream["pix_fmt"] == "yuv420p"
            and int(stream["width"]) == config.width_px
            and int(stream["height"]) == config.height_px
            and int(stream["nb_frames"]) == int(expected_frames)
            and float(stream["duration"]) > 0.0
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        raise AnimationError("ffprobe metadata does not match render contract")


def encode_animation(input_bag, output_mp4, fps=10, max_frames=600, config=RenderConfig(), raw_topic="/lidar3D", horizontal_topic="/lidar3D/horizontal", pair_stamps_ns=None):
    """Create and atomically publish a validated comparison MP4."""
    input_path = Path(input_bag).expanduser().resolve(strict=False)
    output_path = Path(output_mp4).expanduser().resolve(strict=False)
    if not pair_stamps_ns and not input_path.is_file():
        raise AnimationError("input bag does not exist: {}".format(input_path))
    if int(fps) < 1 or int(max_frames) < 1:
        raise AnimationError("FPS and max frames must be positive")
    if pair_stamps_ns is not None:
        stamps = tuple(pair_stamps_ns)
    else:
        stamps = collect_pair_stamps(
            input_path, raw_topic, horizontal_topic
        ).paired_stamps_ns
    choices = uniform_indices(len(stamps), min(len(stamps), int(max_frames)))
    selected = tuple(stamps[index] for index in choices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=".{}-".format(output_path.stem), suffix=".mp4", dir=str(output_path.parent), delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        rendered = _encode_selected_pairs(input_path, temporary, selected, raw_topic, horizontal_topic, int(fps), config)
        _validate_video(temporary, config, rendered)
        size = temporary.stat().st_size
        if size <= 0:
            raise AnimationError("encoded video is empty")
        os.replace(str(temporary), str(output_path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return AnimationStats(len(stamps), rendered, int(fps), rendered / float(fps), size)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render raw and horizontalized LiDAR overlays to MP4.")
    parser.add_argument("--input-bag", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--max-points", type=int, default=12000)
    arguments = parser.parse_args(argv)
    try:
        stats = encode_animation(
            arguments.input_bag,
            arguments.output_mp4,
            fps=arguments.fps,
            max_frames=arguments.max_frames,
            config=RenderConfig(max_points=arguments.max_points),
        )
    except AnimationError as error:
        print("error: {}".format(error))
        return 2
    print("paired={} rendered={} fps={} duration_s={:.1f} bytes={}".format(stats.paired_frames, stats.rendered_frames, stats.fps, stats.duration_s, stats.output_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
