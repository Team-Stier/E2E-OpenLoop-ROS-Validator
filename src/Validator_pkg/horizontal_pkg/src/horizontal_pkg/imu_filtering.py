"""Filter offline IMU roll and pitch without changing source timestamps."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


class FilteringError(RuntimeError):
    """Raised when filtering input cannot satisfy the offline contract."""


@dataclass(frozen=True)
class FilterResult:
    """Filtered angles sampled at the original measurement timestamps."""

    roll_deg_filtered: np.ndarray
    pitch_deg_filtered: np.ndarray
    sample_rate_hz: float
    cutoff_hz: float


@dataclass(frozen=True)
class CsvFilterResult:
    """Summary of one successful CSV and plot export."""

    sample_count: int
    filter_result: FilterResult
    raw_roll_std_deg: float
    filtered_roll_std_deg: float
    raw_pitch_std_deg: float
    filtered_pitch_std_deg: float


REQUIRED_FIELDS = (
    "header_stamp_ns",
    "elapsed_s",
    "roll_deg",
    "pitch_deg",
)
FILTERED_FIELDS = (
    "roll_deg_filtered",
    "pitch_deg_filtered",
)


def _finite_array(values, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise FilteringError("{} must contain finite numbers".format(name)) from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise FilteringError("{} must contain finite numbers".format(name))
    return array


def _relative_seconds(header_stamp_ns):
    stamps = list(header_stamp_ns)
    if not stamps:
        raise FilteringError("input is too short for zero-phase filtering")
    try:
        numeric_stamps = [float(value) for value in stamps]
    except (TypeError, ValueError) as error:
        raise FilteringError(
            "header_stamp_ns must contain finite numbers"
        ) from error
    if not all(math.isfinite(value) for value in numeric_stamps):
        raise FilteringError("header_stamp_ns must contain finite numbers")
    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise FilteringError("header_stamp_ns must be strictly increasing")
    first = stamps[0]
    try:
        return np.asarray(
            [(value - first) * 1e-9 for value in stamps],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise FilteringError(
            "header_stamp_ns must contain finite numbers"
        ) from error


def filter_orientation(
    header_stamp_ns,
    roll_deg,
    pitch_deg,
    cutoff_hz=4.0,
):
    """Low-pass roll and pitch and return values at their original stamps."""
    elapsed = _relative_seconds(header_stamp_ns)
    roll = _finite_array(roll_deg, "roll_deg")
    pitch = _finite_array(pitch_deg, "pitch_deg")
    if not (len(elapsed) == len(roll) == len(pitch)):
        raise FilteringError("timestamp and angle lengths must match")
    if len(elapsed) < 16:
        raise FilteringError("input is too short for zero-phase filtering")

    try:
        cutoff = float(cutoff_hz)
    except (TypeError, ValueError) as error:
        raise FilteringError("cutoff_hz must be a positive finite number") from error
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise FilteringError("cutoff_hz must be a positive finite number")

    median_interval = float(np.median(np.diff(elapsed)))
    if not math.isfinite(median_interval) or median_interval <= 0.0:
        raise FilteringError("timestamp interval must be positive and finite")
    approximate_rate = 1.0 / median_interval
    if cutoff >= approximate_rate / 2.0:
        raise FilteringError("cutoff_hz must be below the derived Nyquist frequency")

    uniform_count = int(round(elapsed[-1] / median_interval)) + 1
    if uniform_count < 16:
        raise FilteringError("input is too short for zero-phase filtering")
    uniform_time = np.linspace(0.0, elapsed[-1], uniform_count)
    sample_rate = (uniform_count - 1) / elapsed[-1]
    if cutoff >= sample_rate / 2.0:
        raise FilteringError("cutoff_hz must be below the derived Nyquist frequency")

    uniform_roll = np.interp(uniform_time, elapsed, roll)
    uniform_pitch = np.interp(uniform_time, elapsed, pitch)
    sos = butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    try:
        filtered_roll = sosfiltfilt(sos, uniform_roll)
        filtered_pitch = sosfiltfilt(sos, uniform_pitch)
    except ValueError as error:
        raise FilteringError("input is too short for zero-phase filtering") from error

    return FilterResult(
        roll_deg_filtered=np.interp(elapsed, uniform_time, filtered_roll),
        pitch_deg_filtered=np.interp(elapsed, uniform_time, filtered_pitch),
        sample_rate_hz=sample_rate,
        cutoff_hz=cutoff,
    )


def _distinct_paths(input_csv, output_csv, output_png):
    paths = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in (input_csv, output_csv, output_png)
    )
    if len(set(paths)) != len(paths):
        raise FilteringError("input CSV, output CSV, and output PNG must differ")
    return paths


def _load_csv(input_path):
    try:
        with input_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = [field for field in REQUIRED_FIELDS if field not in fieldnames]
            if missing:
                raise FilteringError(
                    "input CSV is missing required columns: {}".format(
                        ", ".join(missing)
                    )
                )
            existing = [field for field in FILTERED_FIELDS if field in fieldnames]
            if existing:
                raise FilteringError(
                    "input CSV already contains filtered columns: {}".format(
                        ", ".join(existing)
                    )
                )
            rows = list(reader)
    except FilteringError:
        raise
    except (OSError, csv.Error) as error:
        raise FilteringError(
            "failed to read input CSV {}: {}".format(input_path, error)
        ) from error

    stamps = []
    elapsed = []
    roll = []
    pitch = []
    for line_number, row in enumerate(rows, start=2):
        try:
            stamps.append(int(row["header_stamp_ns"]))
            elapsed.append(float(row["elapsed_s"]))
            roll.append(float(row["roll_deg"]))
            pitch.append(float(row["pitch_deg"]))
        except (TypeError, ValueError) as error:
            raise FilteringError(
                "input CSV line {} contains a non-numeric required value".format(
                    line_number
                )
            ) from error
    if not np.all(np.isfinite(np.asarray(elapsed, dtype=float))):
        raise FilteringError("elapsed_s must contain finite numbers")
    return fieldnames, rows, np.asarray(elapsed, dtype=float), stamps, roll, pitch


def _temporary_name(destination, suffix):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".{}-".format(destination.name),
        suffix=suffix,
        dir=str(destination.parent),
        delete=False,
    )
    name = Path(handle.name)
    handle.close()
    return name


def _atomic_write_csv(destination, fieldnames, rows, result):
    temporary = _temporary_name(destination, ".csv")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames + FILTERED_FIELDS)
            writer.writeheader()
            for index, source in enumerate(rows):
                output = dict(source)
                output["roll_deg_filtered"] = repr(
                    float(result.roll_deg_filtered[index])
                )
                output["pitch_deg_filtered"] = repr(
                    float(result.pitch_deg_filtered[index])
                )
                writer.writerow(output)
        os.replace(str(temporary), str(destination))
    except (OSError, csv.Error) as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise FilteringError(
            "failed to write output CSV {}: {}".format(destination, error)
        ) from error


def _atomic_write_plot(destination, elapsed, roll, pitch, result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    temporary = _temporary_name(destination, ".png")
    figure = None
    try:
        figure, axes = plt.subplots(2, 1, figsize=(18, 10.5), sharex=True)
        axes[0].plot(elapsed, roll, color="#8a8a8a", linewidth=0.8, label="Raw roll")
        axes[0].plot(
            elapsed,
            result.roll_deg_filtered,
            color="#1976d2",
            linewidth=1.1,
            label="Filtered roll",
        )
        axes[0].set_ylabel("Roll [deg]")
        axes[1].plot(
            elapsed,
            pitch,
            color="#8a8a8a",
            linewidth=0.8,
            label="Raw pitch",
        )
        axes[1].plot(
            elapsed,
            result.pitch_deg_filtered,
            color="#ef6c00",
            linewidth=1.1,
            label="Filtered pitch",
        )
        axes[1].set_ylabel("Pitch [deg]")
        axes[1].set_xlabel("Elapsed time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")
        figure.suptitle(
            "IMU roll/pitch: 2nd-order Butterworth, "
            "cutoff {:.1f} Hz, zero-phase".format(result.cutoff_hz)
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(str(temporary), format="png", dpi=100)
        os.replace(str(temporary), str(destination))
    except (OSError, ValueError) as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise FilteringError(
            "failed to write output PNG {}: {}".format(destination, error)
        ) from error
    finally:
        if figure is not None:
            plt.close(figure)


def filter_csv(input_csv, output_csv, output_png, cutoff_hz=4.0):
    """Filter one analyzer CSV and atomically export CSV and comparison PNG."""
    input_path, output_path, png_path = _distinct_paths(
        input_csv,
        output_csv,
        output_png,
    )
    fieldnames, rows, elapsed, stamps, roll, pitch = _load_csv(input_path)
    result = filter_orientation(stamps, roll, pitch, cutoff_hz)
    _atomic_write_csv(output_path, fieldnames, rows, result)
    _atomic_write_plot(png_path, elapsed, roll, pitch, result)

    roll_array = np.asarray(roll, dtype=float)
    pitch_array = np.asarray(pitch, dtype=float)
    return CsvFilterResult(
        sample_count=len(rows),
        filter_result=result,
        raw_roll_std_deg=float(np.std(roll_array)),
        filtered_roll_std_deg=float(np.std(result.roll_deg_filtered)),
        raw_pitch_std_deg=float(np.std(pitch_array)),
        filtered_pitch_std_deg=float(np.std(result.pitch_deg_filtered)),
    )


def build_argument_parser():
    """Build the offline filtering command-line parser."""
    parser = argparse.ArgumentParser(
        description="Filter analyzer roll/pitch CSV and render a comparison PNG."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--cutoff-hz", type=float, default=4.0)
    return parser


def main(argv=None):
    """Run the offline filter CLI and return a process-style exit code."""
    arguments = build_argument_parser().parse_args(argv)
    try:
        result = filter_csv(
            arguments.input_csv,
            arguments.output_csv,
            arguments.output_png,
            arguments.cutoff_hz,
        )
    except FilteringError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    print(
        "samples={} sample_rate_hz={:.6f} cutoff_hz={:.6f}".format(
            result.sample_count,
            result.filter_result.sample_rate_hz,
            result.filter_result.cutoff_hz,
        )
    )
    print(
        "roll_std_deg raw={:.9f} filtered={:.9f}".format(
            result.raw_roll_std_deg,
            result.filtered_roll_std_deg,
        )
    )
    print(
        "pitch_std_deg raw={:.9f} filtered={:.9f}".format(
            result.raw_pitch_std_deg,
            result.filtered_pitch_std_deg,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
