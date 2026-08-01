"""Filter offline IMU roll and pitch without changing source timestamps."""

import math
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
