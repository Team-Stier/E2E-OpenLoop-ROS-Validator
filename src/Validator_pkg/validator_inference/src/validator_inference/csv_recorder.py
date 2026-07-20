"""Durable per-run CSV persistence for inference metric messages."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional


SEOUL = timezone(timedelta(hours=9))

CSV_COLUMNS = (
    "run_id",
    "run_started_at",
    "model_id",
    "profile_sha256",
    "checkpoint_manifest",
    "tick_index",
    "tick_wall_time",
    "reference_stamp",
    "outcome",
    "elapsed_ms",
    "input_keys",
    "input_age_ms",
    "missing_input_keys",
    "compatibility_flags",
    "history_count",
    "history_span_ms",
    "rng_seed",
    "age_lidar_raw_ms",
    "age_lidar_bev_ms",
    "age_camera_main_ms",
    "age_camera_left_ms",
    "age_camera_right_ms",
    "age_camera_center_ms",
    "age_driving_feature_ms",
    "age_route_target_point_ms",
    "age_route_target_point_previous_ms",
    "age_route_target_point_next_ms",
    "calibration_profile_sha256",
)

_DYNAMIC_AGE_COLUMNS = tuple(
    column for column in CSV_COLUMNS if column.startswith("age_")
)
_OUTCOME_ELAPSED = {
    "inference_completed": "nonnegative",
    "not_ready": "minus_one",
    "busy_tick_skipped": "minus_one",
}


def resolve_project_root(module_file: Optional[str | Path] = None) -> Path:
    """Find the workspace root from source, devel, or install module paths."""
    file_path = Path(module_file or __file__).resolve()
    for candidate in (file_path.parent, *file_path.parents):
        if (candidate / "src/Validator_pkg/config/profile.json").is_file():
            return candidate
        if candidate.name in {"devel", "install"}:
            return candidate.parent
    raise RuntimeError("cannot resolve project root from {}".format(file_path))


def default_record_dir(module_file: Optional[str | Path] = None) -> Path:
    return resolve_project_root(module_file) / "records" / "openloop"


def _seoul_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=SEOUL)
    return value.astimezone(SEOUL)


def _format_stamp(stamp: Any) -> str:
    seconds = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nanoseconds = getattr(stamp, "nsecs", getattr(stamp, "nanosec", None))
    if seconds is None or nanoseconds is None:
        raise ValueError("metric header stamp requires secs and nsecs")
    return "{}.{:09d}".format(int(seconds), int(nanoseconds))


def _json_list(value: Any) -> str:
    return json.dumps(list(value), separators=(",", ":"), ensure_ascii=False)


class CsvRunWriter:
    """Create one startup CSV and append exactly one row per metric."""

    def __init__(
        self,
        record_dir: str | Path,
        model_id: str,
        wall_clock: Callable[[], datetime] = datetime.now,
        checkpoint_manifest: str = "",
    ) -> None:
        self._wall_clock = wall_clock
        self._model_id = str(model_id)
        self._checkpoint_manifest = str(checkpoint_manifest)
        startup = _seoul_time(self._wall_clock())
        self.run_started_at = startup.isoformat()
        filename_time = startup.strftime("%Y%m%dT%H%M%S%z")
        directory = Path(record_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "inference_{}_{}.csv".format(
            filename_time, self._model_id
        )
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._dict_writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._dict_writer.writeheader()
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def append_metric(self, metric: Any) -> None:
        input_keys = list(metric.input_keys)
        input_age_ms = list(metric.input_age_ms)
        if len(input_keys) != len(input_age_ms):
            raise ValueError("input_keys and input_age_ms must have equal length")

        elapsed_ms = float(metric.elapsed_ms)
        if not math.isfinite(elapsed_ms):
            raise ValueError("elapsed_ms must be finite")
        outcome = str(metric.outcome)
        expected = _OUTCOME_ELAPSED.get(outcome)
        if expected is None:
            raise ValueError("unsupported inference outcome: {}".format(outcome))
        if expected == "nonnegative" and elapsed_ms < 0.0:
            raise ValueError("inference_completed elapsed_ms must be nonnegative")
        if expected == "minus_one" and elapsed_ms != -1.0:
            raise ValueError("{} elapsed_ms must be -1.0".format(outcome))

        ages = {str(key): float(age) for key, age in zip(input_keys, input_age_ms)}
        row = {
            "run_id": str(metric.run_id),
            "run_started_at": self.run_started_at,
            "model_id": str(metric.model_id),
            "profile_sha256": str(metric.profile_sha256),
            "checkpoint_manifest": self._checkpoint_manifest,
            "tick_index": int(metric.tick_index),
            "tick_wall_time": _seoul_time(self._wall_clock()).isoformat(),
            "reference_stamp": _format_stamp(metric.header.stamp),
            "outcome": outcome,
            "elapsed_ms": elapsed_ms,
            "input_keys": _json_list(input_keys),
            "input_age_ms": _json_list(input_age_ms),
            "missing_input_keys": _json_list(metric.missing_input_keys),
            "compatibility_flags": _json_list(metric.compatibility_flags),
            "history_count": int(metric.history_count),
            "history_span_ms": float(metric.history_span_ms),
            "rng_seed": int(metric.rng_seed),
            "calibration_profile_sha256": str(
                metric.calibration_profile_sha256
            ),
        }
        for column in _DYNAMIC_AGE_COLUMNS:
            input_key = column[len("age_") : -len("_ms")]
            if input_key in ages:
                row[column] = ages[input_key]

        self._dict_writer.writerow(row)
        self._file.flush()
