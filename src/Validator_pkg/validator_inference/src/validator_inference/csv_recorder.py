"""
inference metric을 평가별 CSV로 내구성 있게 기록하는 순수 Python writer.
입력: InferenceMetric과 record directory.
출력(pub): 없음. records/openloop CSV 파일.
구독(sub): 없음.
"""

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
    """source, devel, install module path에서 workspace root를 찾는다."""
    file_path = Path(module_file or __file__).resolve()
    for candidate in (file_path.parent, *file_path.parents):
        if (candidate / "src/Validator_pkg/config/profile.json").is_file():
            return candidate
        if candidate.name in {"devel", "install"}:
            return candidate.parent
    raise RuntimeError("cannot resolve project root from {}".format(file_path))


def default_record_dir(module_file: Optional[str | Path] = None) -> Path:
    """평가 CSV의 기본 저장 위치를 계산한다.

    입력: 선택적으로 source/devel/install 안의 module 파일 경로.
    반환: workspace ``records/openloop`` 디렉터리 경로.
    """
    return resolve_project_root(module_file) / "records" / "openloop"


def _seoul_time(value: datetime) -> datetime:
    """datetime을 KST timezone-aware 값으로 표준화한다.

    입력: naive 또는 timezone-aware datetime.
    반환: ``Asia/Seoul`` 기준 datetime.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=SEOUL)
    return value.astimezone(SEOUL)


def _format_stamp(stamp: Any) -> str:
    """ROS time 호환 stamp를 정렬 가능한 초.나노초 문자열로 만든다.

    입력: ``secs/nsecs`` 또는 ``sec/nanosec`` 속성이 있는 stamp.
    반환: 9자리 나노초를 갖는 문자열.
    """
    seconds = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nanoseconds = getattr(stamp, "nsecs", getattr(stamp, "nanosec", None))
    if seconds is None or nanoseconds is None:
        raise ValueError("metric header stamp requires secs and nsecs")
    return "{}.{:09d}".format(int(seconds), int(nanoseconds))


def _json_list(value: Any) -> str:
    """반복 가능한 metric 필드를 CSV 셀용 압축 JSON 배열로 변환한다.

    입력: input key나 age처럼 순서가 있는 값.
    반환: UTF-8 보존 JSON 배열 문자열.
    """
    return json.dumps(list(value), separators=(",", ":"), ensure_ascii=False)


class CsvRunWriter:
    """startup CSV 하나를 만들고 metric마다 정확히 한 행을 추가한다."""

    def __init__(
        self,
        record_dir: str | Path,
        model_id: str,
        wall_clock: Callable[[], datetime] = datetime.now,
        checkpoint_manifest: str = "",
    ) -> None:
        """평가 시작 시각을 이름에 넣은 단일 CSV 파일을 독점 생성한다.

        입력: 기록 directory, model id, wall clock, checkpoint manifest.
        부작용: directory를 만들고 CSV header를 즉시 flush한다.
        """
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
        """열린 CSV file handle을 닫아 마지막 행까지 디스크에 반영한다.

        입력: 없음.
        부작용: 이미 닫히지 않은 file handle을 close한다.
        """
        if not self._file.closed:
            self._file.close()

    def append_metric(self, metric: Any) -> None:
        """한 inference outcome을 검증한 뒤 CSV의 정확히 한 행으로 기록한다.

        입력: ``validator_msgs/InferenceMetric`` 호환 객체.
        부작용: outcome·age·history·호환성 정보를 CSV에 쓰고 flush한다.
        """
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
