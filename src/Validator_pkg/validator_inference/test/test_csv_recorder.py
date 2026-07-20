import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from validator_inference.csv_recorder import (
    CsvRunWriter,
    default_record_dir,
    resolve_project_root,
)


def _wall_clock(value="2026-07-21T10:11:12+09:00"):
    instant = datetime.fromisoformat(value)

    def clock():
        return instant

    return clock


def _metric(
    outcome="inference_completed",
    elapsed_ms=12.5,
    input_keys=("lidar_raw", "driving_feature"),
    input_age_ms=(45.0, 0.0),
):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(secs=17, nsecs=42)),
        run_id="run-20260721",
        tick_index=3,
        model_id="tfv6",
        outcome=outcome,
        elapsed_ms=elapsed_ms,
        input_keys=list(input_keys),
        input_age_ms=list(input_age_ms),
        missing_input_keys=["camera_center"],
        compatibility_flags=["camera_rig_mismatch"],
        history_count=10,
        history_span_ms=450.0,
        rng_seed=99,
        profile_sha256="profile-hash",
        calibration_profile_sha256="calibration-hash",
    )


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_filename_uses_seoul_startup_wall_time_and_model_id(tmp_path):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=_wall_clock())

    assert writer.path.name == "inference_20260721T101112+0900_tfv6.csv"
    assert writer.path.is_file()
    assert writer.run_started_at == "2026-07-21T10:11:12+09:00"
    writer.close()


def test_completed_metric_writes_one_flushed_row_with_all_dynamic_json_data(tmp_path):
    writer = CsvRunWriter(
        tmp_path,
        "tfv6",
        wall_clock=_wall_clock("2026-07-21T01:11:12+00:00"),
        checkpoint_manifest="models/tfv6/asset-manifest.json",
    )

    writer.append_metric(_metric())
    rows = _read_rows(writer.path)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-20260721"
    assert row["run_started_at"] == "2026-07-21T10:11:12+09:00"
    assert row["model_id"] == "tfv6"
    assert row["profile_sha256"] == "profile-hash"
    assert row["checkpoint_manifest"] == "models/tfv6/asset-manifest.json"
    assert row["tick_index"] == "3"
    assert row["tick_wall_time"] == "2026-07-21T10:11:12+09:00"
    assert row["reference_stamp"] == "17.000000042"
    assert row["outcome"] == "inference_completed"
    assert float(row["elapsed_ms"]) == pytest.approx(12.5)
    assert json.loads(row["input_keys"]) == ["lidar_raw", "driving_feature"]
    assert json.loads(row["input_age_ms"]) == [45.0, 0.0]
    assert json.loads(row["missing_input_keys"]) == ["camera_center"]
    assert json.loads(row["compatibility_flags"]) == ["camera_rig_mismatch"]
    assert row["history_count"] == "10"
    assert float(row["history_span_ms"]) == pytest.approx(450.0)
    assert row["rng_seed"] == "99"
    assert float(row["age_lidar_raw_ms"]) == pytest.approx(45.0)
    assert row["age_driving_feature_ms"] == "0.0"
    assert row["calibration_profile_sha256"] == "calibration-hash"
    writer.close()


def test_busy_tick_creates_exactly_one_row_and_preserves_missing_dynamic_age(tmp_path):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=_wall_clock())

    writer.append_metric(
        _metric(
            outcome="busy_tick_skipped",
            elapsed_ms=-1.0,
            input_keys=("lidar_raw",),
            input_age_ms=(123.0,),
        )
    )
    rows = _read_rows(writer.path)

    assert len(rows) == 1
    assert rows[0]["outcome"] == "busy_tick_skipped"
    assert rows[0]["elapsed_ms"] == "-1.0"
    assert rows[0]["age_camera_main_ms"] == ""
    assert json.loads(rows[0]["input_age_ms"]) == [123.0]
    writer.close()


@pytest.mark.parametrize(
    "outcome,elapsed_ms",
    [
        ("inference_completed", -0.01),
        ("not_ready", 0.0),
        ("busy_tick_skipped", -0.5),
        ("unknown", -1.0),
    ],
)
def test_invalid_outcome_elapsed_contract_is_rejected(tmp_path, outcome, elapsed_ms):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=_wall_clock())

    with pytest.raises(ValueError):
        writer.append_metric(_metric(outcome=outcome, elapsed_ms=elapsed_ms))

    assert _read_rows(writer.path) == []
    writer.close()


def test_input_key_and_age_lengths_must_match(tmp_path):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=_wall_clock())

    with pytest.raises(ValueError, match="input_keys and input_age_ms"):
        writer.append_metric(_metric(input_keys=("lidar_raw",), input_age_ms=()))

    writer.close()


@pytest.mark.parametrize("layout", ("source", "devel", "install"))
def test_default_record_dir_resolves_workspace_root_for_source_devel_and_install(
    tmp_path, layout
):
    root = tmp_path / "workspace"
    (root / "src/Validator_pkg/config").mkdir(parents=True)
    (root / "src/Validator_pkg/config/profile.json").write_text("{}")
    if layout == "source":
        module_file = root / "src/Validator_pkg/validator_inference/src/validator_inference/csv_recorder.py"
    else:
        module_file = root / layout / "lib/python3/dist-packages/validator_inference/csv_recorder.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# test location")

    assert resolve_project_root(module_file) == root
    assert default_record_dir(module_file) == root / "records/openloop"


def test_recorder_node_contains_only_inference_metric_subscription():
    script = Path(__file__).parents[1] / "scripts/recorder_node.py"
    content = script.read_text(encoding="utf-8")
    retired_timing_topic = "/inference" + "_time"

    assert '"/inference_metric"' in content
    assert retired_timing_topic not in content
    assert content.count("rospy.Subscriber(") == 1


def test_cmake_installs_recorder_node():
    cmake = (Path(__file__).parents[1] / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "catkin_install_python(PROGRAMS scripts/recorder_node.py" in cmake
