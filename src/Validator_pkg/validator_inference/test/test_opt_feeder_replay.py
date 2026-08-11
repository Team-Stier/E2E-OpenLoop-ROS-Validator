"""MORAI receipt-clock replay helpers for OptFeeder."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


def _row(stamp, command, **overrides):
    names = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
    row = {
        "schema_version": 2,
        "domain_id": "morai-ddv2-v1-oracle-command",
        "resolved_domain_sha256": "a" * 64,
        "sensor_preset_sha256": "b" * 64,
        "source_bag_sha256": "c" * 64,
        "command_rule_sha256": "d" * 64,
        "timestamp_ns": stamp,
        "grid_receipt_ns": stamp,
        "front_receipt_ns": stamp,
        "front_record_index": 0,
        "command_index": command,
        "command_name": names[command] if 0 <= command < 4 else "INVALID",
        "provenance": "future_trajectory_oracle",
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="module")
def opt_feeder():
    pytest.importorskip("rospy")
    path = Path(__file__).resolve().parents[1] / "scripts" / "opt_feeder_node.py"
    spec = spec_from_file_location("opt_feeder_replay_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_sidecar_uses_receipt_clock(opt_feeder, tmp_path):
    path = tmp_path / "commands.jsonl"
    rows = [
        _row(100, 3),
        _row(200, 0),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    schedule = opt_feeder.load_command_sidecar(
        path,
        {
            "domain_id": "morai-ddv2-v1-oracle-command",
            "resolved_domain_sha256": "a" * 64,
        },
    )

    assert schedule == ((100, 3), (200, 0))
    assert opt_feeder.due_commands(schedule, 0, 99) == ((), 0)
    assert opt_feeder.due_commands(schedule, 0, 199) == (((100, 3),), 1)
    assert opt_feeder.due_commands(schedule, 1, 250) == (((200, 0),), 2)


@pytest.mark.parametrize(
    "rows",
    [
        [[100, 1]],
        [_row(100, 4)],
        [_row(-1, 1)],
        [_row(100, 1, schema_version=1)],
        [_row(100, 1, command_name="LEFT")],
        [_row(100, 1, provenance="manual")],
        [_row(100, 1, domain_id="another-domain")],
        [
            _row(100, 1),
            _row(100, 2),
        ],
        [
            _row(100, 1, front_record_index=0),
            _row(200, 2, timestamp_ns=100, front_record_index=1),
        ],
        [
            _row(100, 1),
            _row(200, 2, source_bag_sha256="e" * 64),
        ],
    ],
)
def test_command_sidecar_rejects_invalid_rows(opt_feeder, tmp_path, rows):
    path = tmp_path / "commands.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        opt_feeder.load_command_sidecar(
            path,
            {
                "domain_id": "morai-ddv2-v1-oracle-command",
                "resolved_domain_sha256": "a" * 64,
            },
        )


def test_command_replay_requires_atomic_mode(opt_feeder):
    opt_feeder.validate_command_replay(
        "diffusiondrive_v2",
        True,
        "new_stamp_once",
    )

    for arguments in (
        ("sparsedrive_v2", True, "new_stamp_once"),
        ("diffusiondrive_v2", False, "new_stamp_once"),
        ("diffusiondrive_v2", True, "every_tick"),
    ):
        with pytest.raises(ValueError, match="command_sidecar"):
            opt_feeder.validate_command_replay(*arguments)


def test_clock_before_ego_waits_then_uses_causal_ego(opt_feeder, monkeypatch):
    class Time:
        def __init__(self, secs=0, nsecs=0):
            self.nanoseconds = secs * 1_000_000_000 + nsecs

        def to_nsec(self):
            return self.nanoseconds

    monkeypatch.setattr(opt_feeder.rospy, "Time", Time)
    node = opt_feeder.OptFeederNode.__new__(opt_feeder.OptFeederNode)
    node._command_schedule = ((100, 3),)
    node._next_command = 0
    node._ego_history = [(90, "older-causal")]
    published = []
    node._publish = lambda ego, stamp, command, source: published.append(
        (ego, stamp.to_nsec(), command, source)
    )

    node._drain(Time(nsecs=100))
    node._ego_history.extend(((95, "latest-causal"), (110, "watermark")))
    node._drain(Time(nsecs=110))
    node._drain(Time(nsecs=110))

    assert published == [("latest-causal", 100, 3, "future_trajectory_oracle")]
    assert node._next_command == 1


def test_node_passes_profile_identity_to_sidecar_loader(opt_feeder, monkeypatch):
    class Profile:
        data = {
            "inference_policy": {"reference_stamp_policy": "new_stamp_once"},
            "export_identity": {
                "domain_id": "morai-ddv2-v1-oracle-command",
                "resolved_domain_sha256": "a" * 64,
            },
        }

    params = {
        "~model_id": "diffusiondrive_v2",
        "~profile_file": "profile.json",
        "~use_ros_time_stamp": True,
        "~command_sidecar": "commands.jsonl",
    }
    loaded = []
    monkeypatch.setattr(opt_feeder, "_default_profile_file", lambda: "default.json")
    monkeypatch.setattr(opt_feeder, "load_selected_profile", lambda *_: Profile())
    monkeypatch.setattr(
        opt_feeder.rospy,
        "get_param",
        lambda key, default=None: params.get(key, default),
    )
    monkeypatch.setattr(opt_feeder.rospy, "Publisher", lambda *_, **__: object())
    monkeypatch.setattr(opt_feeder.rospy, "Subscriber", lambda *_, **__: object())
    monkeypatch.setattr(opt_feeder, "build_fallback_targets", lambda *_: ())
    monkeypatch.setattr(
        opt_feeder,
        "load_command_sidecar",
        lambda path, identity: loaded.append((path, identity)) or ((100, 1),),
    )

    opt_feeder.OptFeederNode()

    assert loaded == [
        (
            "commands.jsonl",
            {
                "domain_id": "morai-ddv2-v1-oracle-command",
                "resolved_domain_sha256": "a" * 64,
            },
        )
    ]


def test_opt_feeder_replay_stamp_uses_ros_clock(opt_feeder, monkeypatch):
    source_stamp = object()
    replay_stamp = object()

    class FakeTime:
        @staticmethod
        def now():
            return replay_stamp

    monkeypatch.setattr(opt_feeder.rospy, "Time", FakeTime)
    assert opt_feeder.select_output_stamp(source_stamp, False) is source_stamp
    assert opt_feeder.select_output_stamp(source_stamp, True) is replay_stamp
