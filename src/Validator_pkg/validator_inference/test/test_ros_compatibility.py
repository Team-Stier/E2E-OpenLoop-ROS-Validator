"""모델 interpreter와 ROS Noetic 사이의 경로·통신 계약을 검사한다."""

import json
import os
import signal
import shutil
import subprocess
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message="Unknown pytest.mark.ros_compatibility",
    category=pytest.PytestUnknownMarkWarning,
)

from validator_inference.ros_compatibility import (
    MESSAGE_TYPES,
    RosCompatibilityError,
    RosCompatibilityDependencies,
    RosSmokeReport,
    RuntimePathPolicy,
    _build_message_fixtures,
    _extract_message_fields,
    _policy_from_environment,
    _validate_peer_report,
    main,
    run_ros_compatibility_smoke,
    validate_loaded_module_origins,
)


def _path_policy() -> RuntimePathPolicy:
    """가짜 target runtime의 명시적 허용 경로를 만든다."""
    return RuntimePathPolicy(
        target_python=Path("/workspace/.venvs/interfuser/builds/one/bin/python"),
        target_environment=Path("/workspace/.venvs/interfuser/builds/one"),
        ros_overlay=Path(
            "/workspace/.venvs/interfuser/builds/one/.ros-overlay"
        ),
        catkin_python_path=Path("/workspace/devel/lib/python3/dist-packages"),
        target_stdlib_paths=(Path("/uv/python/cpython-3.8.20/lib/python3.8"),),
        project_source_paths=(
            Path("/workspace/src/Validator_pkg/validator_inference/src"),
        ),
    )


def test_validate_loaded_module_origins_accepts_only_declared_runtime_roots():
    """overlay, active Catkin, target venv와 해당 interpreter stdlib만 허용한다."""
    policy = _path_policy()
    validate_loaded_module_origins(
        {
            "sys": "built-in",
            "importlib._bootstrap": "frozen",
            "rospy": policy.ros_overlay / "rospy" / "__init__.py",
            "sensor_msgs.msg._Image": (
                policy.ros_overlay / "sensor_msgs" / "msg" / "_Image.py"
            ),
            "validator_msgs.msg._DrivingFeature": (
                policy.catkin_python_path
                / "validator_msgs"
                / "msg"
                / "_DrivingFeature.py"
            ),
            "validator_inference": (
                policy.catkin_python_path / "validator_inference" / "__init__.py"
            ),
            "validator_inference.models.select": (
                policy.project_source_paths[0]
                / "validator_inference"
                / "models"
                / "select.py"
            ),
            "numpy._core._multiarray_umath": (
                policy.target_environment
                / "lib"
                / "python3.8"
                / "site-packages"
                / "numpy"
                / "_core"
                / "_multiarray_umath.cpython-38-x86_64-linux-gnu.so"
            ),
            "_ssl": (
                policy.target_stdlib_paths[0]
                / "lib-dynload"
                / "_ssl.cpython-38-x86_64-linux-gnu.so"
            ),
        },
        policy,
    )


@pytest.mark.parametrize(
    ("module_name", "origin", "reason"),
    (
        (
            "tf2_py._tf2",
            "/opt/ros/noetic/lib/python3/dist-packages/tf2_py/_tf2.so",
            "extension",
        ),
        (
            "rospy",
            "/opt/ros/noetic/lib/python3/dist-packages/rospy/__init__.py",
            "ROS package",
        ),
        (
            "validator_msgs.msg._DrivingFeature",
            "/tmp/other-catkin/validator_msgs/msg/_DrivingFeature.py",
            "Catkin",
        ),
        (
            "numpy._core._multiarray_umath",
            "/tmp/foreign/.venv/lib/python3.8/site-packages/numpy/_core/core.so",
            "extension",
        ),
        (
            "yaml",
            "/tmp/foreign/.venv/lib/python3.8/site-packages/yaml/__init__.py",
            "allowed runtime root",
        ),
    ),
)
def test_validate_loaded_module_origins_rejects_cross_runtime_imports(
    module_name, origin, reason
):
    """ROS·message·native·일반 모듈이 foreign 경로에서 오면 거부한다."""
    with pytest.raises(RosCompatibilityError, match=reason):
        validate_loaded_module_origins({module_name: origin}, _path_policy())


def test_validate_loaded_module_origins_rejects_unbounded_target_policy():
    """filesystem root를 target environment로 지정해 allowlist를 무력화하지 못한다."""
    policy = replace(
        _path_policy(),
        target_python=Path("/bin/python"),
        target_environment=Path("/"),
        ros_overlay=Path("/.ros-overlay"),
    )
    with pytest.raises(RosCompatibilityError, match="target environment"):
        validate_loaded_module_origins({"yaml": "/tmp/foreign/yaml.py"}, policy)


def test_validate_loaded_module_origins_never_accepts_opt_ros_as_stdlib():
    """policy 입력이 잘못돼도 /opt/ros native extension을 stdlib로 신뢰하지 않는다."""
    policy = replace(
        _path_policy(),
        target_stdlib_paths=(
            Path("/opt/ros/noetic/lib/python3/dist-packages"),
        ),
    )
    with pytest.raises(RosCompatibilityError, match="stdlib"):
        validate_loaded_module_origins(
            {
                "tf2_py._tf2": (
                    "/opt/ros/noetic/lib/python3/dist-packages/"
                    "tf2_py/_tf2.cpython-38-x86_64-linux-gnu.so"
                )
            },
            policy,
        )


def test_validate_loaded_module_origins_rejects_native_ros_overlay_entries():
    """pure-Python ROS overlay 안의 native binary도 target site package로 보지 않는다."""
    policy = _path_policy()
    with pytest.raises(RosCompatibilityError, match="ROS overlay"):
        validate_loaded_module_origins(
            {
                "rospy._native": (
                    policy.ros_overlay
                    / "rospy"
                    / "_native.cpython-38-x86_64-linux-gnu.so"
                )
            },
            policy,
        )


def _received_fields(sender):
    """peer JSON이 기록해야 하는 일곱 message fixture field를 만든다."""
    sequence = 101 if sender == "system" else 202
    return {
        "sensor_msgs/Image": {
            "frame_id": f"{sender}:image",
            "height": sequence,
            "width": sequence + 1,
        },
        "sensor_msgs/PointCloud2": {
            "frame_id": f"{sender}:pointcloud",
            "width": sequence,
            "point_step": 4,
        },
        "validator_msgs/DrivingFeature": {
            "model_id": sender,
            "source": f"ros-compat-{sender}",
            "status_feature": [float(sequence)],
        },
        "tf2_msgs/TFMessage": {
            "child_frame_id": f"{sender}:tf",
            "translation_x": float(sequence),
        },
        "nav_msgs/Path": {
            "frame_id": f"{sender}:path",
            "pose_count": 1,
            "first_x": float(sequence),
        },
        "validator_msgs/InferenceMetric": {
            "run_id": f"{sender}:metric",
            "tick_index": sequence,
            "model_id": sender,
            "outcome": "compatibility",
        },
        "rosgraph_msgs/Clock": {
            "secs": sequence,
            "nsecs": sequence,
        },
    }


def _peer_report(role, policy):
    """오케스트레이터 검증용 정상 peer JSON payload를 만든다."""
    if role == "target":
        executable = policy.target_python
        pythonpath = [policy.ros_overlay, policy.catkin_python_path]
        imports = [
            "validator_inference.engine",
            "validator_inference.models.select",
        ]
        origins = {
            "rospy": str(policy.ros_overlay / "rospy" / "__init__.py"),
            "validator_msgs.msg._DrivingFeature": str(
                policy.catkin_python_path
                / "validator_msgs"
                / "msg"
                / "_DrivingFeature.py"
            ),
            "validator_inference.engine": str(
                policy.project_source_paths[0]
                / "validator_inference"
                / "engine.py"
            ),
            "validator_inference.models.select": str(
                policy.project_source_paths[0]
                / "validator_inference"
                / "models"
                / "select.py"
            ),
            "_ssl": str(
                policy.target_stdlib_paths[0]
                / "lib-dynload"
                / "_ssl.cpython-38-x86_64-linux-gnu.so"
            ),
        }
    else:
        executable = Path("/usr/bin/python3")
        pythonpath = []
        imports = []
        origins = {}
    sender = "target" if role == "system" else "system"
    return {
        "schema": 1,
        "role": role,
        "python_executable": str(executable),
        "pythonpath": [str(path) for path in pythonpath],
        "pythonhome": None,
        "use_sim_time": True,
        "clock_before_ns": 1_000_000_000,
        "clock_after_ns": 2_000_000_000,
        "sent_types": list(MESSAGE_TYPES),
        "received": _received_fields(sender),
        "inference_imports": imports,
        "module_origins": origins,
    }


class _FakeProcess:
    """poll, wait와 process-group 종료를 기록하는 Popen 대역이다."""

    def __init__(self, pid, returncode=None, stubborn=False):
        self.pid = pid
        self.returncode = returncode
        self.stubborn = stubborn

    def poll(self):
        """현재 가짜 종료 코드를 반환한다."""
        return self.returncode

    def wait(self, timeout=None):
        """종료되지 않은 process는 실제 Popen처럼 timeout을 발생시킨다."""
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["fake"], timeout)
        return self.returncode


def _dependencies(processes, reports, *, fail_call=None, timeout=False):
    """process 생성·master 준비·JSON load를 완전히 메모리에서 대체한다."""
    calls = []
    signals = []
    clock = {"value": 0.0}

    def popen(argv, **kwargs):
        """미리 만든 process를 순서대로 반환하거나 지정 call에서 실패한다."""
        calls.append((list(argv), kwargs))
        if fail_call is not None and len(calls) == fail_call:
            raise OSError("forced peer startup failure")
        return processes[len(calls) - 1]

    def killpg(pid, selected_signal):
        """signal을 기록하고 SIGKILL 또는 비고집 process를 종료 처리한다."""
        signals.append((pid, selected_signal))
        process = next(item for item in processes if item.pid == pid)
        if selected_signal == signal.SIGKILL or not process.stubborn:
            process.returncode = -selected_signal

    def monotonic():
        """timeout case에서만 매 호출마다 deadline을 전진시킨다."""
        if timeout:
            clock["value"] += 1.0
        return clock["value"]

    dependencies = RosCompatibilityDependencies(
        popen=popen,
        killpg=killpg,
        monotonic=monotonic,
        sleep=lambda _seconds: None,
        master_ready=lambda _uri, _deadline: True,
        set_parameter=lambda _uri, _name, _value: None,
        load_report=lambda path: reports[path.name],
        reserve_port=lambda: 43210,
    )
    return dependencies, calls, signals


def test_run_ros_compatibility_smoke_validates_reports_and_cleans_roscore():
    """정상 peer 결과도 검증하고 마지막에는 임시 roscore를 종료한다."""
    policy = _path_policy()
    processes = [
        _FakeProcess(1001),
        _FakeProcess(1002, returncode=0),
        _FakeProcess(1003, returncode=0),
    ]
    reports = {
        "system.json": _peer_report("system", policy),
        "target.json": _peer_report("target", policy),
    }
    dependencies, calls, signals = _dependencies(processes, reports)

    result = run_ros_compatibility_smoke(
        policy,
        system_python=Path("/usr/bin/python3"),
        timeout_seconds=5.0,
        dependencies=dependencies,
    )

    assert isinstance(result, RosSmokeReport)
    assert result.target_python == policy.target_python
    assert result.target_pythonpath == (
        policy.ros_overlay,
        policy.catkin_python_path,
    )
    assert result.system_received == MESSAGE_TYPES
    assert result.target_received == MESSAGE_TYPES
    assert all(kwargs["start_new_session"] is True for _argv, kwargs in calls)
    target_environment = calls[2][1]["env"]
    assert target_environment["PYTHONPATH"] == ":".join(
        (str(policy.ros_overlay), str(policy.catkin_python_path))
    )
    assert target_environment["VIRTUAL_ENV"] == str(policy.target_environment)
    assert "PYTHONHOME" not in target_environment
    assert signals == [(1001, signal.SIGTERM)]


def test_run_ros_compatibility_smoke_cleans_started_groups_on_peer_start_failure():
    """target peer 시작이 실패해도 먼저 시작한 roscore와 system peer를 종료한다."""
    policy = _path_policy()
    processes = [_FakeProcess(2001), _FakeProcess(2002)]
    dependencies, calls, signals = _dependencies(
        processes, {}, fail_call=3
    )

    with pytest.raises(RosCompatibilityError, match="startup failure"):
        run_ros_compatibility_smoke(
            policy,
            system_python=Path("/usr/bin/python3"),
            timeout_seconds=5.0,
            dependencies=dependencies,
        )

    assert len(calls) == 3
    assert signals == [
        (2002, signal.SIGTERM),
        (2001, signal.SIGTERM),
    ]


def test_run_ros_compatibility_smoke_timeout_kills_every_stubborn_group():
    """bounded deadline을 넘기면 세 process group 모두 SIGKILL까지 정리한다."""
    policy = _path_policy()
    processes = [
        _FakeProcess(3001, stubborn=True),
        _FakeProcess(3002, stubborn=True),
        _FakeProcess(3003, stubborn=True),
    ]
    dependencies, _calls, signals = _dependencies(
        processes, {}, timeout=True
    )

    with pytest.raises(RosCompatibilityError, match="timed out"):
        run_ros_compatibility_smoke(
            policy,
            system_python=Path("/usr/bin/python3"),
            timeout_seconds=2.0,
            dependencies=dependencies,
        )

    assert signals == [
        (3003, signal.SIGTERM),
        (3003, signal.SIGKILL),
        (3002, signal.SIGTERM),
        (3002, signal.SIGKILL),
        (3001, signal.SIGTERM),
        (3001, signal.SIGKILL),
    ]


@pytest.mark.parametrize("role", ("system", "target"))
def test_real_ros_message_fixtures_round_trip_expected_fields(role):
    """일곱 실제 ROS message 객체가 role별 sentinel field를 보존한다."""
    fixtures = _build_message_fixtures(role)

    assert tuple(fixtures) == MESSAGE_TYPES
    assert {
        message_type: _extract_message_fields(message_type, message)
        for message_type, message in fixtures.items()
    } == _received_fields(role)


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    (
        (
            "python_executable",
            "/tmp/foreign/.venv/bin/python",
            "sys.executable",
        ),
        (
            "pythonpath",
            ["/opt/ros/noetic/lib/python3/dist-packages"],
            "PYTHONPATH",
        ),
        ("use_sim_time", False, "use_sim_time"),
        ("clock_after_ns", 1_000_000_000, "clock advancement"),
        ("inference_imports", [], "inference selectors"),
    ),
)
def test_target_report_rejects_runtime_contract_mismatch(field, replacement, reason):
    """target JSON의 interpreter, path, sim-time와 import 계약 불일치를 거부한다."""
    policy = _path_policy()
    payload = _peer_report("target", policy)
    payload[field] = replacement

    with pytest.raises(RosCompatibilityError, match=reason):
        _validate_peer_report(payload, "target", policy.target_python, policy)


def test_policy_from_environment_preserves_exact_target_allowlist():
    """target peer가 controller가 넘긴 경로 정책을 임의 확장 없이 복원한다."""
    expected = _path_policy()
    environment = {
        "VALIDATOR_ROS_TARGET_PYTHON": str(expected.target_python),
        "VALIDATOR_ROS_TARGET_ENVIRONMENT": str(expected.target_environment),
        "VALIDATOR_ROS_OVERLAY": str(expected.ros_overlay),
        "VALIDATOR_ROS_CATKIN_PYTHON": str(expected.catkin_python_path),
        "VALIDATOR_ROS_STDLIB_PATHS": ":".join(
            str(path) for path in expected.target_stdlib_paths
        ),
        "VALIDATOR_ROS_SOURCE_PATHS": ":".join(
            str(path) for path in expected.project_source_paths
        ),
    }

    assert _policy_from_environment(environment) == expected


def test_peer_cli_passes_exact_role_and_absolute_result(tmp_path):
    """module CLI가 role/result만 파싱해 peer 실행기에 그대로 전달한다."""
    calls = []

    def peer_runner(role, result):
        """CLI가 전달한 값을 기록하는 가짜 peer 실행기다."""
        calls.append((role, result))

    result_path = tmp_path / "peer.json"
    assert main(
        ["--role", "target", "--result", str(result_path)],
        peer_runner=peer_runner,
    ) == 0
    assert calls == [("target", result_path)]


def test_peer_cli_reports_runtime_failure_without_traceback(tmp_path, capsys):
    """peer 계약 실패를 nonzero status와 한 줄 stderr로 변환한다."""
    def peer_runner(_role, _result):
        """예상된 peer 오류를 발생시킨다."""
        raise RosCompatibilityError("forced peer mismatch")

    assert main(
        ["--role", "system", "--result", str(tmp_path / "peer.json")],
        peer_runner=peer_runner,
    ) == 1
    assert capsys.readouterr().err.strip() == "ros compatibility error: forced peer mismatch"


@pytest.mark.ros_compatibility
@pytest.mark.parametrize("python_version", ("3.8.20", "3.9.25", "3.10.20"))
def test_real_cross_version_ros_compatibility_smoke(tmp_path, python_version):
    """세 exact CPython으로 최소 overlay를 만들고 일곱 ROS type을 양방향 교환한다."""
    if os.environ.get("VALIDATOR_RUN_ROS_COMPAT") != "1":
        pytest.skip("set VALIDATOR_RUN_ROS_COMPAT=1 to create target interpreters")

    uv = shutil.which("uv") or "/home/stier/.local/bin/uv"
    if not Path(uv).is_file():
        pytest.fail("uv is required for opted-in ROS compatibility tests")
    project_root = Path(__file__).resolve().parents[4]
    target_environment = tmp_path / f"cpython-{python_version}"
    target_python = target_environment / "bin" / "python"
    subprocess.run(
        [
            uv,
            "venv",
            "--no-project",
            "--managed-python",
            "--python",
            python_version,
            str(target_environment),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(target_python),
            "--strict",
            "numpy==1.24.4",
            "catkin-pkg==1.0.0",
            "rospkg==1.6.0",
            "PyYAML==6.0.3",
            "netifaces==0.11.0",
            "python-dateutil==2.9.0.post0",
            "empy==3.3.4",
        ],
        check=True,
        timeout=180,
    )

    from validator_inference.environment_setup import build_ros_overlay

    ros_overlay = target_environment / ".ros-overlay"
    build_ros_overlay(
        Path("/opt/ros/noetic/lib/python3/dist-packages"), ros_overlay
    )
    catkin_python_path = project_root / "devel/lib/python3/dist-packages"
    source_path = (
        project_root
        / "src/Validator_pkg/validator_inference/src"
    )
    completed = subprocess.run(
        [
            str(target_python),
            "-c",
            (
                "import json,sysconfig; "
                "print(json.dumps([sysconfig.get_path('stdlib'), "
                "sysconfig.get_path('platstdlib')]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdlib_paths = tuple(
        dict.fromkeys(Path(value) for value in json.loads(completed.stdout))
    )
    policy = RuntimePathPolicy(
        target_python=target_python,
        target_environment=target_environment,
        ros_overlay=ros_overlay,
        catkin_python_path=catkin_python_path,
        target_stdlib_paths=stdlib_paths,
        project_source_paths=(source_path,),
    )

    report = run_ros_compatibility_smoke(
        policy,
        system_python=Path("/usr/bin/python3"),
        timeout_seconds=45.0,
    )

    assert report.target_python == target_python
    assert report.target_pythonpath == (ros_overlay, catkin_python_path)
    assert report.system_received == MESSAGE_TYPES
    assert report.target_received == MESSAGE_TYPES
    assert report.system_clock_advanced is True
    assert report.target_clock_advanced is True
