"""모델 interpreter와 ROS Noetic 사이의 경로·통신 계약을 검사한다."""

import json
import importlib.machinery
import importlib.util
import os
import re
import runpy
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import types
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    _loaded_module_origins,
    _policy_from_environment,
    _process_group_alive,
    _reserve_local_port,
    _set_ros_parameter,
    _validate_peer_report,
    _wait_for_ros_master,
    _xmlrpc_request,
    main,
    run_ros_compatibility_smoke,
    validate_loaded_module_origins,
)
from validator_inference.environment_setup import PURE_ROS_PACKAGES, build_ros_overlay
from validator_inference.model_environment import (
    ENVIRONMENT_STATE_FILENAME,
    EXPECTED_PYTHON_PLATFORM,
    EXPECTED_UV_VERSION,
    ROS_OVERLAY_MANIFEST_FILENAME,
    EnvironmentState,
    load_environment_manifest,
    select_model_environment,
    sha256_file,
    write_environment_state,
)


_POLICY_DIRECTORY = tempfile.TemporaryDirectory(prefix="validator-ros-policy-")
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_VALIDATOR_PACKAGE_ROOT = _PACKAGE_ROOT.parent


def _path_policy() -> RuntimePathPolicy:
    """실제 symlink interpreter를 가진 target runtime 허용 경로를 만든다."""
    root = Path(_POLICY_DIRECTORY.name)
    base = root / "uv/python/cpython-3.8.20"
    base_python = base / "bin/python3.8"
    base_python.parent.mkdir(parents=True, exist_ok=True)
    if not base_python.exists():
        base_python.write_bytes(b"test interpreter")
        base_python.chmod(0o755)
    stdlib = base / "lib/python3.8"
    (stdlib / "lib-dynload").mkdir(parents=True, exist_ok=True)
    environment = root / "workspace/.venvs/interfuser/builds/one"
    (environment / "bin").mkdir(parents=True, exist_ok=True)
    target_python = environment / "bin/python"
    if not target_python.exists():
        target_python.symlink_to(base_python)
    overlay = environment / ".ros-overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    catkin_python = root / "workspace/devel/lib/python3/dist-packages"
    catkin_python.mkdir(parents=True, exist_ok=True)
    source = root / "workspace/src/Validator_pkg/validator_inference/src"
    source.mkdir(parents=True, exist_ok=True)
    return RuntimePathPolicy(
        target_python=target_python,
        target_environment=environment,
        ros_overlay=overlay,
        catkin_python_path=catkin_python,
        target_stdlib_paths=(stdlib,),
        project_source_paths=(source,),
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


def _install_fake_torch_generated_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """실제 Torch invariant를 모사한 owner-only generated module을 설치한다."""
    name = "_remote_module_non_scriptable"
    directory = tmp_path / "torch-generated"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    origin = directory / f"{name}.py"
    template = (
        "{assign_module_interface_cls}\n"
        "{arg_types}\n"
        "{arrow_and_return_type}\n"
        "{arrow_and_future_return_type}\n"
        "{args}\n"
        "{kwargs}\n"
        "{jit_script_decorator}\n"
    )
    expected = template.format(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    origin.write_text(expected, encoding="utf-8")
    origin.chmod(0o600)

    generated = types.ModuleType(name)
    loader = importlib.machinery.SourceFileLoader(name, str(origin))
    specification = importlib.util.spec_from_loader(name, loader)
    assert specification is not None
    generated.__file__ = str(origin)
    generated.__loader__ = loader
    generated.__spec__ = specification

    instantiator = types.ModuleType("torch.distributed.nn.jit.instantiator")
    instantiator._FILE_PREFIX = "_remote_module_"
    instantiator._TEMP_DIR = SimpleNamespace(name=str(directory))
    instantiator.INSTANTIATED_TEMPLATE_DIR_PATH = str(directory)
    instantiator.get_remote_module_template = lambda enabled: (
        template if enabled is True else ""
    )
    remote_module = types.ModuleType("torch.distributed.nn.api.remote_module")
    remote_module.instantiator = instantiator
    remote_module._NON_SCRIPTABLE_REMOTE_MODULE_MODULE = generated

    monkeypatch.setitem(sys.modules, name, generated)
    monkeypatch.setitem(
        sys.modules, "torch.distributed.nn.jit.instantiator", instantiator
    )
    monkeypatch.setitem(
        sys.modules, "torch.distributed.nn.api.remote_module", remote_module
    )
    monkeypatch.syspath_prepend(str(directory))
    return generated, remote_module, origin


def test_loaded_module_origins_attests_exact_torch_generated_remote_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """살아 있는 pinned Torch template와 일치한 단일 임시 module만 분리한다."""
    _install_fake_torch_generated_module(tmp_path, monkeypatch)

    origins = _loaded_module_origins()

    assert "_remote_module_non_scriptable" not in origins


def test_loaded_module_origins_accepts_torch_file_from_standard_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """owner-only Torch tempdir 안의 일반 umask 022 source mode를 허용한다."""
    _, _, origin = _install_fake_torch_generated_module(tmp_path, monkeypatch)
    origin.chmod(0o644)

    origins = _loaded_module_origins()

    assert "_remote_module_non_scriptable" not in origins


def test_loaded_module_origins_accepts_torch_file_from_collaborative_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """owner-only Torch tempdir 안의 umask 002 source mode도 허용한다."""
    _, _, origin = _install_fake_torch_generated_module(tmp_path, monkeypatch)
    origin.chmod(0o664)

    origins = _loaded_module_origins()

    assert "_remote_module_non_scriptable" not in origins


@pytest.mark.parametrize(
    ("directory_mode", "source_mode"),
    ((0o770, 0o664), (0o700, 0o674), (0o700, 0o2664)),
)
def test_loaded_module_origins_rejects_unsafe_collaborative_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_mode: int,
    source_mode: int,
):
    """group-write는 owner-only 부모와 비실행 일반 source일 때만 허용한다."""
    _, _, origin = _install_fake_torch_generated_module(tmp_path, monkeypatch)
    origin.parent.chmod(directory_mode)
    origin.chmod(source_mode)

    origins = _loaded_module_origins()

    assert origins["_remote_module_non_scriptable"] == str(origin)


def test_loaded_module_origins_does_not_execute_dynamic_file_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """torch.classes 같은 namespace proxy의 __getattr__ 경로를 origin으로 오인하지 않는다."""
    class DynamicModule(types.ModuleType):
        def __getattr__(self, name):
            if name == "__file__":
                return str(tmp_path / "_classes.py")
            raise AttributeError(name)

    proxy = DynamicModule("torch.classes")
    monkeypatch.setitem(sys.modules, "torch.classes", proxy)

    origins = _loaded_module_origins()

    assert "torch.classes" not in origins


@pytest.mark.parametrize(
    "mutation",
    ("identity", "source", "spec", "loader", "permissions"),
)
def test_loaded_module_origins_keeps_unattested_torch_temp_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
):
    """이름이 같아도 Torch identity·bytes·path·권한 하나가 다르면 감사에서 뺄 수 없다."""
    generated, remote_module, origin = _install_fake_torch_generated_module(
        tmp_path, monkeypatch
    )
    if mutation == "identity":
        remote_module._NON_SCRIPTABLE_REMOTE_MODULE_MODULE = types.ModuleType(
            "_remote_module_non_scriptable"
        )
    elif mutation == "source":
        origin.write_text("FOREIGN = True\n", encoding="utf-8")
    elif mutation == "spec":
        generated.__spec__.origin = str(origin.with_name("foreign.py"))
    elif mutation == "loader":
        generated.__loader__.path = str(origin.with_name("foreign.py"))
    else:
        origin.chmod(0o666)

    origins = _loaded_module_origins()

    assert origins["_remote_module_non_scriptable"] == str(origin)


def test_validate_loaded_module_origins_still_rejects_named_torch_temp_file(
    tmp_path: Path,
):
    """collector attestation 없이 같은 Torch 이름만 제시한 /tmp file은 허용하지 않는다."""
    origin = tmp_path / "_remote_module_non_scriptable.py"
    origin.write_text("FOREIGN = True\n", encoding="utf-8")

    with pytest.raises(RosCompatibilityError, match="allowed runtime root"):
        validate_loaded_module_origins(
            {"_remote_module_non_scriptable": origin}, _path_policy()
        )


def test_validate_loaded_module_origins_accepts_only_exact_attested_source_native():
    """source tree native는 policy가 명시한 exact artifact만 예외로 허용한다."""
    policy = _path_policy()
    native = (
        policy.project_source_paths[0]
        / "ops"
        / "declared.cpython-38-x86_64-linux-gnu.so"
    )
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native")
    attested = replace(
        policy,
        trusted_native_source_paths=(native.parent,),
        trusted_native_extension_paths=(native,),
    )

    validate_loaded_module_origins({"declared": native}, attested)

    sibling = native.with_name("undeclared.cpython-38-x86_64-linux-gnu.so")
    sibling.write_bytes(b"native")
    with pytest.raises(RosCompatibilityError, match="outside the target interpreter"):
        validate_loaded_module_origins({"undeclared": sibling}, attested)


@pytest.mark.parametrize("invalid_kind", ("directory", "outside", "symlink"))
def test_runtime_path_policy_rejects_unsafe_attested_source_native(invalid_kind):
    """attested native 예외도 canonical source regular file만 허용한다."""
    policy = _path_policy()
    source = policy.project_source_paths[0]
    outside = source.parent / f"outside-{invalid_kind}.cpython-38-x86_64-linux-gnu.so"
    outside.write_bytes(b"outside")
    candidate = (
        source
        / "ops"
        / f"candidate-{invalid_kind}.cpython-38-x86_64-linux-gnu.so"
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if invalid_kind == "directory":
        candidate.mkdir()
        reason = "regular file"
    elif invalid_kind == "outside":
        candidate = outside
        reason = "project source"
    else:
        real = candidate.with_name("real.cpython-38-x86_64-linux-gnu.so")
        real.write_bytes(b"native")
        candidate.symlink_to(real.name)
        reason = "symlink"
    unsafe = replace(
        policy,
        trusted_native_source_paths=(source,),
        trusted_native_extension_paths=(candidate,),
    )

    with pytest.raises(RosCompatibilityError, match=reason):
        validate_loaded_module_origins({"candidate": candidate}, unsafe)


def test_trusted_source_module_path_is_scoped_to_exact_module_namespace():
    """vendored pure source root는 선언한 module namespace에만 적용한다."""
    policy = _path_policy()
    project_root = policy.target_environment.parents[3]
    timm = project_root / "models" / "sources" / "interfuser" / "interfuser" / "timm"
    timm.mkdir(parents=True)
    module = timm / "models" / "interfuser.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    scoped = replace(policy, trusted_source_module_paths=(("timm", timm),))

    validate_loaded_module_origins({"timm.models.interfuser": module}, scoped)

    with pytest.raises(RosCompatibilityError, match="allowed runtime root"):
        validate_loaded_module_origins({"foreign.models.interfuser": module}, scoped)


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


def test_validate_loaded_module_origins_rejects_foreign_base_stdlib():
    """target interpreter prefix가 소유하지 않은 일반 stdlib root를 거부한다."""
    policy = _path_policy()
    foreign_stdlib = policy.target_environment.parent / "foreign/lib/python3.8"
    foreign_stdlib.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RosCompatibilityError, match="base stdlib"):
        validate_loaded_module_origins(
            {"json": foreign_stdlib / "json/__init__.py"},
            replace(policy, target_stdlib_paths=(foreign_stdlib,)),
        )


def test_validate_loaded_module_origins_rejects_stdlib_site_package_extension():
    """base stdlib 아래라도 exact lib-dynload 밖의 native extension은 거부한다."""
    policy = _path_policy()
    with pytest.raises(RosCompatibilityError, match="lib-dynload"):
        validate_loaded_module_origins(
            {
                "foreign_native": (
                    policy.target_stdlib_paths[0]
                    / "site-packages/foreign_native.cpython-38-x86_64-linux-gnu.so"
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


def test_xmlrpc_request_uses_only_the_remaining_monotonic_deadline():
    """XML-RPC connection timeout이 전체 deadline의 남은 시간보다 길지 않다."""
    observed = {}
    clock = iter((10.0, 11.0))

    class Proxy:
        """호출 인자와 정상 ROS XML-RPC 응답을 제공한다."""

        def getPid(self, *arguments):
            """getPid 인자를 기록하고 성공 응답을 반환한다."""
            observed["arguments"] = arguments
            return [1, "ready", 42]

        def close(self):
            """ServerProxy close 호출을 기록한다."""
            observed["closed"] = True

    def proxy_factory(uri, timeout_seconds):
        """계산된 per-request socket timeout을 기록한다."""
        observed["uri"] = uri
        observed["timeout"] = timeout_seconds
        return Proxy()

    response = _xmlrpc_request(
        "http://127.0.0.1:43210/",
        "getPid",
        ("/validator_ros_compatibility",),
        15.0,
        monotonic=lambda: next(clock),
        proxy_factory=proxy_factory,
    )

    assert response == [1, "ready", 42]
    assert observed == {
        "uri": "http://127.0.0.1:43210/",
        "timeout": pytest.approx(5.0),
        "arguments": ("/validator_ros_compatibility",),
        "closed": True,
    }


def test_xmlrpc_request_closes_server_proxy_transport_without_remote_close():
    """Python 3.8 ServerProxy의 동적 ``close`` 원격 method를 호출하지 않는다."""
    observed = []

    class Proxy:
        """ServerProxy의 private local close와 동적 remote method 동작을 흉내 낸다."""

        def getPid(self, *_arguments):
            """정상 master 응답을 반환한다."""
            return [1, "ready", 42]

        def _ServerProxy__close(self):
            """실제 ServerProxy context manager가 쓰는 local transport close다."""
            observed.append("local")

        def __getattr__(self, name):
            """존재하지 않는 이름이 원격 method가 되는 동작을 재현한다."""
            if name == "close":
                return lambda: observed.append("remote")
            raise AttributeError(name)

    _xmlrpc_request(
        "http://127.0.0.1:43210/",
        "getPid",
        ("/validator_ros_compatibility",),
        15.0,
        monotonic=iter((10.0, 11.0)).__next__,
        proxy_factory=lambda _uri, _timeout: Proxy(),
    )

    assert observed == ["local"]


def test_xmlrpc_request_interrupts_slow_blocking_call_at_absolute_deadline():
    """지속적으로 깨어나는 blocking call도 absolute deadline에서 중단한다."""
    observed = []

    class Proxy:
        """deadline보다 오래 조금씩 진행되는 XML-RPC peer를 흉내 낸다."""

        def getPid(self, *_arguments):
            """socket slow-trickle처럼 짧은 blocking 구간을 반복한다."""
            limit = time.monotonic() + 1.0
            while time.monotonic() < limit:
                time.sleep(0.01)
            return [1, "late", 42]

        def _ServerProxy__close(self):
            """deadline 뒤에도 local transport cleanup이 호출됐음을 기록한다."""
            observed.append("closed")

    started = time.monotonic()
    with pytest.raises(RosCompatibilityError, match="deadline"):
        _xmlrpc_request(
            "http://127.0.0.1:43210/",
            "getPid",
            ("/validator_ros_compatibility",),
            started + 0.05,
            proxy_factory=lambda _uri, _timeout: Proxy(),
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert observed == ["closed"]


def test_xmlrpc_request_bounds_blocking_transport_cleanup_after_deadline():
    """request deadline 뒤 transport close도 짧은 cleanup bound를 넘지 않는다."""
    observed = []

    class Proxy:
        """request와 local close가 모두 영구 blocking할 수 있는 proxy다."""

        def getPid(self, *_arguments):
            """absolute request deadline까지 blocking한다."""
            time.sleep(1.0)

        def _ServerProxy__close(self):
            """두 번째 alarm이 cleanup을 중단하는지 앞뒤 marker로 드러낸다."""
            observed.append("close-started")
            time.sleep(1.0)
            observed.append("close-finished")

    started = time.monotonic()
    with pytest.raises(RosCompatibilityError, match="deadline"):
        _xmlrpc_request(
            "http://127.0.0.1:43210/",
            "getPid",
            ("/validator_ros_compatibility",),
            started + 0.05,
            proxy_factory=lambda _uri, _timeout: Proxy(),
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert observed == ["close-started"]


def test_xmlrpc_request_restores_process_alarm_when_transport_close_raises():
    """local close 예외에서도 process-global SIGALRM 상태를 유출하지 않는다."""
    previous_handler = signal.getsignal(signal.SIGALRM)

    class Proxy:
        """정상 응답 뒤 local close만 실패하는 proxy다."""

        def getPid(self, *_arguments):
            """deadline 전에 정상 master 응답을 반환한다."""
            return [1, "ready", 42]

        def _ServerProxy__close(self):
            """alarm restoration 경로를 통과시키는 cleanup 오류를 낸다."""
            raise RuntimeError("forced close failure")

    try:
        with pytest.raises(RuntimeError, match="forced close failure"):
            _xmlrpc_request(
                "http://127.0.0.1:43210/",
                "getPid",
                ("/validator_ros_compatibility",),
                time.monotonic() + 1.0,
                proxy_factory=lambda _uri, _timeout: Proxy(),
            )
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGALRM) is previous_handler
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def test_master_readiness_requires_a_successful_xmlrpc_response():
    """listener가 열렸더라도 getPid XML-RPC 성공 응답 전에는 ready가 아니다."""
    responses = iter((OSError("accepted TCP only"), [0, "not ready", 0], [1, "ok", 7]))
    calls = []
    now = {"value": 0.0}

    def monotonic():
        """readiness retry마다 bounded clock을 전진시킨다."""
        now["value"] += 0.1
        return now["value"]

    def request(uri, method, arguments, deadline):
        """TCP-only 실패, invalid 응답, 정상 응답을 순서대로 돌려준다."""
        calls.append((uri, method, arguments, deadline))
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    assert _wait_for_ros_master(
        "http://127.0.0.1:43210/",
        2.0,
        monotonic=monotonic,
        sleep=lambda _seconds: None,
        request=request,
    ) is True
    assert len(calls) == 3
    assert all(call[1] == "getPid" for call in calls)


def test_set_parameter_forwards_the_overall_deadline():
    """setParam도 readiness와 같은 absolute deadline으로 제한한다."""
    calls = []

    def request(uri, method, arguments, deadline):
        """setParam 요청의 URI, 인자와 deadline을 기록한다."""
        calls.append((uri, method, arguments, deadline))
        return [1, "set", 0]

    _set_ros_parameter(
        "http://127.0.0.1:43210/",
        "/use_sim_time",
        True,
        9.5,
        request=request,
    )
    assert calls == [
        (
            "http://127.0.0.1:43210/",
            "setParam",
            ("/validator_ros_compatibility", "/use_sim_time", True),
            9.5,
        )
    ]


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

    def __init__(self, pid, returncode=None, stubborn=False, group_alive=None):
        """leader 상태와 별도 process-group 생존 상태를 초기화한다."""
        self.pid = pid
        self.returncode = returncode
        self.stubborn = stubborn
        self.group_alive = returncode is None if group_alive is None else group_alive

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
    control_calls = {"master": [], "parameter": []}
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
            process.group_alive = False
            if process.returncode is None:
                process.returncode = -selected_signal

    def group_alive(pid):
        """leader 종료와 독립적인 process-group 생존 상태를 반환한다."""
        process = next(item for item in processes if item.pid == pid)
        return process.group_alive

    def monotonic():
        """timeout case에서만 매 호출마다 deadline을 전진시킨다."""
        if timeout:
            clock["value"] += 1.0
        return clock["value"]

    def master_ready(_uri, deadline):
        """readiness에 전달된 overall deadline을 기록한다."""
        control_calls["master"].append(deadline)
        return True

    def set_parameter(_uri, _name, _value, deadline):
        """setParam에 동일하게 전달된 overall deadline을 기록한다."""
        control_calls["parameter"].append(deadline)

    dependencies = RosCompatibilityDependencies(
        popen=popen,
        killpg=killpg,
        process_group_alive=group_alive,
        monotonic=monotonic,
        sleep=lambda _seconds: None,
        master_ready=master_ready,
        set_parameter=set_parameter,
        load_report=lambda path: reports[path.name],
        reserve_port=lambda: 43210,
    )
    return dependencies, calls, signals, control_calls


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
    dependencies, calls, signals, control_calls = _dependencies(processes, reports)

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
    assert calls[1][0][1:3] == ["-B", "-m"]
    assert calls[2][0][1:3] == ["-B", "-m"]
    system_environment = calls[1][1]["env"]
    assert system_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    target_environment = calls[2][1]["env"]
    assert target_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert target_environment["PYTHONPATH"] == ":".join(
        (str(policy.ros_overlay), str(policy.catkin_python_path))
    )
    assert target_environment["VIRTUAL_ENV"] == str(policy.target_environment)
    assert "PYTHONHOME" not in target_environment
    assert control_calls == {"master": [5.0], "parameter": [5.0]}
    assert signals == [(1001, signal.SIGTERM)]
    assert not any(process.group_alive for process in processes)


def test_smoke_report_preserves_protocol_order_after_sorted_json_round_trip():
    """sorted JSON object 순서와 무관하게 report type 순서는 protocol을 따른다."""
    policy = _path_policy()
    processes = [
        _FakeProcess(1101),
        _FakeProcess(1102, returncode=0),
        _FakeProcess(1103, returncode=0),
    ]
    reports = {
        name: json.loads(json.dumps(_peer_report(role, policy), sort_keys=True))
        for name, role in (("system.json", "system"), ("target.json", "target"))
    }
    dependencies, _calls, _signals, _control_calls = _dependencies(
        processes, reports
    )

    result = run_ros_compatibility_smoke(
        policy,
        system_python=Path("/usr/bin/python3"),
        dependencies=dependencies,
    )

    assert result.system_received == MESSAGE_TYPES
    assert result.target_received == MESSAGE_TYPES


def test_run_ros_compatibility_smoke_cleans_started_groups_on_peer_start_failure():
    """target peer 시작이 실패해도 먼저 시작한 roscore와 system peer를 종료한다."""
    policy = _path_policy()
    processes = [_FakeProcess(2001), _FakeProcess(2002)]
    dependencies, calls, signals, _control_calls = _dependencies(
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
    dependencies, _calls, signals, _control_calls = _dependencies(
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
    assert not any(process.group_alive for process in processes)


def test_cleanup_terminates_descendants_after_the_group_leader_exits():
    """leader가 이미 성공 종료했어도 같은 PGID의 살아 있는 descendant를 종료한다."""
    policy = _path_policy()
    processes = [
        _FakeProcess(4001),
        _FakeProcess(4002, returncode=0, group_alive=True),
        _FakeProcess(4003, returncode=0),
    ]
    reports = {
        "system.json": _peer_report("system", policy),
        "target.json": _peer_report("target", policy),
    }
    dependencies, _calls, signals, _control_calls = _dependencies(
        processes, reports
    )

    run_ros_compatibility_smoke(
        policy,
        system_python=Path("/usr/bin/python3"),
        dependencies=dependencies,
    )

    assert signals == [
        (4002, signal.SIGTERM),
        (4001, signal.SIGTERM),
    ]
    assert not any(process.group_alive for process in processes)


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


def _build_catkin_install_prefix(tmp_path):
    """필요한 두 package만 가진 격리 Catkin workspace를 실제 install한다."""
    workspace = tmp_path / "catkin_install_workspace"
    validator_root = workspace / "src/Validator_pkg"
    validator_root.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    shutil.copytree(
        _VALIDATOR_PACKAGE_ROOT / "validator_inference",
        validator_root / "validator_inference",
        ignore=ignore,
    )
    shutil.copytree(
        _VALIDATOR_PACKAGE_ROOT / "validator_msgs",
        validator_root / "validator_msgs",
        ignore=ignore,
    )
    shutil.copytree(
        _VALIDATOR_PACKAGE_ROOT / "config",
        validator_root / "config",
        ignore=ignore,
    )
    install_prefix = (tmp_path / "isolated_install_prefix").resolve()
    environment = dict(os.environ)
    environment["PWD"] = str(workspace)
    result = subprocess.run(
        [
            "catkin_make",
            "install",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            "-DPYTHON_EXECUTABLE=/usr/bin/python3",
        ],
        cwd=workspace,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert f"Base path: {workspace}" in result.stdout
    assert f"Source space: {workspace / 'src'}" in result.stdout
    assert f"Install space: {install_prefix}" in result.stdout
    return install_prefix


def _minimal_ros_source(root):
    """wrapper deep verification에 필요한 exact pure-Python overlay source를 만든다."""
    source = root / "minimal ros source"
    for package in PURE_ROS_PACKAGES:
        package_root = source / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            f"PACKAGE = {package!r}\n", encoding="utf-8"
        )
    return source


def _write_signal_stub(path):
    """argv/environment/signal/exit 상태를 JSON으로 남기는 target Python stub을 쓴다."""
    path.parent.mkdir(parents=True)
    path.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "capture = Path(os.environ['ROSLAUNCH_CAPTURE_PATH'])\n"
        "payload = {\n"
        "    'argv': sys.argv,\n"
        "    'pid': os.getpid(),\n"
        "    'pgid': os.getpgrp(),\n"
        "    'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "    'virtual_env': os.environ.get('VIRTUAL_ENV'),\n"
        "    'signal': None,\n"
        "    'exit_status': None,\n"
        "}\n"
        "def write():\n"
        "    temporary = capture.with_suffix('.tmp')\n"
        "    temporary.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')\n"
        "    os.replace(str(temporary), str(capture))\n"
        "def finish(signum, status):\n"
        "    payload['signal'] = signum\n"
        "    payload['exit_status'] = status\n"
        "    write()\n"
        "    raise SystemExit(status)\n"
        "signal.signal(signal.SIGINT, lambda signum, _frame: finish(signum, 23))\n"
        "signal.signal(signal.SIGTERM, lambda signum, _frame: finish(signum, 24))\n"
        "write()\n"
        "if os.environ.get('ROSLAUNCH_STUB_MODE') == 'required-exit':\n"
        "    time.sleep(0.25)\n"
        "    finish(0, 23)\n"
        "while True:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _prepare_installed_model_environment(tmp_path, install_prefix):
    """installed wrapper가 검증할 active state, overlay와 signal stub을 구성한다."""
    project_root = (tmp_path / "installed runtime project").resolve()
    project_root.mkdir()
    package_root = install_prefix / "share/validator_inference"
    manifest_path = package_root / "config/model_environments.json"
    manifest = load_environment_manifest(manifest_path)
    definition = select_model_environment(manifest, "interfuser")
    environment_dir = (
        project_root / ".venvs/interfuser/builds/install-signal-test"
    )
    environment_dir.mkdir(parents=True)
    target_python = environment_dir / "bin/python"
    _write_signal_stub(target_python)
    overlay = environment_dir / ".ros-overlay"
    build_ros_overlay(_minimal_ros_source(tmp_path), overlay)
    overlay_manifest = environment_dir / ROS_OVERLAY_MANIFEST_FILENAME
    state = EnvironmentState(
        schema=1,
        model_id="interfuser",
        python_version=definition.python_version,
        python_platform=EXPECTED_PYTHON_PLATFORM,
        uv_version=EXPECTED_UV_VERSION,
        manifest_sha256=sha256_file(manifest_path),
        lock_sha256=definition.requirements_lock_sha256,
        ros_overlay_sha256=sha256_file(overlay_manifest),
        source_commit=definition.source_commit,
        installed_packages_sha256="a" * 64,
        python_executable=Path("bin/python"),
        assets=(),
        completed_at="2026-07-22T00:00:00Z",
        preflight_ids=(),
    )
    write_environment_state(environment_dir / ENVIRONMENT_STATE_FILENAME, state)
    current = project_root / ".venvs/interfuser/current"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(Path("builds") / environment_dir.name)
    lifecycle_lock = current.parent / ".lifecycle.lock"
    lifecycle_lock.write_text("", encoding="utf-8")
    lifecycle_lock.chmod(0o600)
    return project_root, environment_dir, overlay


def _installed_environment(install_prefix, ros_home, port):
    """install prefix setup.bash를 source한 ROS tool environment를 읽는다."""
    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1/setup.bash" && env -0',
            "validator-install-environment",
            str(install_prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    environment = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode()] = value.decode()
    ros_home.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "ROS_HOME": str(ros_home),
            "ROS_LOG_DIR": str(ros_home / "log"),
            "ROS_MASTER_URI": f"http://127.0.0.1:{port}/",
        }
    )
    return environment


def _write_installed_launch(
    path, install_prefix, project_root, capture_path, mode
):
    """installed wrapper prefix와 required stub 하나만 포함한 launch XML을 쓴다."""
    launch = ET.Element("launch")
    node = ET.SubElement(
        launch,
        "node",
        {
            "pkg": "validator_inference",
            "type": "inference_engine_node.py",
            "name": "installed_signal_stub",
            "required": "true",
            "output": "screen",
            "args": "literal_argument",
            "launch-prefix": '"{}"'.format(
                install_prefix / "share/validator_inference/scripts/model_env_exec"
            ),
        },
    )
    values = {
        "OPENLOOP_MODEL_ENV_ROOT": project_root / ".venvs",
        "OPENLOOP_MODEL_ID": "interfuser",
        "OPENLOOP_PROJECT_ROOT": project_root,
        "ROSLAUNCH_CAPTURE_PATH": capture_path,
        "ROSLAUNCH_STUB_MODE": mode,
    }
    for name, value in values.items():
        ET.SubElement(node, "env", {"name": name, "value": str(value)})
    ET.SubElement(
        node,
        "remap",
        {"from": "camera/image", "to": "/recorded_camera/image_raw"},
    )
    ET.ElementTree(launch).write(path, encoding="utf-8", xml_declaration=True)


def _wait_for_path(path, process, timeout_seconds=30.0):
    """process 조기 종료를 감시하며 capture file을 bounded wait한다."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            output, _unused = process.communicate()
            pytest.fail(f"roslaunch exited before capture ({process.returncode}):\n{output}")
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for roslaunch capture: {path}")


def _pid_exists(pid):
    """signal 0으로 한 process ID가 아직 존재하는지 확인한다."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run_installed_launch(launch_path, environment, capture_path, send_sigint):
    """real roslaunch를 독립 group으로 실행하고 signal/required 종료를 수집한다."""
    process = subprocess.Popen(
        ["roslaunch", str(launch_path)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    output = ""
    try:
        _wait_for_path(capture_path, process)
        initial = json.loads(capture_path.read_text(encoding="utf-8"))
        if send_sigint:
            process.send_signal(signal.SIGINT)
        output, _unused = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    final = json.loads(capture_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5.0
    while _pid_exists(initial["pid"]) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(initial["pid"])
    assert not _process_group_alive(process.pid)
    return process.returncode, output, initial, final


@pytest.mark.ros_compatibility
def test_real_install_prefix_and_roslaunch_signal_contract(tmp_path):
    """실제 install wrapper의 경로 파생, SIGINT, required shutdown과 정리를 검증한다."""
    if os.environ.get("VALIDATOR_RUN_ROS_COMPAT") != "1":
        pytest.skip("set VALIDATOR_RUN_ROS_COMPAT=1 for install/roslaunch integration")

    install_prefix = _build_catkin_install_prefix(tmp_path)
    wrapper = install_prefix / "share/validator_inference/scripts/model_env_exec"
    relay = install_prefix / "lib/validator_inference/inference_engine_node.py"
    catkin_python = install_prefix / "lib/python3/dist-packages"
    assert wrapper.read_bytes().splitlines()[0] == b"#!/usr/bin/python3"
    assert wrapper.read_bytes() == (
        _VALIDATOR_PACKAGE_ROOT / "validator_inference/scripts/model_env_exec"
    ).read_bytes()
    assert relay.is_file()
    assert (catkin_python / "validator_inference/__init__.py").is_file()
    assert (catkin_python / "validator_msgs/msg/__init__.py").is_file()

    project_root, environment_dir, overlay = _prepare_installed_model_environment(
        tmp_path, install_prefix
    )
    ros_home = tmp_path / "ros home"
    launch_environment = _installed_environment(
        install_prefix, ros_home, 1
    )
    installed_cli = runpy.run_path(
        str(install_prefix / "lib/validator_inference/openloop_cli.py")
    )
    launch_argv = installed_cli["_launch_argv"](
        project_root, "interfuser", False
    )
    launch_arguments = subprocess.run(
        [launch_argv[0], "--args", "/inference_engine", *launch_argv[1:]],
        env=launch_environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    ).stdout
    assert 'OPENLOOP_PROJECT_ROOT="{}"'.format(project_root) in launch_arguments
    assert 'OPENLOOP_MODEL_ENV_ROOT="{}"'.format(
        project_root / ".venvs"
    ) in launch_arguments
    assert str(wrapper) in launch_arguments
    assert str(relay) in launch_arguments

    launch_parameters = yaml.safe_load(
        subprocess.run(
            [launch_argv[0], "--dump-params", *launch_argv[1:]],
            env=launch_environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
    )
    profile_file = project_root / "src/Validator_pkg/config/profile.json"
    sensor_config = (
        project_root
        / "src/Validator_pkg/validator_pipeline/config/0725demo_ros.json"
    )
    assert launch_parameters["/inference_engine/profile_file"] == str(profile_file)
    assert launch_parameters["/opt_feeder/profile_file"] == str(profile_file)
    assert launch_parameters["/bev_visualizer/profile_file"] == str(profile_file)
    assert launch_parameters["/recorder/record_dir"] == str(
        project_root / "records/openloop"
    )
    assert launch_parameters["/image_adaptor/config_path"] == str(sensor_config)
    assert launch_parameters["/lidar_bev_processor/config_path"] == str(
        sensor_config
    )
    assert launch_parameters["/camera_bev_stitcher/config_path"] == str(
        sensor_config
    )

    signal_capture = tmp_path / "signal capture.json"
    signal_launch = tmp_path / "signal.launch"
    _write_installed_launch(
        signal_launch,
        install_prefix,
        project_root,
        signal_capture,
        "wait-for-signal",
    )
    signal_environment = _installed_environment(
        install_prefix, ros_home, _reserve_local_port()
    )
    _returncode, signal_output, initial, signaled = _run_installed_launch(
        signal_launch, signal_environment, signal_capture, True
    )
    assert initial["pythonpath"] == os.pathsep.join((str(overlay), str(catkin_python)))
    assert initial["virtual_env"] == str(environment_dir)
    assert initial["argv"][0] == str(environment_dir / "bin/python")
    assert initial["argv"][1] == str(relay)
    assert "literal_argument" in initial["argv"]
    assert "camera/image:=/recorded_camera/image_raw" in initial["argv"]
    assert signaled["signal"] == signal.SIGINT
    assert signaled["exit_status"] == 23
    started_pid = re.search(
        r"process\[installed_signal_stub-\d+\]: started with pid \[(\d+)\]",
        signal_output,
    )
    assert started_pid is not None, signal_output
    assert int(started_pid.group(1)) == initial["pid"]

    required_capture = tmp_path / "required capture.json"
    required_launch = tmp_path / "required.launch"
    _write_installed_launch(
        required_launch,
        install_prefix,
        project_root,
        required_capture,
        "required-exit",
    )
    required_environment = _installed_environment(
        install_prefix, ros_home, _reserve_local_port()
    )
    _returncode, required_output, _initial, required = _run_installed_launch(
        required_launch, required_environment, required_capture, False
    )
    assert required["signal"] == 0
    assert required["exit_status"] == 23
    assert "REQUIRED process [installed_signal_stub-" in required_output
    assert "Initiating shutdown!" in required_output


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
                "print(json.dumps(sysconfig.get_path('stdlib')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    base_stdlib = Path(json.loads(completed.stdout))
    policy = RuntimePathPolicy(
        target_python=target_python,
        target_environment=target_environment,
        ros_overlay=ros_overlay,
        catkin_python_path=catkin_python_path,
        target_stdlib_paths=(base_stdlib,),
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
