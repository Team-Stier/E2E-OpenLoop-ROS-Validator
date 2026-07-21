"""
모델별 Python interpreter와 ROS Noetic 사이의 교차-version 계약을 검증한다.
입력: target runtime 경로 정책, loaded module origin과 ROS smoke 실행 설정.
출력(pub): 양방향 typed smoke topic과 ``/clock`` fixture를 발행한다.
구독(sub): 반대 peer의 typed smoke topic과 ``/clock``을 구독한다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit
import xmlrpc.client


ROS_PACKAGE_NAMES = (
    "rospy",
    "rosgraph",
    "roslib",
    "genpy",
    "genmsg",
    "catkin",
    "roscpp",
    "rosgraph_msgs",
    "std_msgs",
    "sensor_msgs",
    "geometry_msgs",
    "nav_msgs",
    "tf2_msgs",
    "actionlib_msgs",
)
MESSAGE_TYPES = (
    "sensor_msgs/Image",
    "sensor_msgs/PointCloud2",
    "validator_msgs/DrivingFeature",
    "tf2_msgs/TFMessage",
    "nav_msgs/Path",
    "validator_msgs/InferenceMetric",
    "rosgraph_msgs/Clock",
)
_REPORT_KEYS = {
    "schema",
    "role",
    "python_executable",
    "pythonpath",
    "pythonhome",
    "use_sim_time",
    "clock_before_ns",
    "clock_after_ns",
    "sent_types",
    "received",
    "inference_imports",
    "module_origins",
}
_REQUIRED_INFERENCE_IMPORTS = (
    "validator_inference.engine",
    "validator_inference.models.select",
)
_NATIVE_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES) + (
    ".so",
    ".pyd",
    ".dylib",
    ".dll",
)
_VERSIONED_PYTHON = re.compile(r"python([0-9]+)\.([0-9]+)\Z")


class RosCompatibilityError(RuntimeError):
    """교차 interpreter ROS runtime이 격리 또는 통신 계약을 어길 때 발생한다."""


@dataclass(frozen=True)
class RuntimePathPolicy:
    """target interpreter가 module을 적재할 수 있는 신뢰 경로를 보관한다."""

    target_python: Path
    target_environment: Path
    ros_overlay: Path
    catkin_python_path: Path
    target_stdlib_paths: tuple[Path, ...]
    project_source_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RosSmokeReport:
    """검증을 마친 system·target peer의 interpreter와 양방향 수신 결과다."""

    system_python: Path
    target_python: Path
    target_pythonpath: tuple[Path, ...]
    message_types: tuple[str, ...]
    system_received: tuple[str, ...]
    target_received: tuple[str, ...]
    use_sim_time: bool
    system_clock_advanced: bool
    target_clock_advanced: bool
    target_inference_imports: tuple[str, ...]
    target_module_origins: tuple[tuple[str, str], ...]


def _resolved_path(path: Path) -> Path:
    """존재하지 않는 unit-test 경로도 비교할 수 있게 절대 정규 경로로 바꾼다."""
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    """정규화한 path가 같은 root 또는 그 하위인지 판별한다."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_native_extension(path: Path) -> bool:
    """module 파일명이 CPython 또는 플랫폼 native extension suffix인지 판별한다."""
    return any(path.name.endswith(suffix) for suffix in _NATIVE_SUFFIXES)


def _origin_path(origin: object, module_name: str) -> Optional[Path]:
    """built-in·frozen origin은 제외하고 file origin을 정규 경로로 변환한다."""
    if origin is None or (
        isinstance(origin, str) and origin in {"built-in", "frozen"}
    ):
        return None
    if not isinstance(origin, (str, Path)) or not str(origin):
        raise RosCompatibilityError(
            f"module {module_name} has an invalid origin: {origin!r}"
        )
    return _resolved_path(Path(origin))


def _validate_runtime_path_policy(policy: RuntimePathPolicy) -> None:
    """path allowlist가 absolute 최소 root와 고정 target layout을 지키는지 검사한다."""
    if not isinstance(policy, RuntimePathPolicy):
        raise RosCompatibilityError("runtime path policy has an invalid type")
    named_paths = {
        "target environment": policy.target_environment,
        "target Python": policy.target_python,
        "ROS overlay": policy.ros_overlay,
        "Catkin Python": policy.catkin_python_path,
    }
    for context, value in named_paths.items():
        path = Path(value)
        if not path.is_absolute() or path == Path("/"):
            raise RosCompatibilityError(f"{context} must be a bounded absolute path")
    if policy.target_python != policy.target_environment / "bin" / "python":
        raise RosCompatibilityError("target Python must be target environment/bin/python")
    if policy.ros_overlay != policy.target_environment / ".ros-overlay":
        raise RosCompatibilityError("ROS overlay must be inside the target environment")
    try:
        resolved_python = policy.target_python.resolve(strict=True)
    except OSError as exc:
        raise RosCompatibilityError("target Python must resolve to an existing file") from exc
    if not resolved_python.is_file() or not os.access(str(resolved_python), os.X_OK):
        raise RosCompatibilityError("target Python must resolve to an executable file")
    version_match = _VERSIONED_PYTHON.fullmatch(resolved_python.name)
    if version_match is None:
        raise RosCompatibilityError("resolved target Python must be named pythonX.Y")
    expected_stdlib = (
        resolved_python.parent.parent
        / "lib"
        / f"python{version_match.group(1)}.{version_match.group(2)}"
    )
    try:
        expected_stdlib = expected_stdlib.resolve(strict=True)
    except OSError as exc:
        raise RosCompatibilityError("resolved target Python base stdlib is missing") from exc
    if len(policy.target_stdlib_paths) != 1:
        raise RosCompatibilityError("target must declare exactly one base stdlib path")
    try:
        declared_stdlib = policy.target_stdlib_paths[0].resolve(strict=True)
    except OSError as exc:
        raise RosCompatibilityError("declared target base stdlib is missing") from exc
    if declared_stdlib != expected_stdlib:
        raise RosCompatibilityError(
            "target base stdlib does not belong to the resolved target Python"
        )
    opt_ros = Path("/opt/ros").resolve(strict=False)
    if _is_within(declared_stdlib, opt_ros):
        raise RosCompatibilityError("target stdlib paths must not trust /opt/ros")
    for source_path in policy.project_source_paths:
        source = Path(source_path)
        if not source.is_absolute() or source == Path("/"):
            raise RosCompatibilityError("project source paths must be bounded and absolute")


def validate_loaded_module_origins(
    module_origins: Mapping[str, object], policy: RuntimePathPolicy
) -> None:
    """loaded module이 target venv·overlay·Catkin·자체 stdlib 밖에 없음을 검증한다."""
    _validate_runtime_path_policy(policy)
    if not isinstance(module_origins, Mapping):
        raise RosCompatibilityError("module origins must be a mapping")

    target_environment = _resolved_path(policy.target_environment)
    ros_overlay = _resolved_path(policy.ros_overlay)
    catkin_python = _resolved_path(policy.catkin_python_path)
    stdlib_paths = tuple(_resolved_path(path) for path in policy.target_stdlib_paths)
    stdlib_lib_dynload = stdlib_paths[0] / "lib-dynload"
    source_paths = tuple(_resolved_path(path) for path in policy.project_source_paths)
    allowed_roots = (
        target_environment,
        ros_overlay,
        catkin_python,
        *stdlib_paths,
        *source_paths,
    )

    for module_name, raw_origin in sorted(module_origins.items()):
        if not isinstance(module_name, str) or not module_name:
            raise RosCompatibilityError("module origin names must be nonempty strings")
        origin = _origin_path(raw_origin, module_name)
        if origin is None:
            continue

        top_level = module_name.partition(".")[0]
        if _is_native_extension(origin) and _is_within(origin, ros_overlay):
            raise RosCompatibilityError(
                f"native extension {module_name} must not originate in the ROS overlay: {origin}"
            )
        if _is_native_extension(origin):
            in_target_environment = _is_within(origin, target_environment)
            in_base_stdlib = _is_within(origin, stdlib_paths[0])
            in_lib_dynload = _is_within(origin, stdlib_lib_dynload)
            if in_base_stdlib and not in_target_environment and not in_lib_dynload:
                raise RosCompatibilityError(
                    f"native extension {module_name} is outside exact stdlib lib-dynload: {origin}"
                )
            if not in_target_environment and not in_lib_dynload:
                raise RosCompatibilityError(
                    f"native extension {module_name} is outside the target interpreter: {origin}"
                )
        if top_level in ROS_PACKAGE_NAMES and not _is_within(origin, ros_overlay):
            raise RosCompatibilityError(
                f"ROS package {module_name} is outside the audited overlay: {origin}"
            )
        if top_level == "validator_msgs" and not _is_within(origin, catkin_python):
            raise RosCompatibilityError(
                f"project message {module_name} is outside active Catkin: {origin}"
            )
        if top_level == "validator_inference" and not (
            _is_within(origin, catkin_python)
            or any(_is_within(origin, root) for root in source_paths)
        ):
            raise RosCompatibilityError(
                f"project module {module_name} is outside active Catkin/source: {origin}"
            )
        if not any(_is_within(origin, root) for root in allowed_roots):
            raise RosCompatibilityError(
                f"module {module_name} is outside every allowed runtime root: {origin}"
            )


def _reserve_local_port() -> int:
    """localhost에서 kernel이 배정한 현재 미사용 TCP port 번호를 반환한다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _TimeoutTransport(xmlrpc.client.Transport):
    """한 XML-RPC request의 socket I/O에 남은 deadline timeout을 적용한다."""

    def __init__(self, timeout_seconds: float) -> None:
        """양수인 request socket timeout을 보관한다."""
        super().__init__()
        self._timeout_seconds = timeout_seconds

    def make_connection(self, host: str) -> Any:
        """기본 HTTP connection을 만들고 socket timeout을 덮어쓴다."""
        connection = super().make_connection(host)
        connection.timeout = self._timeout_seconds
        return connection


def _xmlrpc_proxy(master_uri: str, timeout_seconds: float) -> Any:
    """bounded transport를 사용하는 단일 ROS master XML-RPC proxy를 만든다."""
    return xmlrpc.client.ServerProxy(
        master_uri,
        transport=_TimeoutTransport(timeout_seconds),
        allow_none=True,
    )


def _close_xmlrpc_proxy(proxy: object) -> None:
    """ServerProxy transport를 원격 ``close`` method 호출 없이 local에서 닫는다."""
    private_close = getattr(proxy, "_ServerProxy__close", None)
    if callable(private_close):
        private_close()
        return
    close_descriptor = getattr(type(proxy), "close", None)
    if callable(close_descriptor):
        close_descriptor(proxy)


def _xmlrpc_request(
    master_uri: str,
    method_name: str,
    arguments: tuple[object, ...],
    deadline: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    proxy_factory: Callable[[str, float], Any] = _xmlrpc_proxy,
) -> object:
    """overall monotonic deadline의 남은 시간 안에서 XML-RPC method를 호출한다."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise RosCompatibilityError("ROS master XML-RPC deadline expired")
    proxy = proxy_factory(master_uri, remaining)
    try:
        method = getattr(proxy, method_name)
        response = method(*arguments)
    except (OSError, socket.timeout, xmlrpc.client.Error) as exc:
        raise RosCompatibilityError(
            f"ROS master XML-RPC {method_name} failed: {exc}"
        ) from exc
    finally:
        _close_xmlrpc_proxy(proxy)
    if monotonic() > deadline:
        raise RosCompatibilityError(
            f"ROS master XML-RPC {method_name} exceeded its deadline"
        )
    return response


def _wait_for_ros_master(
    master_uri: str,
    deadline: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    request: Callable[[str, str, tuple[object, ...], float], object] = _xmlrpc_request,
) -> bool:
    """absolute deadline까지 ROS master의 정상 getPid XML-RPC 응답을 기다린다."""
    parsed = urlsplit(master_uri)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        return False
    while monotonic() < deadline:
        try:
            response = request(
                master_uri,
                "getPid",
                ("/validator_ros_compatibility",),
                deadline,
            )
            if (
                isinstance(response, (list, tuple))
                and len(response) == 3
                and response[0] == 1
                and type(response[2]) is int
                and response[2] > 0
            ):
                return True
        except (OSError, RosCompatibilityError, xmlrpc.client.Error):
            pass
        remaining = deadline - monotonic()
        if remaining > 0:
            sleep(min(0.02, remaining))
    return False


def _set_ros_parameter(
    master_uri: str,
    name: str,
    value: object,
    deadline: float,
    *,
    request: Callable[[str, str, tuple[object, ...], float], object] = _xmlrpc_request,
) -> None:
    """overall deadline 안에서 ROS parameter server에 smoke 값을 설정한다."""
    response = request(
        master_uri,
        "setParam",
        ("/validator_ros_compatibility", name, value),
        deadline,
    )
    if (
        not isinstance(response, (list, tuple))
        or len(response) != 3
        or response[0] != 1
    ):
        raise RosCompatibilityError(f"cannot set ROS parameter {name}: {response!r}")


def _load_json_report(path: Path) -> object:
    """peer가 원자적으로 기록한 bounded UTF-8 JSON report를 읽는다."""
    try:
        metadata = Path(path).stat()
        if metadata.st_size > 4 * 1024 * 1024:
            raise RosCompatibilityError(f"ROS peer report is too large: {path}")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except RosCompatibilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RosCompatibilityError(f"cannot read ROS peer report: {path}") from exc


def _process_group_alive(process_group_id: int) -> bool:
    """signal 0으로 tracked process group에 살아 있는 member가 있는지 확인한다."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class RosCompatibilityDependencies:
    """subprocess·clock·ROS master I/O를 unit test에서 교체하는 의존성 묶음이다."""

    popen: Callable[..., Any] = subprocess.Popen
    killpg: Callable[[int, int], None] = os.killpg
    process_group_alive: Callable[[int], bool] = _process_group_alive
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    master_ready: Callable[[str, float], bool] = _wait_for_ros_master
    set_parameter: Callable[[str, str, object, float], None] = _set_ros_parameter
    load_report: Callable[[Path], object] = _load_json_report
    reserve_port: Callable[[], int] = _reserve_local_port


@dataclass(frozen=True)
class _RosBindings:
    """peer에서 지연 import한 rospy와 일곱 message class를 보관한다."""

    rospy: Any
    image: Any
    point_cloud: Any
    driving_feature: Any
    tf_message: Any
    transform_stamped: Any
    nav_path: Any
    pose_stamped: Any
    inference_metric: Any
    clock: Any


def _load_ros_bindings() -> _RosBindings:
    """controller import 경계 밖의 peer 실행 시점에만 ROS Python API를 적재한다."""
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from nav_msgs.msg import Path as NavPath
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, PointCloud2
        from tf2_msgs.msg import TFMessage
        from validator_msgs.msg import DrivingFeature, InferenceMetric
    except ImportError as exc:
        raise RosCompatibilityError(f"cannot import ROS smoke bindings: {exc}") from exc
    return _RosBindings(
        rospy=rospy,
        image=Image,
        point_cloud=PointCloud2,
        driving_feature=DrivingFeature,
        tf_message=TFMessage,
        transform_stamped=TransformStamped,
        nav_path=NavPath,
        pose_stamped=PoseStamped,
        inference_metric=InferenceMetric,
        clock=Clock,
    )


def _build_message_fixtures(
    role: str, bindings: Optional[_RosBindings] = None
) -> dict[str, object]:
    """system 또는 target sender를 식별하는 일곱 typed ROS message를 만든다."""
    if role not in {"system", "target"}:
        raise RosCompatibilityError(f"unsupported ROS peer role: {role}")
    selected = bindings or _load_ros_bindings()
    sequence = 101 if role == "system" else 202

    image_message = selected.image()
    image_message.header.frame_id = f"{role}:image"
    image_message.height = sequence
    image_message.width = sequence + 1
    image_message.encoding = "mono8"
    image_message.step = sequence + 1
    image_message.data = bytes(sequence * (sequence + 1))

    point_cloud_message = selected.point_cloud()
    point_cloud_message.header.frame_id = f"{role}:pointcloud"
    point_cloud_message.height = 1
    point_cloud_message.width = sequence
    point_cloud_message.point_step = 4
    point_cloud_message.row_step = sequence * 4
    point_cloud_message.data = bytes(sequence * 4)
    point_cloud_message.is_dense = True

    driving_feature_message = selected.driving_feature()
    driving_feature_message.model_id = role
    driving_feature_message.status_feature = [float(sequence)]
    driving_feature_message.source = f"ros-compat-{role}"

    transform = selected.transform_stamped()
    transform.child_frame_id = f"{role}:tf"
    transform.transform.translation.x = float(sequence)
    transform.transform.rotation.w = 1.0
    tf_message = selected.tf_message(transforms=[transform])

    nav_path_message = selected.nav_path()
    nav_path_message.header.frame_id = f"{role}:path"
    pose = selected.pose_stamped()
    pose.pose.position.x = float(sequence)
    pose.pose.orientation.w = 1.0
    nav_path_message.poses = [pose]

    metric_message = selected.inference_metric()
    metric_message.run_id = f"{role}:metric"
    metric_message.tick_index = sequence
    metric_message.model_id = role
    metric_message.outcome = "compatibility"

    clock_message = selected.clock()
    clock_message.clock.secs = sequence
    clock_message.clock.nsecs = sequence

    return {
        "sensor_msgs/Image": image_message,
        "sensor_msgs/PointCloud2": point_cloud_message,
        "validator_msgs/DrivingFeature": driving_feature_message,
        "tf2_msgs/TFMessage": tf_message,
        "nav_msgs/Path": nav_path_message,
        "validator_msgs/InferenceMetric": metric_message,
        "rosgraph_msgs/Clock": clock_message,
    }


def _extract_message_fields(message_type: str, message: object) -> dict[str, object]:
    """받은 typed fixture에서 wire 호환성을 증명하는 sentinel field만 추출한다."""
    try:
        if message_type == "sensor_msgs/Image":
            return {
                "frame_id": message.header.frame_id,
                "height": int(message.height),
                "width": int(message.width),
            }
        if message_type == "sensor_msgs/PointCloud2":
            return {
                "frame_id": message.header.frame_id,
                "width": int(message.width),
                "point_step": int(message.point_step),
            }
        if message_type == "validator_msgs/DrivingFeature":
            return {
                "model_id": message.model_id,
                "source": message.source,
                "status_feature": [float(value) for value in message.status_feature],
            }
        if message_type == "tf2_msgs/TFMessage":
            transform = message.transforms[0]
            return {
                "child_frame_id": transform.child_frame_id,
                "translation_x": float(transform.transform.translation.x),
            }
        if message_type == "nav_msgs/Path":
            return {
                "frame_id": message.header.frame_id,
                "pose_count": len(message.poses),
                "first_x": float(message.poses[0].pose.position.x),
            }
        if message_type == "validator_msgs/InferenceMetric":
            return {
                "run_id": message.run_id,
                "tick_index": int(message.tick_index),
                "model_id": message.model_id,
                "outcome": message.outcome,
            }
        if message_type == "rosgraph_msgs/Clock":
            return {
                "secs": int(message.clock.secs),
                "nsecs": int(message.clock.nsecs),
            }
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RosCompatibilityError(
            f"cannot extract ROS fixture fields for {message_type}"
        ) from exc
    raise RosCompatibilityError(f"unsupported ROS fixture type: {message_type}")


def _expected_received_fields(sender: str) -> dict[str, dict[str, object]]:
    """sender role별 일곱 typed fixture의 검증 가능한 field 값을 만든다."""
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


def _require_report_mapping(value: object, context: str) -> Mapping[str, object]:
    """peer report 값이 JSON object인지 확인한다."""
    if not isinstance(value, Mapping):
        raise RosCompatibilityError(f"{context} must be a JSON object")
    return value


def _validate_peer_report(
    payload: object,
    role: str,
    expected_python: Path,
    policy: RuntimePathPolicy,
) -> Mapping[str, object]:
    """한 peer JSON의 schema, interpreter, 통신 field와 origin을 엄격히 검증한다."""
    report = _require_report_mapping(payload, f"{role} report")
    if set(report) != _REPORT_KEYS:
        raise RosCompatibilityError(f"{role} report has an invalid schema")
    if report["schema"] != 1 or report["role"] != role:
        raise RosCompatibilityError(f"{role} report identity is invalid")
    if report["python_executable"] != str(expected_python):
        raise RosCompatibilityError(
            f"{role} peer used unexpected sys.executable: {report['python_executable']!r}"
        )
    pythonpath = report["pythonpath"]
    if not isinstance(pythonpath, list) or any(
        not isinstance(entry, str) for entry in pythonpath
    ):
        raise RosCompatibilityError(f"{role} report PYTHONPATH is invalid")
    if role == "target" and pythonpath != [
        str(policy.ros_overlay),
        str(policy.catkin_python_path),
    ]:
        raise RosCompatibilityError("target peer PYTHONPATH is not the exact allowlist")
    if report["pythonhome"] is not None:
        raise RosCompatibilityError(f"{role} peer inherited PYTHONHOME")
    if report["use_sim_time"] is not True:
        raise RosCompatibilityError(f"{role} peer did not observe /use_sim_time=true")
    clock_before = report["clock_before_ns"]
    clock_after = report["clock_after_ns"]
    if (
        type(clock_before) is not int
        or type(clock_after) is not int
        or clock_before < 0
        or clock_after <= clock_before
    ):
        raise RosCompatibilityError(f"{role} peer did not observe clock advancement")
    if report["sent_types"] != list(MESSAGE_TYPES):
        raise RosCompatibilityError(f"{role} peer did not publish every message type")
    sender = "target" if role == "system" else "system"
    if report["received"] != _expected_received_fields(sender):
        raise RosCompatibilityError(f"{role} peer received mismatched message fields")

    imports = report["inference_imports"]
    if not isinstance(imports, list) or any(
        not isinstance(name, str) for name in imports
    ):
        raise RosCompatibilityError(f"{role} peer import report is invalid")
    origins = _require_report_mapping(report["module_origins"], "module_origins")
    if role == "target":
        if imports != list(_REQUIRED_INFERENCE_IMPORTS):
            raise RosCompatibilityError("target peer did not import inference selectors")
        required_origins = {
            "rospy",
            "validator_msgs.msg._DrivingFeature",
            *_REQUIRED_INFERENCE_IMPORTS,
        }
        if not required_origins.issubset(origins):
            raise RosCompatibilityError("target peer omitted required module origins")
        validate_loaded_module_origins(origins, policy)
    return report


def _target_environment(
    inherited: Mapping[str, str], policy: RuntimePathPolicy, master_uri: str
) -> dict[str, str]:
    """target peer에 exact PYTHONPATH와 path-policy metadata만 전달한다."""
    blocked_prefixes = (
        "VIRTUAL_ENV",
        "CONDA_",
        "_CONDA",
        "__CONDA",
        "_OLD_VIRTUAL_",
    )
    child = {
        key: value
        for key, value in inherited.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}
        and not key.startswith(blocked_prefixes)
    }
    selected_bin = str(policy.target_environment / "bin")
    retained_path = [
        entry
        for entry in inherited.get("PATH", "").split(os.pathsep)
        if entry and entry != selected_bin
    ]
    child.update(
        {
            "ROS_MASTER_URI": master_uri,
            "PYTHONPATH": os.pathsep.join(
                (str(policy.ros_overlay), str(policy.catkin_python_path))
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "VIRTUAL_ENV": str(policy.target_environment),
            "PATH": os.pathsep.join((selected_bin, *retained_path)),
            "VALIDATOR_ROS_TARGET_PYTHON": str(policy.target_python),
            "VALIDATOR_ROS_TARGET_ENVIRONMENT": str(policy.target_environment),
            "VALIDATOR_ROS_OVERLAY": str(policy.ros_overlay),
            "VALIDATOR_ROS_CATKIN_PYTHON": str(policy.catkin_python_path),
            "VALIDATOR_ROS_STDLIB_PATHS": os.pathsep.join(
                str(path) for path in policy.target_stdlib_paths
            ),
            "VALIDATOR_ROS_SOURCE_PATHS": os.pathsep.join(
                str(path) for path in policy.project_source_paths
            ),
        }
    )
    return child


def _spawn_process(
    dependencies: RosCompatibilityDependencies,
    argv: list[str],
    environment: Mapping[str, str],
) -> Any:
    """stdin 없는 독립 process group으로 bounded smoke child를 시작한다."""
    return dependencies.popen(
        argv,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def _terminate_process_group(
    process: Any, dependencies: RosCompatibilityDependencies
) -> None:
    """leader 상태와 무관하게 tracked PGID를 TERM 후 bounded KILL로 종료한다."""
    process_group_id = int(process.pid)
    if not dependencies.process_group_alive(process_group_id):
        try:
            process.wait(timeout=0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        dependencies.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    term_deadline = dependencies.monotonic() + 2.0
    while dependencies.process_group_alive(process_group_id):
        try:
            process.poll()
        except (AttributeError, OSError):
            pass
        remaining = term_deadline - dependencies.monotonic()
        if remaining <= 0:
            break
        dependencies.sleep(min(0.02, remaining))
    if not dependencies.process_group_alive(process_group_id):
        try:
            process.wait(timeout=0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        dependencies.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    kill_deadline = dependencies.monotonic() + 2.0
    while dependencies.process_group_alive(process_group_id):
        try:
            process.poll()
        except (AttributeError, OSError):
            pass
        remaining = kill_deadline - dependencies.monotonic()
        if remaining <= 0:
            raise RosCompatibilityError(
                f"process group {process_group_id} survived SIGKILL"
            )
        dependencies.sleep(min(0.02, remaining))
    try:
        process.wait(timeout=0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_ros_compatibility_smoke(
    policy: RuntimePathPolicy,
    *,
    system_python: Path = Path("/usr/bin/python3"),
    timeout_seconds: float = 30.0,
    dependencies: Optional[RosCompatibilityDependencies] = None,
) -> RosSmokeReport:
    """임시 roscore와 두 peer를 bounded process group으로 실행하고 결과를 검증한다."""
    _validate_runtime_path_policy(policy)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise RosCompatibilityError("timeout_seconds must be positive and finite")
    selected_dependencies = dependencies or RosCompatibilityDependencies()
    if not isinstance(selected_dependencies, RosCompatibilityDependencies):
        raise RosCompatibilityError("dependencies have an invalid type")
    system_executable = Path(system_python)
    if not system_executable.is_absolute():
        raise RosCompatibilityError("system_python must be absolute")

    inherited = dict(os.environ)
    started: list[Any] = []
    try:
        with tempfile.TemporaryDirectory(prefix="validator-ros-compat-") as directory:
            result_root = Path(directory)
            system_result = result_root / "system.json"
            target_result = result_root / "target.json"
            port = selected_dependencies.reserve_port()
            master_uri = f"http://127.0.0.1:{port}/"
            deadline = selected_dependencies.monotonic() + float(timeout_seconds)
            system_environment = dict(inherited)
            system_environment["ROS_MASTER_URI"] = master_uri

            roscore = _spawn_process(
                selected_dependencies,
                ["roscore", "-p", str(port)],
                system_environment,
            )
            started.append(roscore)
            if not selected_dependencies.master_ready(master_uri, deadline):
                raise RosCompatibilityError("ROS compatibility roscore startup timed out")
            if roscore.poll() is not None:
                raise RosCompatibilityError("ROS compatibility roscore exited during startup")
            selected_dependencies.set_parameter(
                master_uri, "/use_sim_time", True, deadline
            )

            module_command = "validator_inference.ros_compatibility"
            system_peer = _spawn_process(
                selected_dependencies,
                [
                    str(system_executable),
                    "-m",
                    module_command,
                    "--role",
                    "system",
                    "--result",
                    str(system_result),
                ],
                system_environment,
            )
            started.append(system_peer)
            target_peer = _spawn_process(
                selected_dependencies,
                [
                    str(policy.target_python),
                    "-m",
                    module_command,
                    "--role",
                    "target",
                    "--result",
                    str(target_result),
                ],
                _target_environment(inherited, policy, master_uri),
            )
            started.append(target_peer)

            while system_peer.poll() is None or target_peer.poll() is None:
                if selected_dependencies.monotonic() >= deadline:
                    raise RosCompatibilityError("ROS compatibility peers timed out")
                selected_dependencies.sleep(0.02)
            if system_peer.poll() != 0 or target_peer.poll() != 0:
                raise RosCompatibilityError(
                    "ROS compatibility peer failed: "
                    f"system={system_peer.poll()} target={target_peer.poll()}"
                )

            system_payload = _validate_peer_report(
                selected_dependencies.load_report(system_result),
                "system",
                system_executable,
                policy,
            )
            target_payload = _validate_peer_report(
                selected_dependencies.load_report(target_result),
                "target",
                policy.target_python,
                policy,
            )
            target_origins = _require_report_mapping(
                target_payload["module_origins"], "module_origins"
            )
            return RosSmokeReport(
                system_python=system_executable,
                target_python=policy.target_python,
                target_pythonpath=(policy.ros_overlay, policy.catkin_python_path),
                message_types=MESSAGE_TYPES,
                system_received=tuple(system_payload["received"]),
                target_received=tuple(target_payload["received"]),
                use_sim_time=True,
                system_clock_advanced=True,
                target_clock_advanced=True,
                target_inference_imports=tuple(target_payload["inference_imports"]),
                target_module_origins=tuple(
                    sorted((str(name), str(origin)) for name, origin in target_origins.items())
                ),
            )
    except RosCompatibilityError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise RosCompatibilityError(
            f"ROS compatibility startup failure: {exc}"
        ) from exc
    finally:
        cleanup_errors: list[RosCompatibilityError] = []
        for process in reversed(started):
            try:
                _terminate_process_group(process, selected_dependencies)
            except RosCompatibilityError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise RosCompatibilityError(
                "; ".join(str(error) for error in cleanup_errors)
            ) from cleanup_errors[0]


def _required_environment_path(
    environ: Mapping[str, str], name: str
) -> Path:
    """peer path-policy의 필수 absolute path 환경 변수를 읽는다."""
    try:
        raw_value = environ[name]
    except KeyError as exc:
        raise RosCompatibilityError(f"missing peer environment variable: {name}") from exc
    if not isinstance(raw_value, str) or not raw_value:
        raise RosCompatibilityError(f"peer environment variable {name} is empty")
    path = Path(raw_value)
    if not path.is_absolute() or str(path) != raw_value:
        raise RosCompatibilityError(
            f"peer environment variable {name} must be an absolute path"
        )
    return path


def _environment_path_tuple(
    environ: Mapping[str, str], name: str, *, allow_empty: bool
) -> tuple[Path, ...]:
    """path-separator로 직렬화한 peer 허용 root 목록을 복원한다."""
    try:
        raw_value = environ[name]
    except KeyError as exc:
        raise RosCompatibilityError(f"missing peer environment variable: {name}") from exc
    if not isinstance(raw_value, str):
        raise RosCompatibilityError(f"peer environment variable {name} is invalid")
    if not raw_value:
        if allow_empty:
            return ()
        raise RosCompatibilityError(f"peer environment variable {name} is empty")
    values = tuple(Path(entry) for entry in raw_value.split(os.pathsep))
    if any(not path.is_absolute() or str(path) != entry for path, entry in zip(values, raw_value.split(os.pathsep))):
        raise RosCompatibilityError(
            f"peer environment variable {name} contains a non-absolute path"
        )
    if len(set(values)) != len(values):
        raise RosCompatibilityError(f"peer environment variable {name} has duplicates")
    return values


def _policy_from_environment(environ: Mapping[str, str]) -> RuntimePathPolicy:
    """controller가 target peer에 전달한 exact runtime path policy를 복원한다."""
    return RuntimePathPolicy(
        target_python=_required_environment_path(
            environ, "VALIDATOR_ROS_TARGET_PYTHON"
        ),
        target_environment=_required_environment_path(
            environ, "VALIDATOR_ROS_TARGET_ENVIRONMENT"
        ),
        ros_overlay=_required_environment_path(environ, "VALIDATOR_ROS_OVERLAY"),
        catkin_python_path=_required_environment_path(
            environ, "VALIDATOR_ROS_CATKIN_PYTHON"
        ),
        target_stdlib_paths=_environment_path_tuple(
            environ, "VALIDATOR_ROS_STDLIB_PATHS", allow_empty=False
        ),
        project_source_paths=_environment_path_tuple(
            environ, "VALIDATOR_ROS_SOURCE_PATHS", allow_empty=True
        ),
    )


def _message_classes(bindings: _RosBindings) -> dict[str, object]:
    """wire type 이름을 지연 import한 ROS message class에 대응시킨다."""
    return {
        "sensor_msgs/Image": bindings.image,
        "sensor_msgs/PointCloud2": bindings.point_cloud,
        "validator_msgs/DrivingFeature": bindings.driving_feature,
        "tf2_msgs/TFMessage": bindings.tf_message,
        "nav_msgs/Path": bindings.nav_path,
        "validator_msgs/InferenceMetric": bindings.inference_metric,
        "rosgraph_msgs/Clock": bindings.clock,
    }


def _topic_name(sender: str, message_type: str) -> str:
    """sender와 wire type으로 충돌 없는 절대 smoke topic 이름을 만든다."""
    slug = message_type.replace("/", "_").lower()
    return f"/validator_ros_compatibility/{sender}/{slug}"


class _PeerState:
    """callback thread가 받은 normalized message field를 안전하게 보관한다."""

    def __init__(self) -> None:
        """빈 수신 mapping과 callback lock을 초기화한다."""
        self._received: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()

    def record(self, message: object, message_type: str) -> None:
        """한 subscriber callback의 fixture field를 type별 최신 값으로 기록한다."""
        fields = _extract_message_fields(message_type, message)
        with self._lock:
            self._received[message_type] = fields

    def snapshot(self) -> dict[str, dict[str, object]]:
        """현재 수신 field mapping의 callback-safe 복사본을 반환한다."""
        with self._lock:
            return {
                message_type: dict(fields)
                for message_type, fields in self._received.items()
            }


def _loaded_module_origins() -> dict[str, object]:
    """현재 process의 모든 file·built-in·frozen module origin을 JSON 값으로 모은다."""
    origins: dict[str, object] = {}
    for name, module in sorted(sys.modules.items()):
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            specification = getattr(module, "__spec__", None)
            origin = getattr(specification, "origin", None)
        if origin is None:
            continue
        if isinstance(origin, Path):
            origin = str(origin)
        if isinstance(origin, str):
            origins[name] = origin
    return origins


def _atomic_write_report(path: Path, payload: Mapping[str, object]) -> None:
    """peer JSON을 sibling 임시 파일에서 fsync한 뒤 원자적으로 게시한다."""
    destination = Path(path)
    if not destination.is_absolute():
        raise RosCompatibilityError("peer result path must be absolute")
    temporary: Optional[Path] = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
        temporary = None
        directory_descriptor = os.open(
            str(destination.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise RosCompatibilityError(f"cannot write ROS peer report: {destination}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _publish_simulated_clock(bindings: _RosBindings, publisher: object, tick: int) -> None:
    """system peer가 wall-loop마다 단조 증가하는 ROS simulated clock을 발행한다."""
    message = bindings.clock()
    message.clock.secs = 10_000 + tick
    message.clock.nsecs = tick % 1_000_000_000
    publisher.publish(message)


def _peer_report_payload(
    role: str,
    clock_before_ns: int,
    clock_after_ns: int,
    received: Mapping[str, object],
    inference_imports: Sequence[str],
    module_origins: Mapping[str, object],
) -> dict[str, object]:
    """검증 가능한 peer runtime·통신 상태를 고정 schema JSON payload로 만든다."""
    return {
        "schema": 1,
        "role": role,
        "python_executable": sys.executable,
        "pythonpath": [
            entry
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry
        ],
        "pythonhome": os.environ.get("PYTHONHOME"),
        "use_sim_time": True,
        "clock_before_ns": int(clock_before_ns),
        "clock_after_ns": int(clock_after_ns),
        "sent_types": list(MESSAGE_TYPES),
        "received": dict(received),
        "inference_imports": list(inference_imports),
        "module_origins": dict(module_origins),
    }


def _run_peer(role: str, result_path: Path, timeout_seconds: float = 20.0) -> None:
    """한 ROS peer로 typed fixtures를 반복 발행하고 반대 방향 결과 JSON을 기록한다."""
    if role not in {"system", "target"}:
        raise RosCompatibilityError(f"unsupported ROS peer role: {role}")
    bindings = _load_ros_bindings()
    inference_imports: list[str] = []
    policy: Optional[RuntimePathPolicy] = None
    if role == "target":
        policy = _policy_from_environment(os.environ)
        if sys.executable != str(policy.target_python):
            raise RosCompatibilityError(
                f"target peer used unexpected sys.executable: {sys.executable}"
            )
        expected_pythonpath = os.pathsep.join(
            (str(policy.ros_overlay), str(policy.catkin_python_path))
        )
        if os.environ.get("PYTHONPATH") != expected_pythonpath:
            raise RosCompatibilityError("target peer PYTHONPATH is not the exact allowlist")
        if "PYTHONHOME" in os.environ:
            raise RosCompatibilityError("target peer inherited PYTHONHOME")
        for module_name in _REQUIRED_INFERENCE_IMPORTS:
            importlib.import_module(module_name)
            inference_imports.append(module_name)

    bindings.rospy.init_node(
        f"validator_ros_compatibility_{role}",
        anonymous=True,
        disable_signals=True,
    )
    use_sim_time = bindings.rospy.get_param("/use_sim_time", False)
    if use_sim_time is not True:
        raise RosCompatibilityError("ROS peer requires /use_sim_time=true")
    clock_before_ns = int(bindings.rospy.Time.now().to_nsec())

    sender_messages = _build_message_fixtures(role, bindings)
    message_classes = _message_classes(bindings)
    peer_role = "target" if role == "system" else "system"
    state = _PeerState()
    subscribers = [
        bindings.rospy.Subscriber(
            _topic_name(peer_role, message_type),
            message_classes[message_type],
            state.record,
            callback_args=message_type,
            queue_size=1,
        )
        for message_type in MESSAGE_TYPES
    ]
    publishers = {
        message_type: bindings.rospy.Publisher(
            _topic_name(role, message_type),
            message_classes[message_type],
            queue_size=1,
            latch=True,
        )
        for message_type in MESSAGE_TYPES
    }
    clock_publisher = None
    if role == "system":
        clock_publisher = bindings.rospy.Publisher(
            "/clock", bindings.clock, queue_size=1, latch=True
        )

    started_at = time.monotonic()
    completion_started: Optional[float] = None
    tick = 0
    received: dict[str, dict[str, object]] = {}
    clock_after_ns = clock_before_ns
    while time.monotonic() - started_at < timeout_seconds:
        if bindings.rospy.is_shutdown():
            raise RosCompatibilityError("ROS peer shut down before smoke completion")
        tick += 1
        if clock_publisher is not None:
            _publish_simulated_clock(bindings, clock_publisher, tick)
        for message_type, message in sender_messages.items():
            publishers[message_type].publish(message)
        received = state.snapshot()
        clock_after_ns = int(bindings.rospy.Time.now().to_nsec())
        if (
            received == _expected_received_fields(peer_role)
            and clock_after_ns > clock_before_ns
        ):
            if completion_started is None:
                completion_started = time.monotonic()
            elif time.monotonic() - completion_started >= 0.5:
                break
        time.sleep(0.02)
    else:
        raise RosCompatibilityError(
            f"{role} peer timed out waiting for messages or simulated clock"
        )

    del subscribers
    module_origins = _loaded_module_origins()
    if role == "target":
        if policy is None:
            raise RosCompatibilityError("target peer path policy is missing")
        validate_loaded_module_origins(module_origins, policy)
    payload = _peer_report_payload(
        role,
        clock_before_ns,
        clock_after_ns,
        received,
        inference_imports,
        module_origins,
    )
    _atomic_write_report(result_path, payload)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    peer_runner: Optional[Callable[[str, Path], None]] = None,
) -> int:
    """system|target peer CLI를 실행하고 예상된 계약 오류를 한 줄로 보고한다."""
    parser = argparse.ArgumentParser(
        prog="python -m validator_inference.ros_compatibility",
        allow_abbrev=False,
    )
    parser.add_argument("--role", choices=("system", "target"), required=True)
    parser.add_argument("--result", required=True)
    arguments = parser.parse_args(argv)
    result_path = Path(arguments.result)
    selected_runner = _run_peer if peer_runner is None else peer_runner
    try:
        if not result_path.is_absolute():
            raise RosCompatibilityError("peer result path must be absolute")
        selected_runner(arguments.role, result_path)
        return 0
    except RosCompatibilityError as exc:
        print(f"ros compatibility error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
