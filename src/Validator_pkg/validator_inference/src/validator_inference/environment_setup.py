"""
모델 환경 host preflight와 candidate/순수 Python ROS overlay lifecycle을 관리한다.
입력: project/model, 검증된 environment paths, build ID와 ROS Python source tree.
출력(pub): 읽기 전용 host 점검 결과, candidate와 overlay lifecycle 결과를 반환한다.
구독(sub): 없음.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from validator_inference.model_environment import (
    ENVIRONMENT_STATE_FILENAME,
    ENVIRONMENT_STATE_SCHEMA,
    EXPECTED_PYTHON_PLATFORM,
    EXPECTED_UV_VERSION,
    ROS_OVERLAY_MANIFEST_FILENAME,
    SUPPORTED_MODEL_IDS,
    AssetFingerprint,
    EnvironmentState,
    EnvironmentStateError,
    EnvironmentConfigError,
    EnvironmentManifest,
    EnvironmentPaths,
    ModelEnvironmentDefinition,
    build_model_process_environment,
    build_uv_sync_argv,
    build_uv_venv_argv,
    environment_paths,
    inspect_active_environment,
    inspect_active_environment_for_verification,
    load_environment_manifest,
    read_environment_state,
    select_model_environment,
    sha256_file,
    write_environment_state,
)
if TYPE_CHECKING:
    from validator_inference.profile_store import SelectedProfile
else:
    SelectedProfile = Any


PURE_ROS_PACKAGES = (
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
ROS_OVERLAY_MANIFEST_SCHEMA = 1
_BUILD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = {"schema", "packages", "files"}
_FILE_KEYS = {"source", "destination", "sha256"}
_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")
_LIFECYCLE_LOCK_FILENAME = ".lifecycle.lock"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_RECORDS = 4096
_MAX_PATH_STRING_LENGTH = 4096
_MAX_OVERLAY_FILE_BYTES = 64 * 1024 * 1024
_READ_BLOCK_SIZE = 1024 * 1024
_HOST_PREFLIGHT_ACTION = "host_preflight"
_HOST_COMMAND_TIMEOUT_SECONDS = 10.0
_SUPPORTED_CUDA_POLICIES = {"required", "recommended", "extension_required"}
_UV_VERSION_OUTPUT = re.compile(r"uv ([^\s]+)(?:\s+.*)?\Z")
_NVCC_11_8_OUTPUT = re.compile(r"\brelease 11\.8(?:,|\s|\Z)")
_COMPILER_VERSION_OUTPUT = re.compile(
    r"([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?\Z"
)
_VERSIONED_PYTHON_EXECUTABLE = re.compile(r"python([0-9]+)\.([0-9]+)\Z")
_SPARSE_COMPILER_MIN_MAJOR = 6
_SPARSE_COMPILER_MAX_MAJOR = 11
_ASSETS_ACTION = "assets"
_INTERFUSER_CHECKPOINT_PATH = Path(
    "models/interfuser/official_carla/interfuser.pth.tar"
)
_INTERFUSER_BACKBONE_PATH = Path(
    "models/shared/torch-cache/hub/checkpoints/resnet50d_ra2-464e36ba.pth"
)
_INTERFUSER_USER_CACHE = (
    Path.home()
    / ".cache"
    / "torch"
    / "hub"
    / "checkpoints"
    / "resnet50d_ra2-464e36ba.pth"
)
_SETUP_COMMAND_TIMEOUT_SECONDS = 60.0
_MODEL_LOAD_TIMEOUT_SECONDS = 300.0
_SETUP_PREFLIGHT_IDS = (
    "host",
    "catkin_messages",
    "runtime_assets",
    "python_version",
    "required_imports",
    "cuda_policy",
    "ros_overlay",
    "ros_smoke",
    "model_load",
    "package_freeze",
)
_DIFFUSION_DERIVATIVE = Path(
    "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
)
_SPARSE_OPS = Path("navsim/agents/sparsedrive/ops")
_SPARSE_EXTENSION_MODULES = (
    "deformable_aggregation_ext",
    "deformable_aggregation_with_depth_ext",
)


def _read_host_os_release() -> Mapping[str, str]:
    """표준 os-release를 shell 실행 없이 key/value mapping으로 읽는다."""
    values: dict[str, str] = {}
    payload = Path("/etc/os-release").read_text(encoding="utf-8")
    for line in payload.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value, comments=False, posix=True)
        if len(parsed) == 1:
            values[key] = parsed[0]
    return values


def _default_gpu_available(
    driver_gpu_root: Path = Path("/proc/driver/nvidia/gpus"),
    device_root: Path = Path("/dev"),
) -> bool:
    """driver metadata와 접근 가능한 NVIDIA device node를 함께 확인한다."""
    try:
        if not driver_gpu_root.is_dir() or not any(driver_gpu_root.iterdir()):
            return False
        control = device_root / "nvidiactl"
        if not control.exists() or not os.access(control, os.R_OK | os.W_OK):
            return False
        devices = (
            entry
            for entry in device_root.iterdir()
            if re.fullmatch(r"nvidia[0-9]+", entry.name)
        )
        return any(
            device.exists() and os.access(device, os.R_OK | os.W_OK)
            for device in devices
        )
    except OSError:
        return False


def _load_profile_for_preflight(path: Path, model_id: str) -> "SelectedProfile":
    """profile dependency를 preflight 호출 경계에서만 import해 runtime closure를 보존한다."""
    from validator_inference.profile_store import load_selected_profile

    return load_selected_profile(path, model_id)


def _fetch_assets_for_setup(
    profile: dict[str, Any], *, model_id: str, project_root: Path
) -> Any:
    """asset fetcher를 setup mutation phase에서만 지연 import해 실행한다."""
    from validator_inference.assets import fetch_runtime_assets

    return fetch_runtime_assets(
        profile, model_id=model_id, project_root=project_root
    )


def _verify_assets_for_setup(
    profile: dict[str, Any], *, project_root: Path
) -> Mapping[str, Any]:
    """asset/source deep verifier를 setup 호출 경계에서만 지연 import한다."""
    from validator_inference.assets import verify_runtime_assets

    return verify_runtime_assets(profile, project_root=project_root)


def _new_environment_build_id() -> str:
    """UTC second와 random suffix를 결합한 안전한 candidate ID를 만든다."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(6)}"


def _utc_second_timestamp() -> str:
    """EnvironmentState가 요구하는 UTC second timestamp를 반환한다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EnvironmentSetupError(RuntimeError):
    """candidate 또는 overlay action 실패 원인과 단계를 보관한다."""

    def __init__(self, action: str, message: str) -> None:
        """실패 action과 사람이 조치할 수 있는 원인을 초기화한다."""
        self.action = action
        super().__init__(f"{action}: {message}")


@dataclass(frozen=True)
class RosOverlayFile:
    """복사한 ROS Python 파일의 source, 상대 destination과 hash를 보관한다."""

    source_path: Path
    destination_path: Path
    sha256: str


@dataclass(frozen=True)
class RosOverlayManifest:
    """한 candidate의 허용 ROS package와 파일 provenance를 보관한다."""

    schema: int
    packages: tuple[str, ...]
    files: tuple[RosOverlayFile, ...]
    manifest_path: Path


@dataclass(frozen=True)
class EnvironmentPreflightDependencies:
    """host probe와 권위 loader를 테스트에서 대체할 수 있는 읽기 전용 bundle이다."""

    platform_system: Callable[[], str] = field(
        default_factory=lambda: platform.system
    )
    platform_machine: Callable[[], str] = field(
        default_factory=lambda: platform.machine
    )
    read_os_release: Callable[[], Mapping[str, str]] = field(
        default_factory=lambda: _read_host_os_release
    )
    getenv: Callable[[str], Optional[str]] = field(
        default_factory=lambda: os.environ.get
    )
    which: Callable[[str], Optional[str]] = field(
        default_factory=lambda: shutil.which
    )
    gpu_available: Callable[[], bool] = field(
        default_factory=lambda: _default_gpu_available
    )
    load_manifest: Callable[[Path], EnvironmentManifest] = field(
        default_factory=lambda: load_environment_manifest
    )
    select_definition: Callable[
        [EnvironmentManifest, str], ModelEnvironmentDefinition
    ] = field(default_factory=lambda: select_model_environment)
    load_profile: Callable[[Path, str], SelectedProfile] = field(
        default_factory=lambda: _load_profile_for_preflight
    )
    ros_root: Path = Path("/opt/ros/noetic")
    ros_python_root: Path = Path("/opt/ros/noetic/lib/python3/dist-packages")
    cuda_root: Path = Path("/usr/local/cuda-11.8")
    nvcc_path: Path = Path("/usr/local/cuda-11.8/bin/nvcc")
    gcc_path: Path = Path("/usr/bin/gcc")
    gxx_path: Path = Path("/usr/bin/g++")


@dataclass(frozen=True)
class EnvironmentPreflight:
    """검증된 model definition, host executable과 경고를 보관한다."""

    root: Path
    package_root: Path
    manifest: EnvironmentManifest
    definition: ModelEnvironmentDefinition
    selected_profile: SelectedProfile
    paths: EnvironmentPaths
    uv_path: Path
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentSetupDependencies:
    """setup의 host, asset, 시간과 ROS smoke 경계를 대체하는 dependency bundle이다."""

    preflight_dependencies: EnvironmentPreflightDependencies = field(
        default_factory=EnvironmentPreflightDependencies
    )
    asset_fetcher: Callable[..., Any] = field(
        default_factory=lambda: _fetch_assets_for_setup
    )
    asset_verifier: Callable[..., Mapping[str, Any]] = field(
        default_factory=lambda: _verify_assets_for_setup
    )
    build_id: Callable[[], str] = field(
        default_factory=lambda: _new_environment_build_id
    )
    utc_now: Callable[[], str] = field(
        default_factory=lambda: _utc_second_timestamp
    )
    ros_smoke: Optional[Callable[[Path, Mapping[str, str]], None]] = None


@dataclass(frozen=True)
class EnvironmentOperationResult:
    """setup이 선택한 environment, state, 경고와 변경 여부를 보관한다."""

    environment_dir: Path
    state: EnvironmentState
    warnings: Tuple[str, ...]
    changed: bool


def _raise(action: str, message: str) -> None:
    """일관된 action prefix를 가진 setup domain 오류를 발생시킨다."""
    raise EnvironmentSetupError(action, message)


def _validate_environment_paths(paths: EnvironmentPaths, action: str) -> None:
    """EnvironmentPaths가 project/.venvs/MODEL에서 안전하게 유도됐는지 검사한다."""
    model_root = Path(paths.model_root)
    if (
        not model_root.is_absolute()
        or model_root.name not in SUPPORTED_MODEL_IDS
        or model_root.parent.name != ".venvs"
        or Path(paths.builds) != model_root / "builds"
        or Path(paths.current) != model_root / "current"
    ):
        _raise(action, "environment path layout is invalid")


def _validate_component(value: str, context: str, action: str) -> None:
    """build 또는 package 이름이 안전한 단일 filesystem component인지 검사한다."""
    pattern = _BUILD_ID if context == "build_id" else _PACKAGE_NAME
    if not isinstance(value, str) or not pattern.fullmatch(value) or value in {".", ".."}:
        _raise(action, f"{context} must be a safe single component")


def _canonical_absolute_posix_path(value: Any, context: str, action: str) -> Path:
    """입력 문자열을 정규화하지 않고 canonical absolute POSIX path인지 검사한다."""
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise EnvironmentSetupError(
            action, f"{context} must be a canonical absolute POSIX path"
        ) from exc
    if isinstance(raw, str) and len(raw) > _MAX_PATH_STRING_LENGTH:
        _raise(action, f"{context} path length limit exceeded")
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        _raise(action, f"{context} must be a canonical absolute POSIX path")
    pure = PurePosixPath(raw)
    if (
        not pure.is_absolute()
        or raw.startswith("//")
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _raise(action, f"{context} must be a canonical absolute POSIX path")
    return Path(*pure.parts)


@contextmanager
def _open_absolute_nofollow(
    path: Path,
    final_flags: int,
    action: str,
    context: str,
) -> Iterator[int]:
    """절대 path를 root dir_fd부터 component별 O_NOFOLLOW로 열어 fd를 반환한다."""
    canonical = _canonical_absolute_posix_path(path, context, action)
    components = canonical.parts[1:]
    if not components:
        _raise(action, f"{context} must not be the filesystem root")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        _raise(action, f"{context} requires O_NOFOLLOW support")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | close_on_exec
    parent_descriptor: Optional[int] = None
    final_descriptor: Optional[int] = None
    try:
        parent_descriptor = os.open("/", directory_flags)
        for component in components[:-1]:
            child_descriptor = os.open(
                component,
                directory_flags | no_follow,
                dir_fd=parent_descriptor,
            )
            previous_descriptor = parent_descriptor
            parent_descriptor = child_descriptor
            os.close(previous_descriptor)
        final_descriptor = os.open(
            components[-1],
            final_flags | no_follow | close_on_exec,
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"{context} is missing: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            message = f"{context} path contains a symlink: {path}"
        elif exc.errno == errno.ENOTDIR:
            message = f"{context} path component is not a directory or is a symlink: {path}"
        else:
            message = f"cannot securely open {context}: {path}"
        raise EnvironmentSetupError(action, message) from exc
    finally:
        if final_descriptor is None and parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
    try:
        yield final_descriptor
    finally:
        try:
            os.close(final_descriptor)
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)


@contextmanager
def _open_regular_file(
    path: Path, action: str, context: str
) -> Iterator[tuple[int, os.stat_result]]:
    """special file에서 block하지 않고 final regular file fd와 metadata를 반환한다."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0)
    )
    with _open_absolute_nofollow(path, flags, action, context) as descriptor:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot inspect opened {context}: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            _raise(action, f"{context} is not a regular file: {path}")
        yield descriptor, metadata


def _read_bounded_regular_file(
    path: Path,
    maximum_bytes: int,
    action: str,
    context: str,
) -> bytes:
    """같은 regular fd에서 maximum_bytes 이하의 bytes만 읽는다."""
    with _open_regular_file(path, action, context) as (descriptor, metadata):
        if metadata.st_size > maximum_bytes:
            _raise(action, f"{context} size limit exceeded")
        encoded = bytearray()
        try:
            while True:
                remaining = maximum_bytes + 1 - len(encoded)
                block = os.read(descriptor, min(_READ_BLOCK_SIZE, remaining))
                if not block:
                    break
                encoded.extend(block)
                if len(encoded) > maximum_bytes:
                    _raise(action, f"{context} size limit exceeded")
        except EnvironmentSetupError:
            raise
        except MemoryError as exc:
            raise EnvironmentSetupError(
                action, f"{context} exceeds safe allocation limits"
            ) from exc
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot read {context}: {path}") from exc
    return bytes(encoded)


def _sha256_open_regular_file(path: Path, action: str, context: str) -> str:
    """같은 regular fd를 bounded streaming read해 SHA-256을 계산한다."""
    with _open_regular_file(path, action, context) as (descriptor, metadata):
        if metadata.st_size > _MAX_OVERLAY_FILE_BYTES:
            _raise(action, f"{context} size limit exceeded")
        digest = hashlib.sha256()
        total = 0
        try:
            while True:
                remaining = _MAX_OVERLAY_FILE_BYTES + 1 - total
                block = os.read(descriptor, min(_READ_BLOCK_SIZE, remaining))
                if not block:
                    break
                total += len(block)
                if total > _MAX_OVERLAY_FILE_BYTES:
                    _raise(action, f"{context} size limit exceeded")
                digest.update(block)
        except EnvironmentSetupError:
            raise
        except (MemoryError, OSError) as exc:
            raise EnvironmentSetupError(action, f"cannot hash {context}: {path}") from exc
    return digest.hexdigest()


def _fsync_directory(path: Path, action: str, context: str) -> None:
    """secure하게 연 directory fd를 fsync해 name 변경의 내구성을 보장한다."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with _open_absolute_nofollow(path, flags, action, context) as descriptor:
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                _raise(action, f"{context} is not a directory: {path}")
            os.fsync(descriptor)
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot fsync {context}: {path}"
            ) from exc


def _reject_existing_symlink_chain(path: Path, action: str, context: str) -> None:
    """존재하는 path와 상위 component 중 symlink가 있으면 따라가기 전에 거부한다."""
    value = Path(path)
    candidates = [value, *value.parents]
    for candidate in reversed(candidates):
        if candidate == Path(candidate.anchor):
            continue
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot inspect {context} path: {candidate}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _raise(action, f"{context} path contains a symlink: {candidate}")


def _require_real_directory(path: Path, action: str, context: str) -> None:
    """path가 symlink가 아닌 기존 directory인지 lstat 기준으로 검사한다."""
    _reject_existing_symlink_chain(path, action, context)
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"{context} directory is missing: {path}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect {context}: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(action, f"{context} must be a directory: {path}")


def _ensure_real_directory(path: Path, action: str, context: str) -> None:
    """새 directory를 만들거나 기존 실제 directory만 재사용한다."""
    _reject_existing_symlink_chain(path, action, context)
    try:
        Path(path).mkdir()
    except FileExistsError:
        pass
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create {context}: {path}") from exc
    _require_real_directory(path, action, context)


def _require_preflight_regular_file(path: Path, context: str) -> None:
    """preflight 입력이 symlink가 아닌 기존 regular file인지 확인한다."""
    with _open_regular_file(path, _HOST_PREFLIGHT_ACTION, context):
        pass


def _require_executable_file(path: Any, context: str) -> Path:
    """lookup한 command가 canonical absolute executable regular file인지 확인한다."""
    candidate = _canonical_absolute_posix_path(
        path, context, _HOST_PREFLIGHT_ACTION
    )
    try:
        metadata = candidate.stat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"{context} executable is missing: {candidate}"
        ) from exc
    except OSError as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"cannot inspect {context}: {candidate}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(candidate, os.X_OK):
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"{context} must be an executable regular file: {candidate}",
        )
    return candidate


def _lookup_preflight_executable(
    dependencies: EnvironmentPreflightDependencies,
    command: str,
    context: str,
) -> Path:
    """PATH lookup 실패를 deterministic host_preflight 오류로 바꾼다."""
    try:
        value = dependencies.which(command)
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"cannot locate {context} executable"
        ) from exc
    if value is None:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"{context} executable was not found on PATH",
        )
    return _require_executable_file(value, context)


def _run_preflight_probe(
    runner: Callable[..., Any], argv: Sequence[str], context: str
) -> str:
    """bounded read-only command를 shell 없이 실행하고 stdout을 반환한다."""
    command = [str(part) for part in argv]
    try:
        result = runner(
            command,
            check=True,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_HOST_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"{context} failed"
        ) from exc
    if getattr(result, "returncode", 0) != 0:
        _raise(_HOST_PREFLIGHT_ACTION, f"{context} failed")
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        _raise(_HOST_PREFLIGHT_ACTION, f"{context} returned invalid stdout")
    return stdout.strip()


def _validate_host_platform(
    dependencies: EnvironmentPreflightDependencies,
) -> None:
    """지원 host를 Linux Ubuntu 20.04 x86_64로 제한한다."""
    try:
        system = dependencies.platform_system()
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot determine host operating system"
        ) from exc
    if system != "Linux":
        _raise(_HOST_PREFLIGHT_ACTION, "host must be Linux")
    try:
        release = dependencies.read_os_release()
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot read host os-release"
        ) from exc
    if not isinstance(release, Mapping):
        _raise(_HOST_PREFLIGHT_ACTION, "host os-release must be a mapping")
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "20.04":
        _raise(_HOST_PREFLIGHT_ACTION, "host must be Ubuntu 20.04")
    try:
        machine = dependencies.platform_machine()
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot determine host machine architecture"
        ) from exc
    if machine != "x86_64":
        _raise(_HOST_PREFLIGHT_ACTION, "host machine must be x86_64")


def _validate_uv(
    manifest: EnvironmentManifest,
    dependencies: EnvironmentPreflightDependencies,
    runner: Callable[..., Any],
) -> Path:
    """PATH의 uv executable과 exact manifest version을 검증한다."""
    uv_path = _lookup_preflight_executable(dependencies, "uv", "uv")
    output = _run_preflight_probe(runner, (str(uv_path), "--version"), "uv version probe")
    match = _UV_VERSION_OUTPUT.fullmatch(output)
    if (
        manifest.uv_version != EXPECTED_UV_VERSION
        or match is None
        or match.group(1) != EXPECTED_UV_VERSION
    ):
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"uv {EXPECTED_UV_VERSION} is required; got {output or '<empty>'}",
        )
    return uv_path


def _validate_ros_noetic(
    dependencies: EnvironmentPreflightDependencies,
) -> None:
    """sourced Noetic shell, setup, tools와 exact overlay package dirs를 확인한다."""
    try:
        ros_distro = dependencies.getenv("ROS_DISTRO")
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot read ROS_DISTRO"
        ) from exc
    if ros_distro != "noetic":
        _raise(
            _HOST_PREFLIGHT_ACTION,
            "ROS_DISTRO=noetic is required; source /opt/ros/noetic/setup.bash",
        )
    ros_root = _canonical_absolute_posix_path(
        dependencies.ros_root, "ROS Noetic root", _HOST_PREFLIGHT_ACTION
    )
    ros_python_root = _canonical_absolute_posix_path(
        dependencies.ros_python_root,
        "ROS Noetic Python root",
        _HOST_PREFLIGHT_ACTION,
    )
    _require_real_directory(ros_root, _HOST_PREFLIGHT_ACTION, "ROS Noetic root")
    _require_preflight_regular_file(
        ros_root / "setup.bash", "ROS Noetic setup.bash"
    )
    _require_real_directory(
        ros_python_root, _HOST_PREFLIGHT_ACTION, "ROS Noetic Python root"
    )
    for package in PURE_ROS_PACKAGES:
        _require_real_directory(
            ros_python_root / package,
            _HOST_PREFLIGHT_ACTION,
            f"ROS Python package {package}",
        )
    _lookup_preflight_executable(dependencies, "catkin_make", "catkin_make")
    _lookup_preflight_executable(dependencies, "roscore", "roscore")


def _probe_gpu(dependencies: EnvironmentPreflightDependencies) -> bool:
    """injected GPU probe의 실패와 잘못된 결과를 fail closed 한다."""
    try:
        available = dependencies.gpu_available()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot determine CUDA GPU availability"
        ) from exc
    if type(available) is not bool:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            "CUDA GPU probe must return a boolean",
        )
    return available


def _parse_compiler_version(output: str, context: str) -> tuple[int, int, int]:
    """GCC machine-readable version을 비교 가능한 세 정수로 변환한다."""
    match = _COMPILER_VERSION_OUTPUT.fullmatch(output)
    if match is None:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"{context} returned an invalid compiler version: {output or '<empty>'}",
        )
    return tuple(int(component or 0) for component in match.groups())  # type: ignore[return-value]


def _validate_sparse_host(
    dependencies: EnvironmentPreflightDependencies,
    runner: Callable[..., Any],
) -> None:
    """SparseDrive extension에 필요한 exact Toolkit, nvcc와 compiler를 검증한다."""
    cuda_root = _canonical_absolute_posix_path(
        dependencies.cuda_root, "CUDA Toolkit root", _HOST_PREFLIGHT_ACTION
    )
    try:
        cuda_home = dependencies.getenv("CUDA_HOME")
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot read CUDA_HOME"
        ) from exc
    if cuda_home != str(cuda_root):
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"CUDA_HOME must be exactly {cuda_root}",
        )
    _require_real_directory(
        cuda_root, _HOST_PREFLIGHT_ACTION, "CUDA Toolkit 11.8"
    )
    nvcc_path = _canonical_absolute_posix_path(
        dependencies.nvcc_path, "nvcc", _HOST_PREFLIGHT_ACTION
    )
    if nvcc_path != cuda_root / "bin" / "nvcc":
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"nvcc must be exactly {cuda_root / 'bin' / 'nvcc'}",
        )
    nvcc_path = _require_executable_file(nvcc_path, "nvcc")
    nvcc_output = _run_preflight_probe(
        runner, (str(nvcc_path), "--version"), "nvcc version probe"
    )
    if _NVCC_11_8_OUTPUT.search(nvcc_output) is None:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"nvcc release 11.8 is required; got {nvcc_output or '<empty>'}",
        )

    gcc_path = _require_executable_file(
        dependencies.gcc_path, "SparseDrive /usr/bin/gcc compiler"
    )
    gxx_path = _require_executable_file(
        dependencies.gxx_path, "SparseDrive /usr/bin/g++ compiler"
    )
    version_argv = ("-dumpfullversion", "-dumpversion")
    gcc_output = _run_preflight_probe(
        runner, (str(gcc_path), *version_argv), "gcc version probe"
    )
    gxx_output = _run_preflight_probe(
        runner, (str(gxx_path), *version_argv), "g++ version probe"
    )
    gcc_version = _parse_compiler_version(gcc_output, "gcc")
    gxx_version = _parse_compiler_version(gxx_output, "g++")
    if (
        gcc_version != gxx_version
        or not _SPARSE_COMPILER_MIN_MAJOR
        <= gcc_version[0]
        <= _SPARSE_COMPILER_MAX_MAJOR
    ):
        _raise(
            _HOST_PREFLIGHT_ACTION,
            "SparseDrive requires compatible matching GCC/G++ versions "
            f"{_SPARSE_COMPILER_MIN_MAJOR} through {_SPARSE_COMPILER_MAX_MAJOR}; "
            f"got gcc {gcc_output} and g++ {gxx_output}",
        )


def preflight_environment(
    project_root: Path,
    model_id: str,
    runner: Callable[..., Any] = subprocess.run,
    *,
    dependencies: Optional[EnvironmentPreflightDependencies] = None,
) -> EnvironmentPreflight:
    """setup mutation 전에 project, host, ROS와 model prerequisite를 검증한다."""
    selected_dependencies = dependencies or EnvironmentPreflightDependencies()
    if not isinstance(selected_dependencies, EnvironmentPreflightDependencies):
        _raise(
            _HOST_PREFLIGHT_ACTION,
            "dependencies must be EnvironmentPreflightDependencies",
        )
    root = _canonical_absolute_posix_path(
        project_root, "project root", _HOST_PREFLIGHT_ACTION
    )
    _require_real_directory(root, _HOST_PREFLIGHT_ACTION, "project root")
    package_root = root / "src" / "Validator_pkg" / "validator_inference"
    profile_path = root / "src" / "Validator_pkg" / "config" / "profile.json"
    manifest_path = package_root / "config" / "model_environments.json"
    _require_real_directory(
        package_root, _HOST_PREFLIGHT_ACTION, "validator_inference package root"
    )
    _require_preflight_regular_file(manifest_path, "environment manifest")
    _require_preflight_regular_file(profile_path, "model profile")
    if model_id not in SUPPORTED_MODEL_IDS:
        _raise(_HOST_PREFLIGHT_ACTION, f"unsupported model_id: {model_id}")

    try:
        from validator_inference.profile_store import (
            ProfileError as ProfileLoadError,
            SelectedProfile as SelectedProfileType,
        )
    except ImportError as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, "cannot import model profile loader"
        ) from exc

    try:
        manifest = selected_dependencies.load_manifest(manifest_path)
        definition = selected_dependencies.select_definition(manifest, model_id)
        selected_profile = selected_dependencies.load_profile(profile_path, model_id)
        paths = environment_paths(root, model_id)
    except EnvironmentSetupError:
        raise
    except EnvironmentConfigError as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"environment manifest is invalid: {exc}"
        ) from exc
    except ProfileLoadError as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION, f"model profile is invalid: {exc}"
        ) from exc
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EnvironmentSetupError(
            _HOST_PREFLIGHT_ACTION,
            f"cannot load model environment/profile for {model_id}: {exc}",
        ) from exc
    if not isinstance(manifest, EnvironmentManifest):
        _raise(_HOST_PREFLIGHT_ACTION, "manifest loader returned an invalid value")
    if not isinstance(definition, ModelEnvironmentDefinition):
        _raise(_HOST_PREFLIGHT_ACTION, "environment selector returned an invalid value")
    if not isinstance(selected_profile, SelectedProfileType):
        _raise(_HOST_PREFLIGHT_ACTION, "profile loader returned an invalid value")
    if definition.cuda_policy not in _SUPPORTED_CUDA_POLICIES:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"unsupported cuda_policy: {definition.cuda_policy}",
        )
    source = selected_profile.data.get("source")
    profile_commit = source.get("commit") if isinstance(source, dict) else None
    if profile_commit != definition.source_commit:
        _raise(
            _HOST_PREFLIGHT_ACTION,
            f"profile source commit does not match environment definition for {model_id}",
        )

    _validate_host_platform(selected_dependencies)
    uv_path = _validate_uv(manifest, selected_dependencies, runner)
    _validate_ros_noetic(selected_dependencies)
    gpu_available = _probe_gpu(selected_dependencies)
    warnings: list[str] = []
    if not gpu_available:
        if definition.cuda_policy in {"required", "extension_required"}:
            _raise(
                _HOST_PREFLIGHT_ACTION,
                f"CUDA GPU is required by {model_id} "
                f"(cuda_policy={definition.cuda_policy})",
            )
        warnings.append(
            "CUDA GPU is not available; continuing because "
            f"{model_id} declares cuda_policy=recommended"
        )
    if definition.cuda_policy == "extension_required":
        _validate_sparse_host(selected_dependencies, runner)

    return EnvironmentPreflight(
        root=root,
        package_root=package_root,
        manifest=manifest,
        definition=definition,
        selected_profile=selected_profile,
        paths=paths,
        uv_path=uv_path,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _RuntimeAssetDeclaration:
    """검증된 asset 선언의 role, 상대/절대 경로와 expected hash를 보관한다."""

    role: str
    relative_path: Path
    absolute_path: Path
    expected_sha256: Optional[str]
    payload: Mapping[str, Any]


def _repository_relative_asset_path(value: Any, index: int) -> Path:
    """profile asset 경로를 canonical repository-relative POSIX path로 제한한다."""
    context = f"runtime_assets[{index}].path"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_PATH_STRING_LENGTH
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _raise(_ASSETS_ACTION, f"{context} must be a canonical repository-relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _raise(_ASSETS_ACTION, f"{context} must be a canonical repository-relative path")
    return Path(*pure.parts)


def _runtime_asset_declarations(
    preflight: EnvironmentPreflight,
) -> tuple[_RuntimeAssetDeclaration, ...]:
    """preflight profile의 runtime asset 선언을 mutation 없는 검증 값으로 바꾼다."""
    if not isinstance(preflight, EnvironmentPreflight):
        _raise(_ASSETS_ACTION, "preflight must be an EnvironmentPreflight")
    root = _canonical_absolute_posix_path(
        preflight.root, "project root", _ASSETS_ACTION
    )
    _require_real_directory(root, _ASSETS_ACTION, "project root")
    profile = getattr(preflight.selected_profile, "data", None)
    if not isinstance(profile, Mapping):
        _raise(_ASSETS_ACTION, "selected profile data must be a mapping")
    values = profile.get("runtime_assets")
    if not isinstance(values, list):
        _raise(_ASSETS_ACTION, "runtime_assets must be an array")

    declarations: list[_RuntimeAssetDeclaration] = []
    seen_paths: set[Path] = set()
    for index, payload in enumerate(values):
        if not isinstance(payload, Mapping):
            _raise(_ASSETS_ACTION, f"runtime_assets[{index}] must be an object")
        role = payload.get("role")
        if not isinstance(role, str) or not role:
            _raise(_ASSETS_ACTION, f"runtime_assets[{index}].role must be nonempty")
        expected_sha256 = payload.get("sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            _raise(
                _ASSETS_ACTION,
                f"runtime_assets[{index}].sha256 must be null or lowercase SHA-256",
            )
        relative = _repository_relative_asset_path(payload.get("path"), index)
        if relative in seen_paths:
            _raise(_ASSETS_ACTION, f"duplicate runtime asset path: {relative.as_posix()}")
        seen_paths.add(relative)
        absolute = _canonical_absolute_posix_path(
            root.joinpath(*relative.parts),
            f"runtime asset {role}",
            _ASSETS_ACTION,
        )
        declarations.append(
            _RuntimeAssetDeclaration(
                role=role,
                relative_path=relative,
                absolute_path=absolute,
                expected_sha256=expected_sha256,
                payload=payload,
            )
        )
    return tuple(declarations)


def _require_interfuser_asset_contract(
    preflight: EnvironmentPreflight,
    declarations: tuple[_RuntimeAssetDeclaration, ...],
) -> tuple[_RuntimeAssetDeclaration, _RuntimeAssetDeclaration]:
    """InterFuser의 checkpoint-first 두 asset 계약을 exact 순서로 검증한다."""
    model_id = getattr(preflight.selected_profile, "model_id", None)
    if model_id != "interfuser":
        _raise(_ASSETS_ACTION, "selected profile is not InterFuser")
    expected = (
        ("checkpoint", _INTERFUSER_CHECKPOINT_PATH),
        ("image_backbone", _INTERFUSER_BACKBONE_PATH),
    )
    if (
        len(declarations) != len(expected)
        or tuple(
            (declaration.role, declaration.relative_path)
            for declaration in declarations
        )
        != expected
        or any(declaration.expected_sha256 is None for declaration in declarations)
    ):
        _raise(
            _ASSETS_ACTION,
            "InterFuser requires exact checkpoint-first and image_backbone assets with SHA-256",
        )
    return declarations[0], declarations[1]


def _asset_error_is_missing(error: EnvironmentSetupError) -> bool:
    """secure open 오류 chain이 FileNotFoundError에서 시작됐는지 확인한다."""
    cause: Optional[BaseException] = error
    while cause is not None:
        if isinstance(cause, FileNotFoundError):
            return True
        cause = cause.__cause__
    return False


def _same_open_file(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    """stream 처리 중 열린 파일의 identity와 본문 metadata가 유지됐는지 확인한다."""
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def _fingerprint_runtime_asset(
    declaration: _RuntimeAssetDeclaration,
    *,
    missing_message: Optional[str] = None,
) -> AssetFingerprint:
    """용량 제한 없이 같은 no-follow regular fd에서 asset SHA와 metadata를 읽는다."""
    context = f"runtime asset {declaration.role}"
    try:
        with _open_regular_file(
            declaration.absolute_path, _ASSETS_ACTION, context
        ) as (descriptor, before):
            digest = hashlib.sha256()
            try:
                while True:
                    block = os.read(descriptor, _READ_BLOCK_SIZE)
                    if not block:
                        break
                    digest.update(block)
                after = os.fstat(descriptor)
            except (MemoryError, OSError) as exc:
                raise EnvironmentSetupError(
                    _ASSETS_ACTION,
                    f"cannot hash {context}: {declaration.absolute_path}",
                ) from exc
            if not _same_open_file(before, after):
                _raise(
                    _ASSETS_ACTION,
                    f"{context} changed while hashing: {declaration.absolute_path}",
                )
    except EnvironmentSetupError as exc:
        if missing_message is not None and _asset_error_is_missing(exc):
            raise EnvironmentSetupError(_ASSETS_ACTION, missing_message) from exc
        raise

    actual_sha256 = digest.hexdigest()
    if (
        declaration.expected_sha256 is not None
        and actual_sha256 != declaration.expected_sha256
    ):
        _raise(
            _ASSETS_ACTION,
            f"{context} SHA-256 mismatch for {declaration.absolute_path}: "
            f"expected {declaration.expected_sha256}, got {actual_sha256}",
        )
    return AssetFingerprint(
        path=declaration.relative_path,
        sha256=actual_sha256,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
    )


def _checkpoint_missing_message(declaration: _RuntimeAssetDeclaration) -> str:
    """InterFuser checkpoint의 수동 복구에 필요한 provenance 전체를 직렬화한다."""
    provenance = declaration.payload.get("provenance")
    values = provenance if isinstance(provenance, Mapping) else {}
    return (
        f"required InterFuser checkpoint is missing: {declaration.absolute_path}; "
        f"expected SHA-256 {declaration.expected_sha256}; "
        f"classification={values.get('classification', '<missing>')}; "
        f"repository={values.get('repository', '<missing>')}; "
        f"commit={values.get('commit', '<missing>')}; "
        f"google_drive_id={values.get('google_drive_id', '<missing>')}"
    )


@contextmanager
def _ensure_runtime_asset_parent(
    root: Path,
    relative_parent: Path,
) -> Iterator[int]:
    """project root fd 아래 parent를 component별 no-follow open/mkdir해 반환한다."""
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        _raise(_ASSETS_ACTION, "runtime asset preparation requires O_NOFOLLOW support")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | no_follow
    )
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with _open_absolute_nofollow(
        root, root_flags, _ASSETS_ACTION, "project root"
    ) as root_descriptor:
        current = root_descriptor
        owns_current = False
        try:
            for component in relative_parent.parts:
                try:
                    child = os.open(component, flags, dir_fd=current)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, mode=0o755, dir_fd=current)
                    except FileExistsError:
                        pass
                    except OSError as exc:
                        raise EnvironmentSetupError(
                            _ASSETS_ACTION,
                            f"cannot create runtime asset directory component: {component}",
                        ) from exc
                    try:
                        child = os.open(component, flags, dir_fd=current)
                    except OSError as exc:
                        raise EnvironmentSetupError(
                            _ASSETS_ACTION,
                            f"runtime asset parent is not a real directory: {component}",
                        ) from exc
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        message = (
                            "runtime asset parent contains a symlink or non-directory: "
                            f"{component}"
                        )
                    else:
                        message = f"cannot open runtime asset directory component: {component}"
                    raise EnvironmentSetupError(_ASSETS_ACTION, message) from exc
                if owns_current:
                    os.close(current)
                current = child
                owns_current = True
            yield current
        finally:
            if owns_current:
                os.close(current)


def _write_all(descriptor: int, block: bytes) -> None:
    """한 source block을 short write 없이 destination fd에 모두 기록한다."""
    offset = 0
    while offset < len(block):
        written = os.write(descriptor, block[offset:])
        if written <= 0:
            raise OSError("runtime asset copy made no progress")
        offset += written


def _copy_interfuser_user_cache(
    preflight: EnvironmentPreflight,
    declaration: _RuntimeAssetDeclaration,
    user_cache: Path,
) -> bool:
    """user cache가 있으면 same-fd hash와 atomic no-clobber install로 복사한다."""
    source = _canonical_absolute_posix_path(
        user_cache, "InterFuser user cache", _ASSETS_ACTION
    )
    context = "InterFuser user cache"
    try:
        source_file = _open_regular_file(source, _ASSETS_ACTION, context)
        with source_file as (source_descriptor, source_before):
            with _ensure_runtime_asset_parent(
                preflight.root, declaration.relative_path.parent
            ) as parent_descriptor:
                temporary_name: Optional[str] = None
                temporary_descriptor: Optional[int] = None
                try:
                    for _ in range(16):
                        candidate = f".asset-{secrets.token_hex(12)}.tmp"
                        try:
                            temporary_descriptor = os.open(
                                candidate,
                                os.O_WRONLY
                                | os.O_CREAT
                                | os.O_EXCL
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0),
                                0o644,
                                dir_fd=parent_descriptor,
                            )
                        except FileExistsError:
                            continue
                        temporary_name = candidate
                        break
                    if temporary_descriptor is None or temporary_name is None:
                        _raise(_ASSETS_ACTION, "cannot allocate runtime asset temporary file")

                    digest = hashlib.sha256()
                    try:
                        while True:
                            block = os.read(source_descriptor, _READ_BLOCK_SIZE)
                            if not block:
                                break
                            _write_all(temporary_descriptor, block)
                            digest.update(block)
                        source_after = os.fstat(source_descriptor)
                    except (MemoryError, OSError) as exc:
                        raise EnvironmentSetupError(
                            _ASSETS_ACTION,
                            f"cannot copy {context}: {source}",
                        ) from exc
                    if not _same_open_file(source_before, source_after):
                        _raise(_ASSETS_ACTION, f"{context} changed while copying: {source}")
                    actual_sha256 = digest.hexdigest()
                    if actual_sha256 != declaration.expected_sha256:
                        _raise(
                            _ASSETS_ACTION,
                            f"{context} SHA-256 mismatch for {source}: expected "
                            f"{declaration.expected_sha256}, got {actual_sha256}",
                        )
                    try:
                        os.fsync(temporary_descriptor)
                        os.close(temporary_descriptor)
                        temporary_descriptor = None
                        try:
                            os.link(
                                temporary_name,
                                declaration.absolute_path.name,
                                src_dir_fd=parent_descriptor,
                                dst_dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            )
                        except FileExistsError as exc:
                            raise EnvironmentSetupError(
                                _ASSETS_ACTION,
                                "runtime asset destination already exists; refusing "
                                f"to overwrite: {declaration.absolute_path}",
                            ) from exc
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
                        temporary_name = None
                        os.fsync(parent_descriptor)
                    except EnvironmentSetupError:
                        raise
                    except OSError as exc:
                        raise EnvironmentSetupError(
                            _ASSETS_ACTION,
                            f"cannot copy {context} into {declaration.absolute_path}",
                        ) from exc
                finally:
                    if temporary_descriptor is not None:
                        try:
                            os.close(temporary_descriptor)
                        except OSError:
                            pass
                    if temporary_name is not None:
                        try:
                            os.unlink(temporary_name, dir_fd=parent_descriptor)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            pass
    except EnvironmentSetupError as exc:
        missing_source_message = (
            f"{_ASSETS_ACTION}: {context} is missing: {source}"
        )
        if _asset_error_is_missing(exc) and str(exc) == missing_source_message:
            return False
        raise
    return True


def prepare_local_runtime_assets(
    preflight: EnvironmentPreflight,
    *,
    interfuser_user_cache: Path = _INTERFUSER_USER_CACHE,
) -> None:
    """필수 local checkpoint를 검증하고 exact InterFuser cache만 안전하게 준비한다."""
    declarations = _runtime_asset_declarations(preflight)
    if getattr(preflight.selected_profile, "model_id", None) != "interfuser":
        return
    checkpoint, backbone = _require_interfuser_asset_contract(
        preflight, declarations
    )
    _fingerprint_runtime_asset(
        checkpoint,
        missing_message=_checkpoint_missing_message(checkpoint),
    )
    try:
        _fingerprint_runtime_asset(backbone)
    except EnvironmentSetupError as exc:
        if not _asset_error_is_missing(exc):
            raise
    else:
        return
    _copy_interfuser_user_cache(preflight, backbone, interfuser_user_cache)


def fingerprint_runtime_assets(
    preflight: EnvironmentPreflight,
) -> tuple[AssetFingerprint, ...]:
    """선언 순서대로 모든 runtime asset의 deep SHA와 regular-fd metadata를 반환한다."""
    declarations = _runtime_asset_declarations(preflight)
    if getattr(preflight.selected_profile, "model_id", None) == "interfuser":
        _require_interfuser_asset_contract(preflight, declarations)
    return tuple(
        _fingerprint_runtime_asset(declaration) for declaration in declarations
    )


@dataclass
class _ModelOperationLockState:
    """한 process의 model root별 thread gate, flock fd와 재진입 깊이를 보관한다."""

    process_id: int
    thread_lock: Any
    users: int = 0
    depth: int = 0
    descriptor: Optional[int] = None


_MODEL_OPERATION_STATES: dict[Path, _ModelOperationLockState] = {}
_MODEL_OPERATION_STATES_GUARD = threading.Lock()


def _reset_model_operation_states_after_fork() -> None:
    """child process에서 상속 flock fd를 닫고 thread lock registry를 새로 만든다."""
    global _MODEL_OPERATION_STATES
    global _MODEL_OPERATION_STATES_GUARD
    inherited_states = tuple(_MODEL_OPERATION_STATES.values())
    _MODEL_OPERATION_STATES = {}
    _MODEL_OPERATION_STATES_GUARD = threading.Lock()
    for state in inherited_states:
        if state.descriptor is not None:
            try:
                os.close(state.descriptor)
            except OSError:
                pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_model_operation_states_after_fork)


def _reserve_model_operation_state(model_root: Path) -> _ModelOperationLockState:
    """model root state의 사용자 수를 올리고 process-local state를 반환한다."""
    process_id = os.getpid()
    with _MODEL_OPERATION_STATES_GUARD:
        state = _MODEL_OPERATION_STATES.get(model_root)
        if state is None or state.process_id != process_id:
            state = _ModelOperationLockState(process_id, threading.RLock())
            _MODEL_OPERATION_STATES[model_root] = state
        state.users += 1
        return state


def _release_model_operation_state(
    model_root: Path, state: _ModelOperationLockState
) -> None:
    """model root state 사용자 수를 내리고 마지막 사용 뒤 registry에서 제거한다."""
    with _MODEL_OPERATION_STATES_GUARD:
        state.users -= 1
        if state.users == 0 and _MODEL_OPERATION_STATES.get(model_root) is state:
            del _MODEL_OPERATION_STATES[model_root]


def _acquire_lifecycle_file_lock(model_root: Path, action: str) -> int:
    """secure lifecycle lock file을 열고 cross-process exclusive flock을 획득한다."""
    lock_path = model_root / _LIFECYCLE_LOCK_FILENAME
    _reject_existing_symlink_chain(lock_path, action, "lifecycle lock")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        _raise(action, "lifecycle lock requires O_NOFOLLOW support")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | no_follow
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(str(lock_path), flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _raise(action, "lifecycle lock must be a regular single-link file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_metadata = lock_path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            _raise(action, "lifecycle lock path changed during acquisition")
    except EnvironmentSetupError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise EnvironmentSetupError(
            action, f"cannot acquire lifecycle lock: {lock_path}"
        ) from exc
    return descriptor


def _release_lifecycle_file_lock(
    descriptor: int, model_root: Path, action: str
) -> None:
    """cross-process flock과 descriptor를 해제하고 release 오류를 정규화한다."""
    lock_path = model_root / _LIFECYCLE_LOCK_FILENAME
    release_error: Optional[OSError] = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError as exc:
        release_error = exc
    try:
        os.close(descriptor)
    except OSError as exc:
        if release_error is None:
            release_error = exc
    if release_error is not None:
        raise EnvironmentSetupError(
            action, f"cannot release lifecycle lock: {lock_path}"
        ) from release_error


@contextmanager
def _model_operation_lock(
    paths: EnvironmentPaths, action: str, create_root: bool
) -> Iterator[None]:
    """RLock 재진입과 한 flock fd를 결합해 model operation을 직렬화한다."""
    _validate_environment_paths(paths, action)
    model_root = _canonical_absolute_posix_path(
        paths.model_root, "model root", action
    )
    if create_root:
        _ensure_real_directory(model_root.parent, action, ".venvs")
        _ensure_real_directory(model_root, action, "model root")
    else:
        _require_real_directory(model_root, action, "model root")
    state = _reserve_model_operation_state(model_root)
    acquired_thread_lock = False
    body_failed = False
    try:
        state.thread_lock.acquire()
        acquired_thread_lock = True
        if state.depth == 0:
            state.descriptor = _acquire_lifecycle_file_lock(model_root, action)
        state.depth += 1
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            state.depth -= 1
            if state.depth == 0:
                descriptor = state.descriptor
                state.descriptor = None
                if descriptor is None:
                    _raise(action, "lifecycle lock state lost its descriptor")
                try:
                    _release_lifecycle_file_lock(descriptor, model_root, action)
                except EnvironmentSetupError:
                    if not body_failed:
                        raise
    finally:
        if acquired_thread_lock:
            state.thread_lock.release()
        _release_model_operation_state(model_root, state)


@contextmanager
def model_operation_lock(paths: EnvironmentPaths) -> Iterator[None]:
    """create부터 activate까지 재사용할 수 있는 reentrant model operation lease다."""
    with _model_operation_lock(paths, "model_operation_lock", create_root=True):
        yield


def create_candidate(paths: EnvironmentPaths, build_id: str) -> Path:
    """고유 build ID의 candidate를 움직이지 않을 최종 builds 경로에 만든다."""
    action = "create_candidate"
    _validate_environment_paths(paths, action)
    _validate_component(build_id, "build_id", action)
    model_root = Path(paths.model_root)
    venvs_root = model_root.parent
    _ensure_real_directory(venvs_root, action, ".venvs")
    _ensure_real_directory(model_root, action, "model root")
    _ensure_real_directory(Path(paths.builds), action, "builds")
    candidate = Path(paths.builds) / build_id
    try:
        candidate.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise EnvironmentSetupError(action, f"candidate already exists: {candidate}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create candidate: {candidate}") from exc
    _require_real_directory(candidate, action, "candidate")
    return candidate


def _validate_packages(packages: Sequence[str], action: str) -> tuple[str, ...]:
    """overlay package 목록이 exact 고정 allowlist인지 검사한다."""
    if isinstance(packages, (str, bytes)):
        _raise(action, "packages must be a sequence of package names")
    selected = tuple(packages)
    if not selected:
        _raise(action, "packages must not be empty")
    for package in selected:
        _validate_component(package, "package", action)
    if len(set(selected)) != len(selected):
        _raise(action, "packages must not contain duplicate names")
    if selected != PURE_ROS_PACKAGES:
        _raise(action, "packages must match the exact PURE_ROS_PACKAGES allowlist")
    return selected


def _is_native_filename(name: str) -> bool:
    """파일 이름이 Python extension 또는 native shared library인지 판별한다."""
    lower = name.lower()
    return lower.endswith(_NATIVE_SUFFIXES) or ".so." in lower


def _scan_python_tree(
    directory: Path,
    relative_directory: Path,
    action: str,
) -> list[tuple[Path, Path]]:
    """한 실제 package tree를 symlink 없이 순회해 Python source 목록을 만든다."""
    discovered: list[tuple[Path, Path]] = []
    try:
        with os.scandir(str(directory)) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot scan ROS package: {directory}") from exc
    for entry in ordered:
        source = directory / entry.name
        relative = relative_directory / entry.name
        _canonical_absolute_posix_path(source, "ROS source entry", action)
        if entry.is_symlink():
            _raise(action, f"ROS package tree contains a symlink: {source}")
        try:
            is_directory = entry.is_dir(follow_symlinks=False)
            is_file = entry.is_file(follow_symlinks=False)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot inspect ROS entry: {source}") from exc
        if is_directory:
            if entry.name == "__pycache__":
                continue
            discovered.extend(_scan_python_tree(source, relative, action))
            continue
        if not is_file:
            _raise(action, f"ROS package tree contains a special file: {source}")
        if _is_native_filename(entry.name):
            _raise(action, f"ROS package tree contains a native extension: {source}")
        if entry.name.endswith(".pyc"):
            continue
        if entry.name.endswith(".py"):
            discovered.append((source, relative))
    return discovered


def _source_package_entry(source_root: Path, package: str, action: str) -> Path:
    """source root에서 package entry를 follow 없이 찾아 실제 directory인지 확인한다."""
    try:
        with os.scandir(str(source_root)) as entries:
            entry = next((item for item in entries if item.name == package), None)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot scan ROS source root: {source_root}") from exc
    if entry is None:
        _raise(action, f"declared ROS package is missing: {package}")
    if entry.is_symlink():
        _raise(action, f"declared ROS package is a symlink: {package}")
    try:
        is_directory = entry.is_dir(follow_symlinks=False)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect ROS package: {package}") from exc
    if not is_directory:
        _raise(action, f"declared ROS package must be a directory: {package}")
    return source_root / package


def _require_package_init(package_root: Path, package: str, action: str) -> None:
    """선언 ROS package가 symlink가 아닌 일반 __init__.py를 갖는지 검사한다."""
    initializer = package_root / "__init__.py"
    try:
        metadata = initializer.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(
            action, f"declared ROS package requires __init__.py: {package}"
        ) from exc
    except OSError as exc:
        raise EnvironmentSetupError(
            action, f"cannot inspect ROS package __init__.py: {package}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        _raise(action, f"ROS package __init__.py is a symlink: {package}")
    if not stat.S_ISREG(metadata.st_mode):
        _raise(action, f"ROS package __init__.py must be a regular file: {package}")


def _module_is_copied(module: Sequence[str], copied: set[Path]) -> bool:
    """dotted module이 복사 예정 .py 또는 package/__init__.py로 존재하는지 판별한다."""
    if not module:
        return False
    module_path = Path(*module)
    return module_path.with_suffix(".py") in copied or (
        module_path / "__init__.py"
    ) in copied


@dataclass(frozen=True)
class _ParsedModule:
    """한 copied module의 AST와 static export 판정 자료를 보관한다."""

    relative_path: Path
    tree: ast.Module
    direct_bindings: frozenset[str]
    conditional_bindings: frozenset[str]
    wildcard_imports: tuple[ast.ImportFrom, ...]
    has_conditional_wildcard: bool
    has_dynamic_getattr: bool
    static_all_known: bool
    static_all: Optional[frozenset[str]]


def _target_names(target: ast.AST) -> set[str]:
    """assignment target가 현재 scope에 묶는 단순 name set을 반환한다."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _statement_binding_names(statement: ast.stmt) -> set[str]:
    """한 statement가 실행될 때 현재 scope에 묶을 수 있는 name을 찾는다."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        names: set[str] = set()
        for target in statement.targets:
            names.update(_target_names(target))
        return names
    if isinstance(statement, ast.AnnAssign):
        return _target_names(statement.target) if statement.value is not None else set()
    if isinstance(statement, ast.AugAssign):
        return _target_names(statement.target)
    if isinstance(statement, ast.Import):
        return {
            alias.asname if alias.asname is not None else alias.name.split(".")[0]
            for alias in statement.names
        }
    if isinstance(statement, ast.ImportFrom):
        return {
            alias.asname if alias.asname is not None else alias.name
            for alias in statement.names
            if alias.name != "*"
        }
    return set()


def _nested_statement_bindings(statements: Sequence[ast.stmt]) -> tuple[set[str], bool]:
    """conditional module statement 아래의 잠재 binding과 wildcard 여부를 찾는다."""
    names: set[str] = set()
    has_wildcard = False
    for statement in statements:
        names.update(_statement_binding_names(statement))
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name == "*" for alias in statement.names
        ):
            has_wildcard = True
        nested_groups: list[Sequence[ast.stmt]] = []
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            nested_groups.extend((statement.body, statement.orelse))
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                names.update(_target_names(statement.target))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested_groups.append(statement.body)
            for item in statement.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(statement, ast.Try):
            nested_groups.extend((statement.body, statement.orelse, statement.finalbody))
            nested_groups.extend(handler.body for handler in statement.handlers)
        for group in nested_groups:
            nested_names, nested_wildcard = _nested_statement_bindings(group)
            names.update(nested_names)
            has_wildcard = has_wildcard or nested_wildcard
    return names, has_wildcard


def _literal_string_collection(value: ast.AST) -> Optional[frozenset[str]]:
    """literal string list/tuple/set이면 순서와 중복을 제거한 값을 반환한다."""
    try:
        evaluated = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(evaluated, (list, tuple, set, frozenset)) or not all(
        isinstance(item, str) for item in evaluated
    ):
        return None
    return frozenset(evaluated)


def _static_all_policy(
    tree: ast.Module, conditional_bindings: set[str]
) -> tuple[bool, Optional[frozenset[str]]]:
    """module의 static __all__ set, 기본 정책 또는 동적 여부를 판정한다."""
    known = True
    explicit: Optional[frozenset[str]] = None
    for statement in tree.body:
        assigned_value: Optional[ast.AST] = None
        assigns_all = False
        if isinstance(statement, ast.Assign):
            assigns_all = any("__all__" in _target_names(target) for target in statement.targets)
            if assigns_all:
                assigned_value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            assigns_all = "__all__" in _target_names(statement.target)
            assigned_value = statement.value
        if assigns_all:
            if assigned_value is None:
                known = False
                explicit = None
                continue
            literal = _literal_string_collection(assigned_value)
            known = literal is not None
            explicit = literal
            continue
        if isinstance(statement, ast.AugAssign) and "__all__" in _target_names(
            statement.target
        ):
            addition = _literal_string_collection(statement.value)
            if (
                known
                and explicit is not None
                and isinstance(statement.op, ast.Add)
                and addition is not None
            ):
                explicit = explicit | addition
            else:
                known = False
                explicit = None
            continue
        if isinstance(statement, ast.Delete) and any(
            "__all__" in _target_names(target) for target in statement.targets
        ):
            known = True
            explicit = None
            continue
        if "__all__" in _statement_binding_names(statement):
            known = False
            explicit = None
    if "__all__" in conditional_bindings:
        return False, None
    return known, explicit


def _parsed_module(relative: Path, tree: ast.Module) -> _ParsedModule:
    """module body에서 static/conditional export 판정 자료를 구성한다."""
    direct: set[str] = set()
    conditional: set[str] = set()
    wildcard_imports: list[ast.ImportFrom] = []
    has_conditional_wildcard = False
    for statement in tree.body:
        if isinstance(statement, ast.Delete):
            for target in statement.targets:
                direct.difference_update(_target_names(target))
            continue
        direct.update(_statement_binding_names(statement))
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name == "*" for alias in statement.names
        ):
            wildcard_imports.append(statement)
        if isinstance(
            statement,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try),
        ):
            nested, nested_wildcard = _nested_statement_bindings([statement])
            conditional.update(nested)
            has_conditional_wildcard = (
                has_conditional_wildcard or nested_wildcard
            )
    static_all_known, static_all = _static_all_policy(tree, conditional)
    return _ParsedModule(
        relative_path=relative,
        tree=tree,
        direct_bindings=frozenset(direct),
        conditional_bindings=frozenset(conditional),
        wildcard_imports=tuple(wildcard_imports),
        has_conditional_wildcard=has_conditional_wildcard,
        has_dynamic_getattr=(
            "__getattr__" in direct or "__getattr__" in conditional
        ),
        static_all_known=static_all_known,
        static_all=static_all,
    )


def _module_name(relative: Path) -> tuple[str, ...]:
    """overlay-relative Python path를 import module parts로 변환한다."""
    if relative.name == "__init__.py":
        return tuple(relative.parent.parts)
    return tuple(relative.with_suffix("").parts)


def _relative_import_module(
    source: Path, node: ast.ImportFrom, action: str
) -> tuple[str, ...]:
    """한 relative ImportFrom의 기준 package와 명시 module을 절대 parts로 바꾼다."""
    package = list(source.parent.parts)
    if node.level > len(package):
        _raise(
            action,
            f"import closure escapes top-level package in {source.as_posix()}",
        )
    parent_count = node.level - 1
    if parent_count:
        package = package[:-parent_count]
    if node.module:
        package.extend(node.module.split("."))
    return tuple(package)


def _require_imported_module(
    module: Sequence[str], copied: set[Path], source: Path, action: str
) -> None:
    """참조한 선택 package module이 copied set에 없으면 closure 오류를 낸다."""
    if not _module_is_copied(module, copied):
        _raise(
            action,
            "import closure is missing module "
            + ".".join(module)
            + f" referenced by {source.as_posix()}",
        )


class _StaticExportResolver:
    """copied module의 static export를 cache와 cycle guard로 판정한다."""

    def __init__(
        self,
        modules: Mapping[tuple[str, ...], _ParsedModule],
        packages: tuple[str, ...],
    ) -> None:
        """parsed module map과 selected package set을 보관한다."""
        self._modules = modules
        self._selected = set(packages)
        self._cache: dict[tuple[tuple[str, ...], str], Optional[bool]] = {}
        self._visiting: set[tuple[tuple[str, ...], str]] = set()

    def _wildcard_target(
        self, source: Path, node: ast.ImportFrom, action: str
    ) -> Optional[tuple[str, ...]]:
        """local/selected wildcard target은 parts, 외부 target은 None으로 반환한다."""
        if node.level:
            return _relative_import_module(source, node, action)
        if node.module:
            module = tuple(node.module.split("."))
            if module[0] in self._selected:
                return module
        return None

    def _star_export_status(
        self, module: tuple[str, ...], symbol: str, action: str
    ) -> Optional[bool]:
        """provider module의 wildcard import가 symbol을 묶는지 판정한다."""
        parsed = self._modules.get(module)
        if parsed is None:
            return False
        if parsed.static_all_known:
            if parsed.static_all is None:
                allowed = not symbol.startswith("_")
            else:
                allowed = symbol in parsed.static_all
            if not allowed:
                return False
            return self.export_status(module, symbol, action)
        status = self.export_status(module, symbol, action)
        return False if status is False else None

    def export_status(
        self, module: tuple[str, ...], symbol: str, action: str
    ) -> Optional[bool]:
        """module symbol을 known present/absent 또는 runtime unknown으로 판정한다."""
        key = (module, symbol)
        if key in self._cache:
            return self._cache[key]
        if key in self._visiting:
            return False
        parsed = self._modules.get(module)
        if parsed is None:
            self._cache[key] = False
            return False
        self._visiting.add(key)
        try:
            if symbol in parsed.direct_bindings:
                status: Optional[bool] = True
            else:
                unknown = (
                    symbol in parsed.conditional_bindings
                    or parsed.has_conditional_wildcard
                    or parsed.has_dynamic_getattr
                )
                status = None if unknown else False
                for wildcard in parsed.wildcard_imports:
                    target = self._wildcard_target(
                        parsed.relative_path, wildcard, action
                    )
                    if target is None:
                        status = None
                        continue
                    inherited = self._star_export_status(target, symbol, action)
                    if inherited is True:
                        status = True
                        break
                    if inherited is None:
                        status = None
        finally:
            self._visiting.remove(key)
        self._cache[key] = status
        return status


def _validate_imported_aliases(
    node: ast.ImportFrom,
    module: tuple[str, ...],
    source: Path,
    copied: set[Path],
    resolver: _StaticExportResolver,
    action: str,
) -> None:
    """from-import alias가 copied submodule 또는 base static export인지 검사한다."""
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = (*module, *alias.name.split("."))
        if _module_is_copied(candidate, copied):
            continue
        if resolver.export_status(module, alias.name, action) is False:
            _raise(
                action,
                "import closure is missing export "
                + alias.name
                + " from "
                + ".".join(module)
                + f" referenced by {source.as_posix()}",
            )


def _validate_import_node(
    node: ast.AST,
    source: Path,
    packages: tuple[str, ...],
    copied: set[Path],
    resolver: _StaticExportResolver,
    action: str,
) -> None:
    """한 AST import node가 선택 package copied closure 안에서 해석되는지 검사한다."""
    selected = set(packages)
    if isinstance(node, ast.Import):
        for alias in node.names:
            module = tuple(alias.name.split("."))
            if module[0] in selected:
                _require_imported_module(module, copied, source, action)
        return
    if not isinstance(node, ast.ImportFrom):
        return
    if node.level:
        module = _relative_import_module(source, node, action)
        _require_imported_module(module, copied, source, action)
        _validate_imported_aliases(
            node, module, source, copied, resolver, action
        )
        return
    if not node.module:
        return
    module = tuple(node.module.split("."))
    if module[0] in selected:
        _require_imported_module(module, copied, source, action)
        _validate_imported_aliases(
            node, module, source, copied, resolver, action
        )


def _validate_import_closure(
    discovered: Sequence[tuple[Path, Path]],
    packages: tuple[str, ...],
    action: str,
) -> None:
    """모든 복사 예정 Python source의 선택 package와 relative import closure를 검사한다."""
    copied = {relative for _source, relative in discovered}
    parsed_modules: dict[tuple[str, ...], _ParsedModule] = {}
    for source_path, relative in discovered:
        try:
            tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
        except SyntaxError as exc:
            raise EnvironmentSetupError(
                action, f"Python syntax error in ROS source: {relative.as_posix()}"
            ) from exc
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot read ROS Python source: {source_path}"
            ) from exc
        module = _module_name(relative)
        if module in parsed_modules:
            _raise(action, f"import closure has duplicate module: {'.'.join(module)}")
        parsed_modules[module] = _parsed_module(relative, tree)
    resolver = _StaticExportResolver(parsed_modules, packages)
    for parsed in parsed_modules.values():
        for node in ast.walk(parsed.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _validate_import_node(
                    node,
                    parsed.relative_path,
                    packages,
                    copied,
                    resolver,
                    action,
                )


def _overlay_payload(
    packages: tuple[str, ...], files: tuple[RosOverlayFile, ...]
) -> dict[str, Any]:
    """overlay manifest dataclass를 deterministic JSON 기본 자료형으로 변환한다."""
    return {
        "schema": ROS_OVERLAY_MANIFEST_SCHEMA,
        "packages": list(packages),
        "files": [
            {
                "source": str(record.source_path),
                "destination": record.destination_path.as_posix(),
                "sha256": record.sha256,
            }
            for record in files
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], action: str) -> None:
    """canonical JSON을 sibling 임시 파일에 fsync한 뒤 원자적으로 교체한다."""
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > _MAX_MANIFEST_BYTES:
        _raise(action, "manifest size limit exceeded")
    temporary: Optional[Path] = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False
        )
        temporary = Path(handle.name)
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot write ROS overlay manifest: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_ros_overlay(
    source_root: Path,
    destination: Path,
    packages: Sequence[str] = PURE_ROS_PACKAGES,
) -> RosOverlayManifest:
    """허용 ROS package의 Python source만 새 candidate overlay로 복사한다."""
    action = "build_ros_overlay"
    source = _canonical_absolute_posix_path(source_root, "source_root", action)
    target = _canonical_absolute_posix_path(destination, "destination", action)
    _require_real_directory(source, action, "ROS source root")
    _reject_existing_symlink_chain(target.parent, action, "overlay parent")
    if target.is_symlink() or target.exists():
        _raise(action, f"destination already exists: {target}")
    selected_packages = _validate_packages(packages, action)
    manifest_path = target.parent / ROS_OVERLAY_MANIFEST_FILENAME
    if manifest_path.is_symlink() or manifest_path.exists():
        _raise(action, f"overlay manifest already exists: {manifest_path}")

    discovered: list[tuple[Path, Path]] = []
    for package in selected_packages:
        package_root = _source_package_entry(source, package, action)
        _require_package_init(package_root, package, action)
        package_files = _scan_python_tree(package_root, Path(package), action)
        if not package_files:
            _raise(action, f"declared ROS package has no Python source: {package}")
        discovered.extend(package_files)
    discovered.sort(key=lambda item: item[1].as_posix())
    if len(discovered) > _MAX_MANIFEST_RECORDS:
        _raise(action, "manifest file record limit exceeded")
    _validate_import_closure(discovered, selected_packages, action)

    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot create ROS overlay: {target}") from exc
    records: list[RosOverlayFile] = []
    for source_file, relative in discovered:
        destination_file = target / relative
        try:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                str(source_file), str(destination_file), follow_symlinks=False
            )
            metadata = destination_file.lstat()
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot copy ROS Python source: {source_file}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            _raise(action, f"copied ROS entry is not a regular file: {destination_file}")
        records.append(
            RosOverlayFile(
                source_path=source_file.absolute(),
                destination_path=relative,
                sha256=sha256_file(destination_file),
            )
        )

    # 전체-file AST closure는 Noetic의 optional rosnode/rosservice/em import 때문에
    # 거짓 실패한다. 실제 도달 가능한 import closure는 후속 target runtime smoke가 검증한다.
    files = tuple(records)
    _atomic_write_json(
        manifest_path, _overlay_payload(selected_packages, files), action
    )
    return RosOverlayManifest(
        schema=ROS_OVERLAY_MANIFEST_SCHEMA,
        packages=selected_packages,
        files=files,
        manifest_path=manifest_path.resolve(strict=True),
    )


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """overlay manifest JSON object의 duplicate key를 손상으로 거부한다."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _raise("verify_ros_overlay", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str, action: str
) -> None:
    """manifest object의 누락 key와 알 수 없는 key를 거부한다."""
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        _raise(action, f"{context} missing keys: {', '.join(sorted(missing))}")
    if unknown:
        _raise(action, f"{context} unknown keys: {', '.join(sorted(unknown))}")


def _safe_destination(value: Any, packages: tuple[str, ...], action: str) -> Path:
    """manifest destination을 package 아래 canonical 상대 POSIX Python path로 검증한다."""
    if isinstance(value, str) and len(value) > _MAX_PATH_STRING_LENGTH:
        _raise(action, "file destination path length limit exceeded")
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _raise(action, "file destination must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or not pure.parts
        or pure.parts[0] not in packages
        or not value.endswith(".py")
    ):
        _raise(action, "file destination must stay inside a declared package")
    return Path(*pure.parts)


def _safe_source(value: Any, destination: Path, action: str) -> Path:
    """manifest source를 destination suffix와 일치하는 절대 경로로 검증한다."""
    source = _canonical_absolute_posix_path(value, "file source", action)
    if len(source.parts) < len(destination.parts) or tuple(
        source.parts[-len(destination.parts) :]
    ) != destination.parts:
        _raise(action, "file source must match its destination suffix")
    return source


def _overlay_manifest_from_payload(
    payload: Any, path: Path, action: str
) -> RosOverlayManifest:
    """decoded manifest payload를 bounded immutable record로 검증한다."""
    if not isinstance(payload, dict):
        _raise(action, "manifest must be a JSON object")
    _require_exact_keys(payload, _MANIFEST_KEYS, "manifest", action)
    if (
        not isinstance(payload["schema"], int)
        or isinstance(payload["schema"], bool)
        or payload["schema"] != ROS_OVERLAY_MANIFEST_SCHEMA
    ):
        _raise(action, f"manifest schema must be {ROS_OVERLAY_MANIFEST_SCHEMA}")
    raw_packages = payload["packages"]
    if not isinstance(raw_packages, list):
        _raise(action, "manifest packages must be an array")
    packages = _validate_packages(raw_packages, action)
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        _raise(action, "manifest files must be a nonempty array")
    if len(raw_files) > _MAX_MANIFEST_RECORDS:
        _raise(action, "manifest file record limit exceeded")
    records: list[RosOverlayFile] = []
    destinations: set[Path] = set()
    sources: set[Path] = set()
    package_counts = {package: 0 for package in packages}
    for index, raw_file in enumerate(raw_files):
        context = f"manifest.files[{index}]"
        if not isinstance(raw_file, dict):
            _raise(action, f"{context} must be an object")
        _require_exact_keys(raw_file, _FILE_KEYS, context, action)
        destination = _safe_destination(raw_file["destination"], packages, action)
        source = _safe_source(raw_file["source"], destination, action)
        digest = raw_file["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _raise(action, f"{context} SHA-256 is invalid")
        if destination in destinations:
            _raise(action, f"duplicate file destination: {destination.as_posix()}")
        if source in sources:
            _raise(action, f"duplicate file source: {source}")
        destinations.add(destination)
        sources.add(source)
        package_counts[destination.parts[0]] += 1
        records.append(RosOverlayFile(source, destination, digest))
    missing_packages = sorted(
        package for package, count in package_counts.items() if count == 0
    )
    if missing_packages:
        _raise(action, f"manifest package has no files: {', '.join(missing_packages)}")
    if tuple(record.destination_path.as_posix() for record in records) != tuple(
        sorted(record.destination_path.as_posix() for record in records)
    ):
        _raise(action, "manifest file records must be sorted")
    return RosOverlayManifest(
        schema=ROS_OVERLAY_MANIFEST_SCHEMA,
        packages=packages,
        files=tuple(records),
        manifest_path=path,
    )


def _read_overlay_manifest(
    path: Path, action: str
) -> tuple[RosOverlayManifest, str]:
    """manifest를 같은 bounded regular fd에서 읽고 record와 digest를 반환한다."""
    if not path.is_absolute():
        _raise(action, "manifest path must be absolute")
    encoded = _read_bounded_regular_file(
        path, _MAX_MANIFEST_BYTES, action, "manifest"
    )
    try:
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
        manifest = _overlay_manifest_from_payload(payload, path, action)
    except EnvironmentSetupError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        MemoryError,
        RecursionError,
    ) as exc:
        raise EnvironmentSetupError(action, "manifest is not valid UTF-8 JSON") from exc
    return manifest, hashlib.sha256(encoded).hexdigest()


def _scan_overlay_tree(destination: Path, action: str) -> tuple[set[Path], set[Path]]:
    """overlay 실제 directory와 file set을 symlink를 따라가지 않고 수집한다."""
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()

    def scan(directory: Path, relative_directory: Path) -> None:
        """현재 실제 directory의 자식 entry를 재귀적으로 검사한다."""
        try:
            with os.scandir(str(directory)) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot scan overlay: {directory}") from exc
        for entry in ordered:
            path = directory / entry.name
            relative = relative_directory / entry.name
            if entry.is_symlink():
                _raise(action, f"overlay contains a symlink: {relative.as_posix()}")
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentSetupError(action, f"cannot inspect overlay: {path}") from exc
            if is_directory:
                actual_directories.add(relative)
                scan(path, relative)
            elif is_file:
                if _is_native_filename(entry.name):
                    _raise(action, f"overlay contains a native extension: {relative.as_posix()}")
                actual_files.add(relative)
            else:
                _raise(action, f"overlay contains a special file: {relative.as_posix()}")

    scan(destination, Path())
    return actual_files, actual_directories


def _expected_overlay_directories(files: Sequence[RosOverlayFile]) -> set[Path]:
    """manifest file의 모든 overlay-relative parent directory set을 만든다."""
    expected: set[Path] = set()
    for record in files:
        parent = record.destination_path.parent
        while parent != Path("."):
            expected.add(parent)
            parent = parent.parent
    return expected


def _verify_overlay_file(record: RosOverlayFile, destination: Path, action: str) -> None:
    """provenance source와 destination을 secure fd로 열어 같은 hash인지 검사한다."""
    source = record.source_path
    target = destination / record.destination_path
    for path, context in ((source, "source"), (target, "destination")):
        recorded_context = f"recorded {context} file"
        digest = _sha256_open_regular_file(path, action, recorded_context)
        if digest != record.sha256:
            _raise(action, f"recorded {context} file hash changed: {path}")


def verify_ros_overlay(destination: Path, manifest_path: Path) -> str:
    """overlay manifest와 exact tree를 검증하고 manifest 파일 SHA256을 반환한다."""
    action = "verify_ros_overlay"
    target = Path(destination)
    manifest_file = Path(manifest_path)
    if not target.is_absolute():
        _raise(action, "destination must be an absolute path")
    if manifest_file != target.parent / ROS_OVERLAY_MANIFEST_FILENAME:
        _raise(action, "manifest path must be the fixed overlay sibling")
    _require_real_directory(target, action, "overlay destination")
    manifest, manifest_digest = _read_overlay_manifest(manifest_file, action)
    actual_files, actual_directories = _scan_overlay_tree(target, action)
    expected_files = {record.destination_path for record in manifest.files}
    expected_directories = _expected_overlay_directories(manifest.files)
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files
    missing_directories = expected_directories - actual_directories
    extra_directories = actual_directories - expected_directories
    if missing_files:
        _raise(
            action,
            f"overlay file is missing: {sorted(path.as_posix() for path in missing_files)[0]}",
        )
    if extra_files:
        _raise(
            action,
            f"overlay contains an extra file: {sorted(path.as_posix() for path in extra_files)[0]}",
        )
    if missing_directories:
        _raise(
            action,
            "overlay directory is missing: "
            + sorted(path.as_posix() for path in missing_directories)[0],
        )
    if extra_directories:
        _raise(
            action,
            "overlay contains an extra directory: "
            + sorted(path.as_posix() for path in extra_directories)[0],
        )
    for record in manifest.files:
        _verify_overlay_file(record, target, action)
    return manifest_digest


def _validate_candidate(
    paths: EnvironmentPaths, candidate: Path, action: str
) -> tuple[Path, Path]:
    """candidate가 builds immediate 실제 directory인지 검증해 상대 target을 반환한다."""
    _validate_environment_paths(paths, action)
    model_root = Path(paths.model_root)
    builds = Path(paths.builds)
    selected = Path(candidate)
    if not selected.is_absolute() or selected.parent != builds:
        _raise(action, "candidate must be an immediate child of builds")
    _validate_component(selected.name, "build_id", action)
    try:
        metadata = selected.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(action, f"candidate is missing: {selected}") from exc
    except OSError as exc:
        raise EnvironmentSetupError(action, f"cannot inspect candidate: {selected}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        _raise(action, f"candidate must not be a symlink: {selected}")
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(action, f"candidate must be a real directory: {selected}")
    _require_real_directory(model_root, action, "model root")
    _require_real_directory(builds, action, "builds")
    relative = Path("builds") / selected.name
    return selected, relative


def activate_candidate(paths: EnvironmentPaths, candidate: Path) -> None:
    """candidate를 임시 상대 symlink와 os.replace로 원자적으로 활성화한다."""
    action = "activate_candidate"
    with _model_operation_lock(paths, action, create_root=False):
        _, relative = _validate_candidate(paths, Path(candidate), action)
        current = Path(paths.current)
        model_root = Path(paths.model_root)
        if not current.is_symlink() and current.exists():
            _raise(action, f"current exists and is not a symlink: {current}")
        temporary: Optional[Path] = None
        try:
            for _attempt in range(32):
                candidate_temp = (
                    model_root / f".current.{os.getpid()}.{secrets.token_hex(8)}"
                )
                try:
                    os.symlink(relative.as_posix(), str(candidate_temp))
                    temporary = candidate_temp
                    break
                except FileExistsError:
                    continue
            if temporary is None:
                _raise(action, "cannot allocate a temporary current symlink")
            os.replace(str(temporary), str(current))
            temporary = None
            _fsync_directory(model_root, action, "model root directory")
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"activate replace failed for {current}"
            ) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _active_build(paths: EnvironmentPaths, action: str) -> Optional[Path]:
    """current가 없으면 None, 있으면 안전한 active immediate build를 반환한다."""
    current = Path(paths.current)
    if current.is_symlink():
        try:
            relative = Path(os.readlink(str(current)))
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot read current symlink: {current}") from exc
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "builds"
        ):
            _raise(action, "current symlink target is malformed")
        _validate_component(relative.parts[1], "build_id", action)
        active = Path(paths.model_root) / relative
        try:
            metadata = active.lstat()
        except FileNotFoundError as exc:
            raise EnvironmentSetupError(action, "current symlink is dangling") from exc
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot inspect active build: {active}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _raise(action, "current must select a real immediate build directory")
        return active
    if current.exists():
        _raise(action, "current exists and is not a symlink")
    return None


def _require_unchanged_build(
    candidate: Path, expected_device: int, expected_inode: int, action: str
) -> None:
    """scan 뒤 candidate의 directory inode/device가 바뀌지 않았는지 재검증한다."""
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as exc:
        raise EnvironmentSetupError(
            action, f"build changed since scan: {candidate}"
        ) from exc
    except OSError as exc:
        raise EnvironmentSetupError(
            action, f"cannot revalidate build before prune: {candidate}"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_dev != expected_device
        or metadata.st_ino != expected_inode
    ):
        _raise(action, f"build changed since scan: {candidate}")


def prune_inactive_builds(paths: EnvironmentPaths) -> tuple[Path, ...]:
    """active build를 보존하고 검증된 inactive immediate directory만 정리한다."""
    action = "prune_inactive_builds"
    _validate_environment_paths(paths, action)
    model_root = Path(paths.model_root)
    builds = Path(paths.builds)
    if not model_root.exists() and not model_root.is_symlink():
        return ()
    with _model_operation_lock(paths, action, create_root=False):
        active = _active_build(paths, action)
        if not builds.exists() and not builds.is_symlink():
            if active is not None:
                _raise(action, "active build exists while builds directory is missing")
            return ()
        _require_real_directory(builds, action, "builds")
        candidates: list[tuple[Path, int, int]] = []
        try:
            with os.scandir(str(builds)) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise EnvironmentSetupError(action, f"cannot scan builds: {builds}") from exc
        for entry in ordered:
            candidate = builds / entry.name
            _validate_component(entry.name, "build_id", action)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentSetupError(
                    action, f"cannot inspect build: {candidate}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                _raise(action, f"suspicious symlink in builds: {candidate}")
            if not stat.S_ISDIR(metadata.st_mode):
                _raise(action, f"suspicious non-directory entry in builds: {candidate}")
            candidates.append((candidate, metadata.st_dev, metadata.st_ino))
        candidate_paths = tuple(candidate for candidate, _device, _inode in candidates)
        if active is not None and active not in candidate_paths:
            _raise(action, "active build is not an immediate builds entry")
        inactive_snapshots = tuple(
            snapshot for snapshot in candidates if snapshot[0] != active
        )
        for candidate, device, inode in inactive_snapshots:
            _require_unchanged_build(candidate, device, inode, action)
        for candidate, device, inode in inactive_snapshots:
            _require_unchanged_build(candidate, device, inode, action)
            try:
                shutil.rmtree(str(candidate))
            except OSError as exc:
                raise EnvironmentSetupError(
                    action, f"cannot prune build: {candidate}"
                ) from exc
        return tuple(candidate for candidate, _device, _inode in inactive_snapshots)


def _run_setup_command(
    runner: Callable[..., Any],
    argv: Sequence[str],
    action: str,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    capture_output: bool = False,
    timeout: Optional[float] = None,
) -> Any:
    """setup subprocess를 shell 없이 실행하고 phase별 domain 오류로 정규화한다."""
    command = [str(value) for value in argv]
    keyword_arguments: dict[str, Any] = {"check": True, "shell": False}
    if cwd is not None:
        keyword_arguments["cwd"] = cwd
    if env is not None:
        keyword_arguments["env"] = dict(env)
    if capture_output:
        keyword_arguments.update({"capture_output": True, "text": True})
    if timeout is not None:
        keyword_arguments["timeout"] = timeout
    try:
        result = runner(command, **keyword_arguments)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentSetupError(action, f"command failed: {' '.join(command)}") from exc
    if getattr(result, "returncode", 0) != 0:
        _raise(action, f"command failed: {' '.join(command)}")
    return result


def _setup_stdout(result: Any, action: str, context: str) -> str:
    """capture command의 stdout을 문자열로 제한해 반환한다."""
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        _raise(action, f"{context} returned invalid stdout")
    return stdout


def _validate_catkin_messages_read_only(
    preflight: EnvironmentPreflight,
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
) -> Path:
    """기존 Catkin output과 generated validator_msgs를 변경 없이 검사한다."""
    catkin_python = (
        preflight.root / "devel" / "lib" / "python3" / "dist-packages"
    )
    _require_real_directory(
        catkin_python, "catkin_messages", "Catkin Python output"
    )
    system_python = _require_executable_file(
        Path("/usr/bin/python3"), "system Python"
    )
    ros_python = _canonical_absolute_posix_path(
        dependencies.preflight_dependencies.ros_python_root,
        "ROS Python root",
        "catkin_messages",
    )
    import_code = (
        "import sys; "
        f"sys.path[:0] = [{str(catkin_python)!r}, {str(ros_python)!r}]; "
        "from validator_msgs.msg import "
        "CameraCalibration, DrivingFeature, InferenceMetric"
    )
    _run_setup_command(
        runner,
        (str(system_python), "-B", "-c", import_code),
        "catkin_messages",
        cwd=preflight.root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return catkin_python


def _build_catkin_messages(
    preflight: EnvironmentPreflight,
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
) -> Path:
    """Catkin workspace를 build한 뒤 generated validator_msgs를 검증한다."""
    catkin_make = _lookup_preflight_executable(
        dependencies.preflight_dependencies, "catkin_make", "catkin_make"
    )
    _run_setup_command(
        runner,
        (str(catkin_make),),
        "catkin_build",
        cwd=preflight.root,
    )
    return _validate_catkin_messages_read_only(preflight, dependencies, runner)


def _call_asset_fetcher(
    preflight: EnvironmentPreflight,
    dependencies: EnvironmentSetupDependencies,
) -> None:
    """injected fetcher를 선택 profile/model/root 인자로 실행하고 오류를 정규화한다."""
    try:
        dependencies.asset_fetcher(
            preflight.selected_profile.data,
            model_id=preflight.definition.model_id,
            project_root=preflight.root,
        )
    except EnvironmentSetupError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise EnvironmentSetupError("assets", f"asset fetch failed: {exc}") from exc


def _call_asset_verifier(
    preflight: EnvironmentPreflight,
    dependencies: EnvironmentSetupDependencies,
) -> Mapping[str, Any]:
    """injected deep asset/source verifier 결과를 mapping으로 제한한다."""
    try:
        metadata = dependencies.asset_verifier(
            preflight.selected_profile.data,
            project_root=preflight.root,
        )
    except EnvironmentSetupError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise EnvironmentSetupError("assets", f"asset verification failed: {exc}") from exc
    if not isinstance(metadata, Mapping):
        _raise("assets", "asset verifier must return a mapping")
    return metadata


def _source_checkout_from_metadata(
    preflight: EnvironmentPreflight,
    metadata: Mapping[str, Any],
) -> Path:
    """asset verifier metadata의 checkout을 실제 project source directory로 제한한다."""
    source = metadata.get("source")
    if not isinstance(source, Mapping):
        _raise("source_verification", "asset verifier returned no source metadata")
    checkout_value = source.get("checkout")
    checkout = _canonical_absolute_posix_path(
        checkout_value, "source checkout", "source_verification"
    )
    _require_real_directory(checkout, "source_verification", "source checkout")
    try:
        checkout.relative_to(preflight.root / "models" / "sources")
    except ValueError:
        _raise("source_verification", "source checkout escapes models/sources")
    return checkout


def _validate_source_checkout(
    preflight: EnvironmentPreflight,
    metadata: Mapping[str, Any],
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
    *,
    allowed_generated_paths: Sequence[Path] = (),
) -> Path:
    """source HEAD와 tracked/untracked dirty entries를 pinned contract와 대조한다."""
    source = metadata.get("source")
    if not isinstance(source, Mapping):
        _raise("source_verification", "asset verifier returned no source metadata")
    checkout = _source_checkout_from_metadata(preflight, metadata)
    git = _lookup_preflight_executable(
        dependencies.preflight_dependencies, "git", "git"
    )
    git_environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    head_result = _run_setup_command(
        runner,
        (str(git), "-C", str(checkout), "rev-parse", "HEAD"),
        "source_verification",
        cwd=preflight.root,
        env=git_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    observed = _setup_stdout(
        head_result, "source_verification", "git rev-parse"
    ).strip()
    if observed != preflight.definition.source_commit:
        _raise(
            "source_verification",
            "source HEAD mismatch: expected "
            f"{preflight.definition.source_commit}, observed {observed or '<empty>'}",
        )
    if source.get("observed_commit") != preflight.definition.source_commit:
        _raise("source_verification", "asset verifier source commit is stale")
    status_result = _run_setup_command(
        runner,
        (
            str(git),
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        "source_verification",
        cwd=preflight.root,
        env=git_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    status = _setup_stdout(
        status_result, "source_verification", "git status"
    )
    allowed_paths: set[str] = set()
    for generated in allowed_generated_paths:
        value = Path(generated)
        if (
            value.is_absolute()
            or not value.parts
            or any(part in {"", ".", ".."} for part in value.parts)
        ):
            _raise("source_verification", "generated source path is invalid")
        allowed_paths.add(value.as_posix())
    dirty_lines = tuple(line for line in status.splitlines() if line)
    invalid_lines = tuple(
        line
        for line in dirty_lines
        if len(line) <= 3
        or line[:3] != "?? "
        or line[3:] not in allowed_paths
    )
    if invalid_lines:
        _raise(
            "source_verification",
            "source checkout is dirty: " + ", ".join(invalid_lines),
        )
    return checkout


def _generated_artifact_declaration(
    preflight: EnvironmentPreflight,
    relative_path: Path,
    role: str,
) -> _RuntimeAssetDeclaration:
    """검증 대상으로 정한 repository-local generated file 선언을 만든다."""
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _raise(_ASSETS_ACTION, f"generated artifact path is invalid: {relative}")
    absolute = _canonical_absolute_posix_path(
        preflight.root.joinpath(*relative.parts),
        f"generated artifact {role}",
        _ASSETS_ACTION,
    )
    return _RuntimeAssetDeclaration(
        role=role,
        relative_path=relative,
        absolute_path=absolute,
        expected_sha256=None,
        payload={},
    )


def _diffusion_generated_path(
    preflight: EnvironmentPreflight, checkout: Path
) -> Path:
    """Diffusion derivative의 project-relative 고정 경로를 반환한다."""
    try:
        checkout_relative = checkout.relative_to(preflight.root)
    except ValueError:
        _raise(_ASSETS_ACTION, "Diffusion source checkout escapes project root")
    return checkout_relative / _DIFFUSION_DERIVATIVE


def _is_sparse_extension_filename(module: str, filename: str) -> bool:
    """한 exact Sparse module의 CPython extension filename인지 확인한다."""
    return (
        filename == f"{module}.so"
        or (
            filename.startswith(f"{module}.")
            and filename.endswith(".so")
            and len(filename) > len(module) + len(".so") + 1
        )
    )


@dataclass(frozen=True)
class _SparseGeneratedBaseline:
    """active state로 인증된 기존 Sparse extension fingerprint와 identity다."""

    paths: tuple[Path, ...]
    fingerprints: tuple[AssetFingerprint, ...]
    identities: tuple[tuple[int, int, int], ...]


def _existing_sparse_generated_paths(
    preflight: EnvironmentPreflight, checkout: Path
) -> tuple[Path, ...]:
    """이전 성공 build가 남긴 exact Sparse extension regular files만 찾는다."""
    ops = checkout / _SPARSE_OPS
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with _open_absolute_nofollow(
        ops, flags, "source_verification", "SparseDrive ops"
    ) as descriptor:
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        except OSError as exc:
            raise EnvironmentSetupError(
                "source_verification", "cannot inspect SparseDrive generated extensions"
            ) from exc
        selected: list[Path] = []
        for module in _SPARSE_EXTENSION_MODULES:
            matches = tuple(
                name for name in names if _is_sparse_extension_filename(module, name)
            )
            if len(matches) > 1:
                _raise(
                    "source_verification",
                    f"multiple generated SparseDrive extensions found for {module}",
                )
            if not matches:
                continue
            name = matches[0]
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentSetupError(
                    "source_verification",
                    f"cannot inspect generated SparseDrive extension: {name}",
                ) from exc
            if not stat.S_ISREG(metadata.st_mode):
                _raise(
                    "source_verification",
                    f"generated SparseDrive extension is not a regular file: {name}",
                )
            try:
                selected.append((ops / name).relative_to(preflight.root))
            except ValueError:
                _raise(
                    "source_verification",
                    "generated SparseDrive extension escapes project root",
                )
    return tuple(selected)


def _sparse_generated_identities(
    preflight: EnvironmentPreflight,
    paths: Sequence[Path],
    action: str,
) -> tuple[tuple[int, int, int], ...]:
    """exact Sparse extension의 device/inode/mtime identity를 안전하게 읽는다."""
    identities: list[tuple[int, int, int]] = []
    for relative in paths:
        absolute = preflight.root.joinpath(*Path(relative).parts)
        with _open_regular_file(
            absolute, action, f"generated SparseDrive extension {relative}"
        ) as (_, metadata):
            identities.append(
                (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns)
            )
    return tuple(identities)


def _authenticate_existing_sparse_generated_paths(
    preflight: EnvironmentPreflight,
    checkout: Path,
    paths: Sequence[Path],
    active_state: Optional[EnvironmentState],
) -> _SparseGeneratedBaseline:
    """기존 Sparse .so pair를 active state의 exact 저장 fingerprint로 인증한다."""
    selected = tuple(Path(path) for path in paths)
    if not selected:
        return _SparseGeneratedBaseline((), (), ())
    selected = _validate_sparse_generated_paths(
        preflight, checkout, selected, "source_verification"
    )
    if active_state is None:
        _raise(
            "source_verification",
            "pre-existing SparseDrive extensions require an authenticated active state",
        )
    if active_state.model_id != preflight.definition.model_id:
        _raise(
            "source_verification",
            "active state cannot authenticate SparseDrive extensions for this model",
        )
    if len(active_state.assets) < len(_SPARSE_EXTENSION_MODULES):
        _raise(
            "source_verification",
            "active state is missing authenticated SparseDrive extensions",
        )
    saved = tuple(active_state.assets[-len(_SPARSE_EXTENSION_MODULES) :])
    saved_paths = _validate_sparse_generated_paths(
        preflight,
        checkout,
        tuple(item.path for item in saved),
        "source_verification",
    )
    if selected != saved_paths:
        _raise(
            "source_verification",
            "pre-existing SparseDrive extensions do not match the active state",
        )
    observed = _fingerprint_generated_artifacts(
        preflight, checkout, sparse_paths=selected
    )
    if observed != saved:
        _raise(
            "source_verification",
            "pre-existing SparseDrive extension SHA does not match the active state",
        )
    return _SparseGeneratedBaseline(
        selected,
        saved,
        _sparse_generated_identities(
            preflight, selected, "source_verification"
        ),
    )


def _validate_sparse_generated_paths(
    preflight: EnvironmentPreflight,
    checkout: Path,
    paths: Sequence[Path],
    action: str,
) -> tuple[Path, ...]:
    """Sparse state/import origin을 ops root의 exact 두 ABI extension으로 제한한다."""
    selected = tuple(Path(path) for path in paths)
    try:
        ops_relative = (checkout / _SPARSE_OPS).relative_to(preflight.root)
    except ValueError:
        _raise(action, "SparseDrive ops directory escapes project root")
    if len(selected) != len(_SPARSE_EXTENSION_MODULES):
        _raise(action, "SparseDrive requires exactly two generated extensions")
    for module, relative in zip(_SPARSE_EXTENSION_MODULES, selected):
        if (
            relative.is_absolute()
            or relative.parent != ops_relative
            or not _is_sparse_extension_filename(module, relative.name)
        ):
            _raise(action, f"invalid generated SparseDrive extension: {relative}")
    if len(set(selected)) != len(selected):
        _raise(action, "generated SparseDrive extension paths must be unique")
    return selected


def _source_relative_generated_paths(
    preflight: EnvironmentPreflight,
    checkout: Path,
    project_relative_paths: Sequence[Path],
    action: str,
) -> tuple[Path, ...]:
    """generated project paths를 검증된 checkout 기준 경로로 바꾼다."""
    values: list[Path] = []
    for relative in project_relative_paths:
        try:
            values.append((preflight.root / relative).relative_to(checkout))
        except ValueError:
            _raise(action, f"generated artifact is outside source checkout: {relative}")
    return tuple(values)


def _fingerprint_generated_artifacts(
    preflight: EnvironmentPreflight,
    checkout: Path,
    *,
    sparse_paths: Sequence[Path] = (),
) -> tuple[AssetFingerprint, ...]:
    """선택 model의 executable generated source/native artifacts를 deep hash한다."""
    model_id = preflight.definition.model_id
    if model_id == "diffusiondrive_v2":
        if sparse_paths:
            _raise(_ASSETS_ACTION, "DiffusionDrive cannot declare Sparse extensions")
        declarations = (
            _generated_artifact_declaration(
                preflight,
                _diffusion_generated_path(preflight, checkout),
                "diffusiondrive_derivative",
            ),
        )
    elif model_id == "sparsedrive_v2":
        selected = _validate_sparse_generated_paths(
            preflight, checkout, sparse_paths, _ASSETS_ACTION
        )
        declarations = tuple(
            _generated_artifact_declaration(
                preflight, relative, f"sparse_extension:{module}"
            )
            for module, relative in zip(_SPARSE_EXTENSION_MODULES, selected)
        )
    else:
        if sparse_paths:
            _raise(_ASSETS_ACTION, "model does not declare generated extensions")
        declarations = ()
    return tuple(_fingerprint_runtime_asset(value) for value in declarations)


def _state_generated_paths(
    preflight: EnvironmentPreflight,
    checkout: Path,
    state: EnvironmentState,
    runtime_count: int,
    *,
    action: str = "reuse_verification",
) -> tuple[Path, ...]:
    """state tail이 model별 exact generated artifact set인지 검증해 반환한다."""
    tail = tuple(item.path for item in state.assets[runtime_count:])
    model_id = preflight.definition.model_id
    if model_id == "diffusiondrive_v2":
        expected = (_diffusion_generated_path(preflight, checkout),)
        if tail != expected:
            _raise(
                action,
                "state is missing the trusted DiffusionDrive derivative fingerprint",
            )
        return tail
    if model_id == "sparsedrive_v2":
        return _validate_sparse_generated_paths(
            preflight, checkout, tail, action
        )
    if tail:
        _raise(action, "state contains unexpected generated artifacts")
    return ()


def _validate_state_asset_contract(
    preflight: EnvironmentPreflight,
    checkout: Path,
    state: EnvironmentState,
    *,
    action: str,
) -> tuple[Path, ...]:
    """state asset prefix/tail을 현재 profile과 model generated 계약에 맞춘다."""
    declarations = _runtime_asset_declarations(preflight)
    if preflight.definition.model_id == "interfuser":
        _require_interfuser_asset_contract(preflight, declarations)
    if len(state.assets) < len(declarations):
        _raise(action, "environment state is missing declared runtime assets")
    runtime_state = state.assets[: len(declarations)]
    for declaration, fingerprint in zip(declarations, runtime_state):
        if fingerprint.path != declaration.relative_path:
            _raise(action, "environment state runtime asset paths are stale")
        if (
            declaration.expected_sha256 is not None
            and fingerprint.sha256 != declaration.expected_sha256
        ):
            _raise(action, "environment state runtime asset SHA is stale")
    return _state_generated_paths(
        preflight,
        checkout,
        state,
        len(declarations),
        action=action,
    )


def _expected_source_checkout(preflight: EnvironmentPreflight, action: str) -> Path:
    """선택 profile의 고정 source checkout을 deep asset verifier 없이 해석한다."""
    try:
        from validator_inference.assets import AssetError, source_checkout_path

        checkout = source_checkout_path(
            preflight.selected_profile.data, preflight.root
        )
    except (AssetError, KeyError, TypeError, ValueError) as exc:
        raise EnvironmentSetupError(action, "cannot resolve model source checkout") from exc
    checkout = _canonical_absolute_posix_path(checkout, "source checkout", action)
    _require_real_directory(checkout, action, "source checkout")
    try:
        checkout.relative_to(preflight.root / "models" / "sources")
    except ValueError:
        _raise(action, "source checkout escapes models/sources")
    return checkout


def _fingerprint_environment_inputs(
    preflight: EnvironmentPreflight,
    checkout: Path,
    generated_paths: Sequence[Path] = (),
) -> tuple[AssetFingerprint, ...]:
    """runtime assets와 model별 executable generated artifacts를 한 snapshot으로 묶는다."""
    runtime = fingerprint_runtime_assets(preflight)
    generated = _fingerprint_generated_artifacts(
        preflight,
        checkout,
        sparse_paths=(
            generated_paths
            if preflight.definition.model_id == "sparsedrive_v2"
            else ()
        ),
    )
    return runtime + generated


@dataclass(frozen=True)
class _OwnedSparseExtension:
    """한 transaction이 새로 생성했다고 등록한 extension artifact다."""

    module: str
    path: Path
    name: str
    fingerprint: AssetFingerprint
    identity: tuple[int, int, int]


@contextmanager
def _open_sparse_regular_entry(
    parent_descriptor: int,
    name: str,
    relative_path: Path,
    action: str,
) -> Iterator[tuple[int, os.stat_result, AssetFingerprint]]:
    """dirfd 아래 exact regular entry를 같은 fd에서 hash하고 열린 채 반환한다."""
    descriptor: Optional[int] = None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _raise(action, f"SparseDrive extension is not a regular file: {name}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, _READ_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if not _same_open_file(before, after):
            _raise(action, f"SparseDrive extension changed while hashing: {name}")
        yield descriptor, after, AssetFingerprint(
            path=relative_path,
            sha256=digest.hexdigest(),
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
        )
    except EnvironmentSetupError:
        raise
    except OSError as exc:
        raise EnvironmentSetupError(
            action, f"cannot securely inspect SparseDrive extension: {name}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _sparse_extension_matches(
    metadata: os.stat_result,
    fingerprint: AssetFingerprint,
    identity: tuple[int, int, int],
    expected_fingerprint: AssetFingerprint,
    expected_identity: tuple[int, int, int],
) -> bool:
    """열린 extension의 stored fingerprint와 inode identity가 모두 같은지 비교한다."""
    return (
        fingerprint == expected_fingerprint
        and identity == expected_identity
        and metadata.st_size == expected_fingerprint.size
        and metadata.st_mtime_ns == expected_fingerprint.mtime_ns
    )


def _sparse_extension_names(
    descriptor: int, action: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """열린 ops root에서 module별 ABI extension filename 후보를 반환한다."""
    try:
        names = tuple(sorted(os.listdir(descriptor)))
    except OSError as exc:
        raise EnvironmentSetupError(
            action, "cannot inspect SparseDrive extension output directory"
        ) from exc
    return tuple(
        (
            module,
            tuple(
                name
                for name in names
                if _is_sparse_extension_filename(module, name)
            ),
        )
        for module in _SPARSE_EXTENSION_MODULES
    )


def _duplicate_secure_directory(path: Path, action: str, context: str) -> int:
    """component no-follow로 연 directory fd를 transaction 수명용으로 복제한다."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with _open_absolute_nofollow(path, flags, action, context) as descriptor:
        try:
            duplicate = os.dup(descriptor)
            os.set_inheritable(duplicate, False)
            return duplicate
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot retain opened {context}: {path}"
            ) from exc


def _current_link_snapshot(paths: EnvironmentPaths) -> Optional[str]:
    """transaction 시작 시 current symlink의 exact target을 보관한다."""
    current = Path(paths.current)
    if current.is_symlink():
        try:
            return os.readlink(str(current))
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions", "cannot read current before SparseDrive build"
            ) from exc
    if current.exists():
        _raise(
            "sparse_extensions",
            "current must remain a symlink during SparseDrive build",
        )
    return None


def _restore_current_link_snapshot(
    paths: EnvironmentPaths,
    candidate: Path,
    snapshot: Optional[str],
) -> None:
    """activation failure가 current를 바꿨다면 transaction 이전 target을 복원한다."""
    current = Path(paths.current)
    if current.is_symlink():
        try:
            observed: Optional[str] = os.readlink(str(current))
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions", "cannot inspect current during rollback"
            ) from exc
    elif current.exists():
        _raise(
            "sparse_extensions",
            "current was replaced during SparseDrive rollback; preserving it",
        )
    else:
        observed = None
    if observed == snapshot:
        return
    candidate_target = (Path("builds") / candidate.name).as_posix()
    if observed not in {None, candidate_target}:
        _raise(
            "sparse_extensions",
            "current changed to an unrelated target during rollback; preserving it",
        )
    model_root = Path(paths.model_root)
    try:
        if snapshot is None:
            if observed == candidate_target:
                current.unlink()
                _fsync_directory(
                    model_root, "sparse_extensions", "model root directory"
                )
            return
        temporary: Optional[Path] = None
        for _attempt in range(32):
            value = model_root / f".current.rollback.{secrets.token_hex(8)}"
            try:
                os.symlink(snapshot, str(value))
                temporary = value
                break
            except FileExistsError:
                continue
        if temporary is None:
            _raise("sparse_extensions", "cannot allocate rollback current symlink")
        try:
            os.replace(str(temporary), str(current))
            temporary = None
            _fsync_directory(
                model_root, "sparse_extensions", "model root directory"
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    except EnvironmentSetupError:
        raise
    except OSError as exc:
        raise EnvironmentSetupError(
            "sparse_extensions", "cannot restore current after SparseDrive failure"
        ) from exc


@dataclass
class _SparseExtensionTransaction:
    """shared source extension pair의 backup/build/rollback ownership transaction이다."""

    preflight: EnvironmentPreflight
    checkout: Path
    candidate: Path
    baseline: _SparseGeneratedBaseline
    ops_descriptor: int
    candidate_descriptor: int
    backup_descriptor: int
    backup_name: str
    backup_metadata: os.stat_result
    current_snapshot: Optional[str]
    backup_names: list[str] = field(default_factory=list)
    owned: tuple[_OwnedSparseExtension, ...] = ()
    build_started: bool = False
    captured: bool = False
    finished: bool = False
    closed: bool = False

    @property
    def ops_relative(self) -> Path:
        """project root 기준 Sparse ops 경로다."""
        return (self.checkout / _SPARSE_OPS).relative_to(self.preflight.root)

    def _backup_existing(self) -> None:
        """attested old pair만 canonical ops name에서 private backup으로 이동한다."""
        for path, expected_fingerprint, expected_identity in zip(
            self.baseline.paths,
            self.baseline.fingerprints,
            self.baseline.identities,
        ):
            name = path.name
            with _open_sparse_regular_entry(
                self.ops_descriptor,
                name,
                path,
                "sparse_extensions",
            ) as (opened_descriptor, metadata, fingerprint):
                identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                )
                if not _sparse_extension_matches(
                    metadata,
                    fingerprint,
                    identity,
                    expected_fingerprint,
                    expected_identity,
                ):
                    _raise(
                        "sparse_extensions",
                        f"authenticated SparseDrive extension changed before backup: {name}",
                    )
                try:
                    os.rename(
                        name,
                        name,
                        src_dir_fd=self.ops_descriptor,
                        dst_dir_fd=self.backup_descriptor,
                    )
                    moved = os.stat(
                        name,
                        dir_fd=self.backup_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise EnvironmentSetupError(
                        "sparse_extensions",
                        f"cannot back up authenticated SparseDrive extension: {name}",
                    ) from exc
                opened = os.fstat(opened_descriptor)
                if not _same_sparse_entry_identity(moved, opened):
                    try:
                        os.rename(
                            name,
                            name,
                            src_dir_fd=self.backup_descriptor,
                            dst_dir_fd=self.ops_descriptor,
                        )
                    except OSError:
                        pass
                    _raise(
                        "sparse_extensions",
                        f"SparseDrive extension was replaced during backup: {name}",
                    )
                self.backup_names.append(name)
        try:
            os.fsync(self.ops_descriptor)
            os.fsync(self.backup_descriptor)
            os.fsync(self.candidate_descriptor)
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions", "cannot persist SparseDrive extension backup"
            ) from exc

    def begin_build(self) -> None:
        """canonical old names가 모두 비었음을 확인하고 build ownership window를 연다."""
        if self.build_started:
            _raise("sparse_extensions", "SparseDrive build transaction started twice")
        for _module, names in _sparse_extension_names(
            self.ops_descriptor, "sparse_extensions"
        ):
            if names:
                _raise(
                    "sparse_extensions",
                    "SparseDrive extension output names were not cleared before build",
                )
        self.build_started = True

    def capture_outputs(self, *, require_complete: bool) -> None:
        """controlled build window가 새로 만든 exact extension outputs를 등록한다."""
        if self.captured:
            _raise("sparse_extensions", "SparseDrive outputs were captured twice")
        records: list[_OwnedSparseExtension] = []
        invalid_modules: list[str] = []
        for module, names in _sparse_extension_names(
            self.ops_descriptor, "sparse_extensions"
        ):
            if len(names) != 1:
                invalid_modules.append(module)
            for name in names:
                path = self.ops_relative / name
                with _open_sparse_regular_entry(
                    self.ops_descriptor,
                    name,
                    path,
                    "sparse_extensions",
                ) as (_, metadata, fingerprint):
                    records.append(
                        _OwnedSparseExtension(
                            module=module,
                            path=path,
                            name=name,
                            fingerprint=fingerprint,
                            identity=(
                                metadata.st_dev,
                                metadata.st_ino,
                                metadata.st_mtime_ns,
                            ),
                        )
                    )
        self.owned = tuple(records)
        self.captured = True
        if require_complete and invalid_modules:
            _raise(
                "sparse_extensions",
                "fresh SparseDrive rebuild must create exactly one extension for: "
                + ", ".join(invalid_modules),
            )

    def verified_outputs(
        self, selected_paths: Sequence[Path]
    ) -> tuple[AssetFingerprint, ...]:
        """import origin pair가 transaction이 등록한 unchanged fresh outputs인지 확인한다."""
        selected = _validate_sparse_generated_paths(
            self.preflight,
            self.checkout,
            selected_paths,
            "sparse_extensions",
        )
        if selected != tuple(record.path for record in self.owned):
            _raise(
                "sparse_extensions",
                "imported SparseDrive extensions do not match fresh build outputs",
            )
        fingerprints: list[AssetFingerprint] = []
        for record in self.owned:
            with _open_sparse_regular_entry(
                self.ops_descriptor,
                record.name,
                record.path,
                "sparse_extensions",
            ) as (_, metadata, fingerprint):
                identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                )
                if not _sparse_extension_matches(
                    metadata,
                    fingerprint,
                    identity,
                    record.fingerprint,
                    record.identity,
                ):
                    _raise(
                        "sparse_extensions",
                        f"fresh SparseDrive extension changed after build: {record.name}",
                    )
                fingerprints.append(fingerprint)
        return tuple(fingerprints)

    def _preserve_current_entry(self, name: str) -> str:
        """ownership이 불명확한 canonical entry를 별도 이름으로 이동해 보존한다."""
        try:
            expected = os.stat(
                name, dir_fd=self.ops_descriptor, follow_symlinks=False
            )
            preserved = _unused_sparse_entry_name(
                self.ops_descriptor, f"validator-preserved.{name}"
            )
            os.rename(
                name,
                preserved,
                src_dir_fd=self.ops_descriptor,
                dst_dir_fd=self.ops_descriptor,
            )
            observed = os.stat(
                preserved, dir_fd=self.ops_descriptor, follow_symlinks=False
            )
            if not _same_sparse_entry_identity(expected, observed):
                try:
                    os.rename(
                        preserved,
                        name,
                        src_dir_fd=self.ops_descriptor,
                        dst_dir_fd=self.ops_descriptor,
                    )
                except OSError:
                    pass
                _raise(
                    "sparse_extensions",
                    f"SparseDrive extension changed while preserving it: {name}",
                )
            os.fsync(self.ops_descriptor)
            return preserved
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions",
                f"cannot preserve replaced SparseDrive extension: {name}",
            ) from exc

    def _remove_or_preserve_outputs(self) -> tuple[str, ...]:
        """owned unchanged output만 삭제하고 replacement/unknown entry는 보존한다."""
        expected_by_name = {record.name: record for record in self.owned}
        preserved: list[str] = []
        current_names = tuple(
            name
            for _module, names in _sparse_extension_names(
                self.ops_descriptor, "sparse_extensions"
            )
            for name in names
        )
        for name in current_names:
            record = expected_by_name.get(name)
            matches = False
            if record is not None:
                try:
                    with _open_sparse_regular_entry(
                        self.ops_descriptor,
                        name,
                        record.path,
                        "sparse_extensions",
                    ) as (_, metadata, fingerprint):
                        identity = (
                            metadata.st_dev,
                            metadata.st_ino,
                            metadata.st_mtime_ns,
                        )
                        matches = _sparse_extension_matches(
                            metadata,
                            fingerprint,
                            identity,
                            record.fingerprint,
                            record.identity,
                        )
                        if matches:
                            _remove_sparse_build_leaf(
                                self.ops_descriptor, name, metadata
                            )
                except EnvironmentSetupError:
                    matches = False
            if not matches:
                preserved.append(self._preserve_current_entry(name))
        return tuple(preserved)

    def _restore_baseline(self) -> None:
        """backup의 authenticated old pair를 original inode 그대로 canonical name에 둔다."""
        for path, expected_fingerprint, expected_identity in zip(
            self.baseline.paths,
            self.baseline.fingerprints,
            self.baseline.identities,
        ):
            name = path.name
            if name not in self.backup_names:
                continue
            try:
                os.stat(name, dir_fd=self.ops_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise EnvironmentSetupError(
                    "sparse_extensions",
                    f"cannot inspect SparseDrive restore target: {name}",
                ) from exc
            else:
                self._preserve_current_entry(name)
            with _open_sparse_regular_entry(
                self.backup_descriptor,
                name,
                path,
                "sparse_extensions",
            ) as (opened_descriptor, metadata, fingerprint):
                identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                )
                if not _sparse_extension_matches(
                    metadata,
                    fingerprint,
                    identity,
                    expected_fingerprint,
                    expected_identity,
                ):
                    _raise(
                        "sparse_extensions",
                        f"SparseDrive extension backup changed before restore: {name}",
                    )
                try:
                    os.rename(
                        name,
                        name,
                        src_dir_fd=self.backup_descriptor,
                        dst_dir_fd=self.ops_descriptor,
                    )
                    restored = os.stat(
                        name,
                        dir_fd=self.ops_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise EnvironmentSetupError(
                        "sparse_extensions",
                        f"cannot restore authenticated SparseDrive extension: {name}",
                    ) from exc
                opened = os.fstat(opened_descriptor)
                if not _same_sparse_entry_identity(restored, opened):
                    _raise(
                        "sparse_extensions",
                        f"SparseDrive extension was replaced during restore: {name}",
                    )
            self.backup_names.remove(name)
        try:
            os.fsync(self.ops_descriptor)
            os.fsync(self.backup_descriptor)
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions", "cannot persist SparseDrive extension restore"
            ) from exc

    def _finish_backup_directory(self) -> None:
        """empty private backup directory가 생성 당시 inode일 때만 제거한다."""
        try:
            if os.listdir(self.backup_descriptor):
                _raise(
                    "sparse_extensions",
                    "SparseDrive extension backup is not empty; preserving it",
                )
            observed = os.stat(
                self.backup_name,
                dir_fd=self.candidate_descriptor,
                follow_symlinks=False,
            )
            if not _same_sparse_entry_identity(observed, self.backup_metadata):
                _raise(
                    "sparse_extensions",
                    "SparseDrive extension backup directory was replaced; preserving it",
                )
            os.rmdir(self.backup_name, dir_fd=self.candidate_descriptor)
            os.fsync(self.candidate_descriptor)
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions", "cannot remove SparseDrive extension backup"
            ) from exc

    def rollback(self) -> None:
        """새 outputs를 철회하고 old pair/current snapshot을 exact 복원한다."""
        if self.finished:
            return
        preserved: tuple[str, ...] = ()
        if self.build_started:
            preserved = self._remove_or_preserve_outputs()
        self._restore_baseline()
        _restore_current_link_snapshot(
            self.preflight.paths, self.candidate, self.current_snapshot
        )
        self._finish_backup_directory()
        self.finished = True
        if preserved:
            _raise(
                "sparse_extensions",
                "replaced SparseDrive outputs were preserved during rollback: "
                + ", ".join(preserved),
            )

    def _verified_backup_metadata(
        self,
        path: Path,
        expected_fingerprint: AssetFingerprint,
        expected_identity: tuple[int, int, int],
    ) -> os.stat_result:
        """private backup entry가 authenticated baseline인지 확인한다."""
        name = path.name
        with _open_sparse_regular_entry(
            self.backup_descriptor,
            name,
            path,
            "sparse_extensions",
        ) as (_, metadata, fingerprint):
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime_ns,
            )
            if not _sparse_extension_matches(
                metadata,
                fingerprint,
                identity,
                expected_fingerprint,
                expected_identity,
            ):
                _raise(
                    "sparse_extensions",
                    f"SparseDrive extension backup changed before commit: {name}",
                )
            return metadata

    def prepare_commit(self) -> None:
        """새 output과 rollback backup을 폐기 전 재검증한다."""
        if self.finished:
            _raise("sparse_extensions", "SparseDrive transaction already finished")
        self.verified_outputs(tuple(record.path for record in self.owned))
        for path, expected_fingerprint, expected_identity in zip(
            self.baseline.paths,
            self.baseline.fingerprints,
            self.baseline.identities,
        ):
            name = path.name
            if name not in self.backup_names:
                continue
            self._verified_backup_metadata(
                path, expected_fingerprint, expected_identity
            )

    def finalize_commit(self) -> None:
        """logical activation 성공 뒤 authenticated old backup만 폐기한다."""
        if self.finished:
            _raise("sparse_extensions", "SparseDrive transaction already finished")
        for path, expected_fingerprint, expected_identity in zip(
            self.baseline.paths,
            self.baseline.fingerprints,
            self.baseline.identities,
        ):
            name = path.name
            if name not in self.backup_names:
                continue
            metadata = self._verified_backup_metadata(
                path, expected_fingerprint, expected_identity
            )
            _remove_sparse_build_leaf(self.backup_descriptor, name, metadata)
            self.backup_names.remove(name)
        self._finish_backup_directory()
        self.finished = True

    def close(self) -> None:
        """transaction이 보유한 directory descriptors를 idempotent하게 닫는다."""
        if self.closed:
            return
        self.closed = True
        for descriptor in (
            self.backup_descriptor,
            self.candidate_descriptor,
            self.ops_descriptor,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _begin_sparse_extension_transaction(
    preflight: EnvironmentPreflight,
    checkout: Path,
    candidate: Path,
    baseline: Optional[_SparseGeneratedBaseline],
) -> Optional[_SparseExtensionTransaction]:
    """Sparse model이면 private backup dirfd를 만들고 attested old pair를 이동한다."""
    if preflight.definition.model_id != "sparsedrive_v2":
        return None
    if baseline is None:
        _raise("sparse_extensions", "SparseDrive transaction lost its baseline")
    ops = checkout / _SPARSE_OPS
    ops_descriptor: Optional[int] = None
    candidate_descriptor: Optional[int] = None
    backup_descriptor: Optional[int] = None
    backup_name = f".sparse-extension-backup.{secrets.token_hex(16)}"
    transaction: Optional[_SparseExtensionTransaction] = None
    try:
        ops_descriptor = _duplicate_secure_directory(
            ops, "sparse_extensions", "SparseDrive ops"
        )
        candidate_descriptor = _duplicate_secure_directory(
            candidate, "sparse_extensions", "candidate directory"
        )
        os.mkdir(backup_name, mode=0o700, dir_fd=candidate_descriptor)
        backup_descriptor = os.open(
            backup_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=candidate_descriptor,
        )
        backup_metadata = os.fstat(backup_descriptor)
        if (
            not stat.S_ISDIR(backup_metadata.st_mode)
            or backup_metadata.st_dev != os.fstat(ops_descriptor).st_dev
        ):
            _raise(
                "sparse_extensions",
                "SparseDrive extension backup must share the source filesystem",
            )
        transaction = _SparseExtensionTransaction(
            preflight=preflight,
            checkout=checkout,
            candidate=candidate,
            baseline=baseline,
            ops_descriptor=ops_descriptor,
            candidate_descriptor=candidate_descriptor,
            backup_descriptor=backup_descriptor,
            backup_name=backup_name,
            backup_metadata=backup_metadata,
            current_snapshot=_current_link_snapshot(preflight.paths),
        )
        transaction._backup_existing()
        return transaction
    except BaseException as error:
        if transaction is not None:
            try:
                transaction.rollback()
            except BaseException as rollback_error:
                raise rollback_error from error
            finally:
                transaction.close()
        else:
            if backup_descriptor is not None:
                try:
                    os.close(backup_descriptor)
                except OSError:
                    pass
            if candidate_descriptor is not None:
                try:
                    os.rmdir(backup_name, dir_fd=candidate_descriptor)
                except OSError:
                    pass
                try:
                    os.close(candidate_descriptor)
                except OSError:
                    pass
            if ops_descriptor is not None:
                try:
                    os.close(ops_descriptor)
                except OSError:
                    pass
        raise


@dataclass
class _DiffusionDerivativeTransaction:
    """generated derivative의 pre-fetch hardlink backup과 activation rollback을 관리한다."""

    preflight: EnvironmentPreflight
    derivative_path: Path
    derivative_relative: Path
    model_descriptor: int
    backup_descriptor: int
    backup_name: str
    backup_metadata: os.stat_result
    current_snapshot: Optional[str]
    parent_descriptor: Optional[int] = None
    baseline_fingerprint: Optional[AssetFingerprint] = None
    baseline_identity: Optional[tuple[int, int, int]] = None
    backup_present: bool = False
    captured_fingerprint: Optional[AssetFingerprint] = None
    captured_identity: Optional[tuple[int, int, int]] = None
    captured: bool = False
    candidate: Optional[Path] = None
    finished: bool = False
    closed: bool = False

    def _ensure_parent_descriptor(self) -> int:
        """derivative parent를 component no-follow로 한 번 열어 transaction 동안 유지한다."""
        if self.parent_descriptor is None:
            self.parent_descriptor = _duplicate_secure_directory(
                self.derivative_path.parent,
                "diffusion_derivative",
                "DiffusionDrive derivative parent",
            )
        return self.parent_descriptor

    def _open_current(
        self,
    ) -> Iterator[tuple[int, os.stat_result, AssetFingerprint]]:
        """현재 derivative를 hash한 opened entry context로 반환한다."""
        return _open_sparse_regular_entry(
            self._ensure_parent_descriptor(),
            self.derivative_path.name,
            self.derivative_relative,
            "diffusion_derivative",
        )

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
        """state fingerprint에 없는 device/inode를 mtime과 함께 보관한다."""
        return metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns

    def _matches(
        self,
        metadata: os.stat_result,
        fingerprint: AssetFingerprint,
        expected_fingerprint: AssetFingerprint,
        expected_identity: tuple[int, int, int],
    ) -> bool:
        """derivative의 opened fingerprint와 inode가 expected snapshot인지 확인한다."""
        return _sparse_extension_matches(
            metadata,
            fingerprint,
            self._identity(metadata),
            expected_fingerprint,
            expected_identity,
        )

    def backup_existing(self, active_state: Optional[EnvironmentState]) -> None:
        """기존 derivative를 검증하고 같은 inode hardlink를 private backup에 둔다."""
        try:
            path_metadata = self.derivative_path.lstat()
        except FileNotFoundError:
            if active_state is not None:
                _raise(
                    "diffusion_derivative",
                    "active state derivative is missing before recreate",
                )
            return
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot inspect existing derivative"
            ) from exc
        if not stat.S_ISREG(path_metadata.st_mode):
            _raise(
                "diffusion_derivative",
                "existing DiffusionDrive derivative is not a regular file",
            )
        parent_descriptor = self._ensure_parent_descriptor()
        with self._open_current() as (_, metadata, fingerprint):
            identity = self._identity(metadata)
            if active_state is not None:
                if (
                    active_state.model_id != "diffusiondrive_v2"
                    or not active_state.assets
                    or active_state.assets[-1].path != self.derivative_relative
                ):
                    _raise(
                        "diffusion_derivative",
                        "active state cannot authenticate the DiffusionDrive derivative",
                    )
                if fingerprint != active_state.assets[-1]:
                    _raise(
                        "diffusion_derivative",
                        "existing DiffusionDrive derivative does not match active state SHA",
                    )
            try:
                os.link(
                    self.derivative_path.name,
                    self.derivative_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=self.backup_descriptor,
                    follow_symlinks=False,
                )
                backup_metadata = os.stat(
                    self.derivative_path.name,
                    dir_fd=self.backup_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise EnvironmentSetupError(
                    "diffusion_derivative",
                    "cannot create DiffusionDrive derivative rollback link",
                ) from exc
            if not _same_sparse_entry_identity(metadata, backup_metadata):
                _raise(
                    "diffusion_derivative",
                    "DiffusionDrive derivative changed while creating rollback link",
                )
            self.baseline_fingerprint = fingerprint
            self.baseline_identity = identity
            self.backup_present = True
        try:
            os.fsync(parent_descriptor)
            os.fsync(self.backup_descriptor)
            os.fsync(self.model_descriptor)
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot persist derivative rollback link"
            ) from exc

    def _verify_backup(self) -> tuple[os.stat_result, AssetFingerprint]:
        """private hardlink가 original baseline bytes/metadata를 유지하는지 확인한다."""
        if (
            not self.backup_present
            or self.baseline_fingerprint is None
            or self.baseline_identity is None
        ):
            _raise("diffusion_derivative", "derivative rollback link is unavailable")
        with _open_sparse_regular_entry(
            self.backup_descriptor,
            self.derivative_path.name,
            self.derivative_relative,
            "diffusion_derivative",
        ) as (_, metadata, fingerprint):
            if not self._matches(
                metadata,
                fingerprint,
                self.baseline_fingerprint,
                self.baseline_identity,
            ):
                _raise(
                    "diffusion_derivative",
                    "DiffusionDrive derivative rollback link was modified",
                )
            return metadata, fingerprint

    def capture_output(self) -> None:
        """asset fetcher가 publish한 derivative identity/SHA를 ownership snapshot으로 잡는다."""
        if self.captured:
            _raise("diffusion_derivative", "derivative output was captured twice")
        with self._open_current() as (_, metadata, fingerprint):
            self.captured_fingerprint = fingerprint
            self.captured_identity = self._identity(metadata)
        self.captured = True
        if self.backup_present:
            self._verify_backup()

    def capture_if_present(self) -> None:
        """fetcher 예외가 파일을 남겼다면 rollback ownership을 best-effort로 등록한다."""
        if self.captured:
            return
        try:
            self.derivative_path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot inspect failed derivative output"
            ) from exc
        self.capture_output()

    def bind_candidate(self, candidate: Path) -> None:
        """activation rollback이 허용할 이번 candidate target을 기록한다."""
        if self.candidate is not None:
            _raise("diffusion_derivative", "derivative candidate was bound twice")
        self.candidate = Path(candidate)

    def _preserve_current(self) -> str:
        """transaction-owned snapshot과 다른 derivative를 hidden sibling으로 보존한다."""
        parent_descriptor = self._ensure_parent_descriptor()
        name = self.derivative_path.name
        try:
            expected = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            preserved = _unused_sparse_entry_name(
                parent_descriptor,
                f"validator-preserved.{name}",
                action="diffusion_derivative",
                context="DiffusionDrive derivative",
            )
            os.rename(
                name,
                preserved,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            observed = os.stat(
                preserved, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if not _same_sparse_entry_identity(expected, observed):
                _raise(
                    "diffusion_derivative",
                    "derivative changed while preserving replacement",
                )
            os.fsync(parent_descriptor)
            return preserved
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot preserve replaced derivative"
            ) from exc

    def _current_snapshot(
        self,
    ) -> Optional[tuple[os.stat_result, AssetFingerprint]]:
        """현재 canonical derivative가 없으면 None, 있으면 opened snapshot을 반환한다."""
        try:
            self.derivative_path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot inspect derivative during rollback"
            ) from exc
        with self._open_current() as (_, metadata, fingerprint):
            return metadata, fingerprint

    def _remove_backup_link(self) -> None:
        """verified old hardlink만 quarantine identity check 뒤 제거한다."""
        metadata, _ = self._verify_backup()
        _remove_generated_leaf(
            self.backup_descriptor,
            self.derivative_path.name,
            metadata,
            "diffusion_derivative",
            "DiffusionDrive derivative rollback link",
        )
        self.backup_present = False

    def _restore_baseline(self) -> None:
        """backup hardlink를 canonical name으로 rename해 original inode를 복원한다."""
        self._verify_backup()
        parent_descriptor = self._ensure_parent_descriptor()
        try:
            os.rename(
                self.derivative_path.name,
                self.derivative_path.name,
                src_dir_fd=self.backup_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            restored = os.stat(
                self.derivative_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot restore active derivative"
            ) from exc
        if (
            self.baseline_identity is None
            or (
                restored.st_dev,
                restored.st_ino,
                restored.st_mtime_ns,
            )
            != self.baseline_identity
        ):
            _raise(
                "diffusion_derivative",
                "active derivative was replaced during restore",
            )
        self.backup_present = False
        try:
            os.fsync(parent_descriptor)
            os.fsync(self.backup_descriptor)
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot persist derivative restore"
            ) from exc

    def _finish_backup_directory(self) -> None:
        """empty private derivative backup directory를 original inode일 때만 제거한다."""
        try:
            if os.listdir(self.backup_descriptor):
                _raise(
                    "diffusion_derivative",
                    "derivative rollback directory is not empty; preserving it",
                )
            observed = os.stat(
                self.backup_name,
                dir_fd=self.model_descriptor,
                follow_symlinks=False,
            )
            if not _same_sparse_entry_identity(observed, self.backup_metadata):
                _raise(
                    "diffusion_derivative",
                    "derivative rollback directory was replaced; preserving it",
                )
            os.rmdir(self.backup_name, dir_fd=self.model_descriptor)
            os.fsync(self.model_descriptor)
        except EnvironmentSetupError:
            raise
        except OSError as exc:
            raise EnvironmentSetupError(
                "diffusion_derivative", "cannot remove derivative rollback directory"
            ) from exc

    def rollback(self) -> None:
        """failed setup output을 철회하고 old derivative/current target을 복원한다."""
        if self.finished:
            return
        self.capture_if_present()
        current = self._current_snapshot()
        baseline_is_current = False
        owned_is_current = False
        if current is not None:
            metadata, fingerprint = current
            if (
                self.baseline_fingerprint is not None
                and self.baseline_identity is not None
            ):
                baseline_is_current = self._matches(
                    metadata,
                    fingerprint,
                    self.baseline_fingerprint,
                    self.baseline_identity,
                )
            if (
                self.captured_fingerprint is not None
                and self.captured_identity is not None
            ):
                owned_is_current = self._matches(
                    metadata,
                    fingerprint,
                    self.captured_fingerprint,
                    self.captured_identity,
                )
            if not baseline_is_current:
                if owned_is_current:
                    _remove_generated_leaf(
                        self._ensure_parent_descriptor(),
                        self.derivative_path.name,
                        metadata,
                        "diffusion_derivative",
                        "generated DiffusionDrive derivative",
                    )
                else:
                    self._preserve_current()
        if self.backup_present:
            if baseline_is_current:
                self._remove_backup_link()
            else:
                self._restore_baseline()
        if self.candidate is None:
            current = Path(self.preflight.paths.current)
            observed = os.readlink(str(current)) if current.is_symlink() else None
            if observed != self.current_snapshot:
                _raise(
                    "diffusion_derivative",
                    "current changed before derivative candidate was created",
                )
        else:
            _restore_current_link_snapshot(
                self.preflight.paths, self.candidate, self.current_snapshot
            )
        self._finish_backup_directory()
        self.finished = True

    def prepare_commit(self) -> None:
        """captured derivative와 rollback link를 폐기 전 재검증한다."""
        if self.finished:
            _raise("diffusion_derivative", "derivative transaction already finished")
        current = self._current_snapshot()
        if (
            current is None
            or self.captured_fingerprint is None
            or self.captured_identity is None
            or not self._matches(
                current[0],
                current[1],
                self.captured_fingerprint,
                self.captured_identity,
            )
        ):
            _raise(
                "diffusion_derivative",
                "active derivative changed before transaction commit",
            )
        if self.backup_present:
            self._verify_backup()

    def finalize_commit(self) -> None:
        """logical activation 성공 뒤 old derivative hardlink만 폐기한다."""
        if self.finished:
            _raise("diffusion_derivative", "derivative transaction already finished")
        if self.backup_present:
            self._remove_backup_link()
        self._finish_backup_directory()
        self.finished = True

    def close(self) -> None:
        """derivative transaction directory descriptors를 idempotent하게 닫는다."""
        if self.closed:
            return
        self.closed = True
        descriptors = [self.backup_descriptor, self.model_descriptor]
        if self.parent_descriptor is not None:
            descriptors.append(self.parent_descriptor)
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _begin_diffusion_derivative_transaction(
    preflight: EnvironmentPreflight,
    active_state: Optional[EnvironmentState],
) -> Optional[_DiffusionDerivativeTransaction]:
    """Diffusion model이면 pre-fetch hardlink rollback transaction을 시작한다."""
    if preflight.definition.model_id != "diffusiondrive_v2":
        return None
    checkout = preflight.root / "models" / "sources" / "diffusiondrive_v2"
    derivative_relative = checkout.relative_to(preflight.root) / _DIFFUSION_DERIVATIVE
    derivative_path = preflight.root / derivative_relative
    model_descriptor: Optional[int] = None
    backup_descriptor: Optional[int] = None
    transaction: Optional[_DiffusionDerivativeTransaction] = None
    backup_name = f".diffusion-derivative-backup.{secrets.token_hex(16)}"
    try:
        model_descriptor = _duplicate_secure_directory(
            preflight.paths.model_root,
            "diffusion_derivative",
            "DiffusionDrive model root",
        )
        os.mkdir(backup_name, mode=0o700, dir_fd=model_descriptor)
        backup_descriptor = os.open(
            backup_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=model_descriptor,
        )
        backup_metadata = os.fstat(backup_descriptor)
        transaction = _DiffusionDerivativeTransaction(
            preflight=preflight,
            derivative_path=derivative_path,
            derivative_relative=derivative_relative,
            model_descriptor=model_descriptor,
            backup_descriptor=backup_descriptor,
            backup_name=backup_name,
            backup_metadata=backup_metadata,
            current_snapshot=_current_link_snapshot(preflight.paths),
        )
        transaction.backup_existing(active_state)
        return transaction
    except BaseException as error:
        if transaction is not None:
            try:
                transaction.rollback()
            except BaseException as rollback_error:
                raise rollback_error from error
            finally:
                transaction.close()
        else:
            if backup_descriptor is not None:
                try:
                    os.close(backup_descriptor)
                except OSError:
                    pass
            if model_descriptor is not None:
                try:
                    os.rmdir(backup_name, dir_fd=model_descriptor)
                except OSError:
                    pass
                try:
                    os.close(model_descriptor)
                except OSError:
                    pass
        raise


def _candidate_install_environment(candidate: Path) -> dict[str, str]:
    """native build에 사용할 candidate-first 환경에서 foreign activation을 제거한다."""
    def is_activation_variable(name: str) -> bool:
        """native build에서 제거할 Python/venv/Conda 변수를 식별한다."""
        return (
            name in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "_CE_CONDA", "_CE_M"}
            or name.startswith("VIRTUAL_ENV")
            or name.startswith("_OLD_VIRTUAL_")
            or name.startswith("CONDA_")
            or name.startswith("_CONDA")
            or name.startswith("__CONDA")
        )

    def path_key(value: str) -> str:
        """PATH entry 비교용 realpath key를 만든다."""
        return os.path.normcase(os.path.normpath(os.path.realpath(value)))

    inherited = {
        key: value
        for key, value in os.environ.items()
        if not is_activation_variable(key) and key != "PATH"
    }
    retained = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and path_key(entry) != path_key(str(candidate / "bin"))
    ]
    inherited.update(
        {
            "PATH": os.pathsep.join((str(candidate / "bin"), *retained)),
            "VIRTUAL_ENV": str(candidate),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return inherited


def _validate_candidate_python_version(
    preflight: EnvironmentPreflight,
    candidate_python: Path,
    runner: Callable[..., Any],
) -> None:
    """candidate interpreter의 exact patch version을 manifest 정의와 대조한다."""
    result = _run_setup_command(
        runner,
        (
            str(candidate_python),
            "-B",
            "-c",
            "import platform; print(platform.python_version())",
        ),
        "python_version",
        cwd=preflight.root,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    observed = _setup_stdout(result, "python_version", "Python version").strip()
    if observed != preflight.definition.python_version:
        _raise(
            "python_version",
            f"expected {preflight.definition.python_version}, observed {observed or '<empty>'}",
        )


def _reserve_sparse_build_tree(build_directory: Path) -> int:
    """부재한 ops/build를 선점하고 inode 재사용을 막는 directory fd를 반환한다."""
    try:
        build_directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise EnvironmentSetupError(
            "sparse_extensions",
            f"pre-existing SparseDrive build tree must be preserved: {build_directory}",
        ) from exc
    except OSError as exc:
        raise EnvironmentSetupError(
            "sparse_extensions", "cannot reserve SparseDrive build tree"
        ) from exc
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            str(build_directory),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        metadata = os.fstat(descriptor)
        path_metadata = build_directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _raise(
                "sparse_extensions",
                "reserved SparseDrive build path is not a real directory",
            )
        if (
            path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            _raise("sparse_extensions", "SparseDrive build tree changed during reserve")
        _fsync_directory(
            build_directory.parent, "sparse_extensions", "SparseDrive ops"
        )
    except BaseException:
        remove_owned_entry = False
        if descriptor is not None:
            try:
                expected = os.fstat(descriptor)
                actual = build_directory.lstat()
                remove_owned_entry = (
                    stat.S_ISDIR(actual.st_mode)
                    and not stat.S_ISLNK(actual.st_mode)
                    and actual.st_dev == expected.st_dev
                    and actual.st_ino == expected.st_ino
                )
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        if remove_owned_entry:
            try:
                build_directory.rmdir()
            except OSError:
                pass
        raise
    if descriptor is None:
        _raise("sparse_extensions", "SparseDrive build reservation lost its descriptor")
    return descriptor


def _sparse_build_entry_metadata(
    parent_descriptor: int, name: str
) -> os.stat_result:
    """열린 ops dir 기준 build entry를 symlink를 따르지 않고 검사한다."""
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        _raise(
            "sparse_extensions",
            "generated SparseDrive build tree changed or was removed",
        )
    except OSError as exc:
        raise EnvironmentSetupError(
            "sparse_extensions", "cannot inspect generated SparseDrive build tree"
        ) from exc


def _require_sparse_build_identity(
    metadata: os.stat_result, expected: os.stat_result
) -> None:
    """build name이 선점 당시 열린 directory와 같은 inode인지 확인한다."""
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != expected.st_dev
        or metadata.st_ino != expected.st_ino
    ):
        _raise(
            "sparse_extensions",
            "generated SparseDrive build tree changed or was replaced; preserving it",
        )


def _same_sparse_entry_identity(
    left: os.stat_result, right: os.stat_result
) -> bool:
    """두 no-follow entry metadata가 같은 inode와 file type을 가리키는지 확인한다."""
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _unused_sparse_entry_name(
    descriptor: int,
    prefix: str,
    *,
    action: str = "sparse_extensions",
    context: str = "SparseDrive",
) -> str:
    """열린 directory 안에서 추측하기 어려운 부재 quarantine 이름을 예약한다."""
    for _attempt in range(32):
        name = f".{prefix}.{secrets.token_hex(16)}"
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return name
        except OSError as exc:
            raise EnvironmentSetupError(
                action, f"cannot inspect {context} quarantine name"
            ) from exc
    _raise(action, f"cannot allocate {context} quarantine name")


def _remove_generated_leaf(
    descriptor: int,
    name: str,
    expected: os.stat_result,
    action: str,
    context: str,
) -> None:
    """leaf를 random quarantine으로 옮겨 opened identity 확인 후에만 삭제한다."""
    path_flag = getattr(os, "O_PATH", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not path_flag or not no_follow:
        _raise(
            action,
            f"platform lacks secure {context} cleanup support",
        )
    opened_descriptor: Optional[int] = None
    quarantine = _unused_sparse_entry_name(
        descriptor,
        "validator-remove",
        action=action,
        context=context,
    )
    try:
        opened_descriptor = os.open(
            name,
            path_flag | no_follow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
        opened = os.fstat(opened_descriptor)
        if not _same_sparse_entry_identity(opened, expected):
            _raise(
                action,
                f"{context} changed before cleanup",
            )
        os.rename(
            name,
            quarantine,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        quarantined = os.stat(
            quarantine, dir_fd=descriptor, follow_symlinks=False
        )
        if not _same_sparse_entry_identity(quarantined, opened):
            try:
                os.rename(
                    quarantine,
                    name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                )
            except OSError:
                pass
            _raise(
                action,
                f"{context} was replaced; preserving it",
            )
        os.unlink(quarantine, dir_fd=descriptor)
        os.fsync(descriptor)
    except EnvironmentSetupError:
        raise
    except OSError as exc:
        raise EnvironmentSetupError(
            action,
            f"cannot quarantine {context}",
        ) from exc
    finally:
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def _remove_sparse_build_leaf(
    descriptor: int, name: str, expected: os.stat_result
) -> None:
    """Sparse build leaf를 identity-preserving quarantine으로 삭제한다."""
    _remove_generated_leaf(
        descriptor,
        name,
        expected,
        "sparse_extensions",
        "generated SparseDrive build leaf",
    )


def _clear_sparse_build_tree_fd(descriptor: int) -> None:
    """열린 build dir 아래 entry를 dir_fd 상대 연산으로만 재귀 삭제한다."""
    try:
        names = tuple(sorted(os.listdir(descriptor)))
    except OSError as exc:
        raise EnvironmentSetupError(
            "sparse_extensions", "cannot inspect generated SparseDrive build contents"
        ) from exc
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise EnvironmentSetupError(
                "sparse_extensions",
                "generated SparseDrive build contents changed during cleanup",
            ) from exc
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor: Optional[int] = None
            try:
                child_descriptor = os.open(
                    name, directory_flags, dir_fd=descriptor
                )
                opened = os.fstat(child_descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                ):
                    _raise(
                        "sparse_extensions",
                        "generated SparseDrive build contents changed during cleanup",
                    )
                _clear_sparse_build_tree_fd(child_descriptor)
                current = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or current.st_dev != opened.st_dev
                    or current.st_ino != opened.st_ino
                ):
                    _raise(
                        "sparse_extensions",
                        "generated SparseDrive build contents changed during cleanup",
                    )
                os.rmdir(name, dir_fd=descriptor)
            except EnvironmentSetupError:
                raise
            except OSError as exc:
                raise EnvironmentSetupError(
                    "sparse_extensions",
                    "cannot remove generated SparseDrive build contents",
                ) from exc
            finally:
                if child_descriptor is not None:
                    try:
                        os.close(child_descriptor)
                    except OSError:
                        pass
            continue
        _remove_sparse_build_leaf(descriptor, name, metadata)


def _remove_created_sparse_build_tree(
    build_directory: Path, descriptor: int
) -> None:
    """이번 Sparse build가 만든 intermediate tree만 symlink-safe하게 제거한다."""
    try:
        expected = os.fstat(descriptor)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        with _open_absolute_nofollow(
            build_directory.parent,
            parent_flags,
            "sparse_extensions",
            "SparseDrive ops",
        ) as parent_descriptor:
            metadata = _sparse_build_entry_metadata(
                parent_descriptor, build_directory.name
            )
            _require_sparse_build_identity(metadata, expected)
            _clear_sparse_build_tree_fd(descriptor)
            metadata = _sparse_build_entry_metadata(
                parent_descriptor, build_directory.name
            )
            _require_sparse_build_identity(metadata, expected)
            try:
                os.rmdir(build_directory.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError as exc:
                raise EnvironmentSetupError(
                    "sparse_extensions",
                    "cannot remove generated SparseDrive build tree",
                ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _run_sparse_setup_action(
    preflight: EnvironmentPreflight,
    candidate_python: Path,
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
    *,
    transaction: Optional[_SparseExtensionTransaction] = None,
) -> tuple[AssetFingerprint, ...]:
    """SparseDrive의 유일한 declared CUDA extension action을 exact 환경에서 실행한다."""
    actions = preflight.definition.setup_actions
    if not actions:
        return ()
    if (
        preflight.definition.model_id != "sparsedrive_v2"
        or actions != ("build_sparsedrive_cuda_extensions",)
    ):
        _raise("sparse_extensions", "unsupported setup action contract")
    if transaction is None:
        _raise("sparse_extensions", "SparseDrive build requires an output transaction")
    checkout = preflight.root / "models" / "sources" / "sparsedrive_v2"
    ops = checkout / _SPARSE_OPS
    _require_real_directory(ops, "sparse_extensions", "SparseDrive ops")
    build_directory = ops / "build"
    build_environment = _candidate_install_environment(candidate_python.parent.parent)
    build_environment.update(
        {
            "CUDA_HOME": str(dependencies.preflight_dependencies.cuda_root),
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "CC": str(dependencies.preflight_dependencies.gcc_path),
            "CXX": str(dependencies.preflight_dependencies.gxx_path),
        }
    )
    transaction.begin_build()
    build_descriptor = _reserve_sparse_build_tree(build_directory)
    try:
        try:
            _run_setup_command(
                runner,
                (str(candidate_python), "setup.py", "build_ext", "--inplace"),
                "sparse_extensions",
                cwd=ops,
                env=build_environment,
                timeout=_MODEL_LOAD_TIMEOUT_SECONDS,
            )
        except BaseException:
            transaction.capture_outputs(require_complete=False)
            raise
        else:
            transaction.capture_outputs(require_complete=True)
    finally:
        _remove_created_sparse_build_tree(build_directory, build_descriptor)
    import_code = (
        "import importlib, json, pathlib, sys; "
        "root = pathlib.Path(sys.argv[1]).resolve(); "
        "sys.path.insert(0, str(root)); "
        f"names = {_SPARSE_EXTENSION_MODULES!r}; "
        "origins = [pathlib.Path(importlib.import_module(name).__file__).resolve() "
        "for name in names]; "
        "assert all(root == origin.parent for origin in origins); "
        "print(json.dumps([str(origin) for origin in origins]))"
    )
    result = _run_setup_command(
        runner,
        (str(candidate_python), "-c", import_code, str(ops)),
        "sparse_extensions",
        cwd=preflight.root,
        env=build_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    try:
        values = json.loads(
            _setup_stdout(result, "sparse_extensions", "SparseDrive extension imports")
        )
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise EnvironmentSetupError(
            "sparse_extensions", "SparseDrive extension imports returned invalid JSON"
        ) from exc
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        _raise(
            "sparse_extensions",
            "SparseDrive extension imports must return a JSON path array",
        )
    project_paths: list[Path] = []
    for value in values:
        origin = _canonical_absolute_posix_path(
            value, "SparseDrive extension origin", "sparse_extensions"
        )
        try:
            project_paths.append(origin.relative_to(preflight.root))
        except ValueError:
            _raise(
                "sparse_extensions",
                f"SparseDrive extension origin escapes project root: {origin}",
            )
    selected_paths = _validate_sparse_generated_paths(
        preflight, checkout, project_paths, "sparse_extensions"
    )
    return transaction.verified_outputs(selected_paths)


def _build_candidate_child_environment(
    preflight: EnvironmentPreflight,
    candidate: Path,
    catkin_python: Path,
) -> dict[str, str]:
    """model checks와 ROS smoke에 사용할 sanitized candidate 환경을 만든다."""
    try:
        environment = build_model_process_environment(
            os.environ,
            candidate,
            candidate / ".ros-overlay",
            catkin_python,
            preflight.root,
        )
    except EnvironmentStateError as exc:
        raise EnvironmentSetupError("candidate_environment", str(exc)) from exc
    if preflight.definition.model_id == "sparsedrive_v2":
        ops = (
            preflight.root
            / "models"
            / "sources"
            / "sparsedrive_v2"
            / "navsim"
            / "agents"
            / "sparsedrive"
            / "ops"
        )
        _require_real_directory(ops, "candidate_environment", "SparseDrive ops")
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(ops), environment["PYTHONPATH"])
        )
    return environment


def _run_ros_smoke(
    preflight: EnvironmentPreflight,
    candidate_python: Path,
    child_environment: Mapping[str, str],
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
) -> None:
    """injected hook 또는 system/target 양방향 ROS 호환성 smoke를 실행한다."""
    if dependencies.ros_smoke is not None:
        try:
            dependencies.ros_smoke(candidate_python, dict(child_environment))
        except EnvironmentSetupError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise EnvironmentSetupError("ros_smoke", f"ROS smoke failed: {exc}") from exc
        return
    policy = _ros_runtime_path_policy(
        preflight,
        candidate_python,
        preflight.root / "devel" / "lib" / "python3" / "dist-packages",
    )
    try:
        _run_ros_compatibility_smoke(policy)
    except EnvironmentSetupError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise EnvironmentSetupError("ros_smoke", f"ROS smoke failed: {exc}") from exc


def _ros_runtime_path_policy(
    preflight: EnvironmentPreflight,
    candidate_python: Path,
    catkin_python: Path,
) -> Any:
    """resolved candidate interpreter의 base stdlib 하나로 ROS path policy를 만든다."""
    action = "ros_smoke"
    candidate = candidate_python.parent.parent
    if candidate_python != candidate / "bin" / "python":
        _raise(action, "target Python must be candidate/bin/python")
    try:
        resolved_python = candidate_python.resolve(strict=True)
    except OSError as exc:
        raise EnvironmentSetupError(
            action, "target Python cannot be resolved"
        ) from exc
    version = _VERSIONED_PYTHON_EXECUTABLE.fullmatch(resolved_python.name)
    if version is None:
        _raise(action, "resolved target Python must be named pythonX.Y")
    stdlib = (
        resolved_python.parent.parent
        / "lib"
        / f"python{version.group(1)}.{version.group(2)}"
    )
    try:
        resolved_stdlib = stdlib.resolve(strict=True)
    except OSError as exc:
        raise EnvironmentSetupError(
            action, "resolved target Python base stdlib is missing"
        ) from exc
    if not resolved_stdlib.is_dir():
        _raise(action, "resolved target Python base stdlib must be a directory")
    source_root = _canonical_absolute_posix_path(
        preflight.package_root / "src",
        "validator_inference source root",
        action,
    )
    _require_real_directory(
        source_root, action, "validator_inference source root"
    )
    try:
        from validator_inference.ros_compatibility import RuntimePathPolicy
    except (ImportError, AttributeError) as exc:
        raise EnvironmentSetupError(
            action, "ROS compatibility runtime is unavailable"
        ) from exc
    return RuntimePathPolicy(
        target_python=candidate_python,
        target_environment=candidate,
        ros_overlay=candidate / ".ros-overlay",
        catkin_python_path=catkin_python,
        target_stdlib_paths=(resolved_stdlib,),
        project_source_paths=(source_root,),
    )


def _run_ros_compatibility_smoke(policy: Any) -> None:
    """full ROS smoke implementation을 lazy import해 setup module cycle을 피한다."""
    from validator_inference.ros_compatibility import run_ros_compatibility_smoke

    run_ros_compatibility_smoke(
        policy,
        timeout_seconds=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )


def _run_target_ros_import_smoke(
    candidate_python: Path,
    candidate: Path,
    child_environment: Mapping[str, str],
    runner: Callable[..., Any],
) -> None:
    """roslaunch 직전 target interpreter의 overlay와 generated message imports를 검사한다."""
    code = (
        "import rospy; import std_msgs.msg; import sensor_msgs.msg; "
        "import geometry_msgs.msg; import nav_msgs.msg; import tf2_msgs.msg; "
        "import actionlib_msgs.msg; from validator_msgs.msg import "
        "CameraCalibration, DrivingFeature, InferenceMetric"
    )
    _run_setup_command(
        runner,
        (str(candidate_python), "-B", "-c", code),
        "target_ros_import",
        cwd=candidate,
        env=child_environment,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )


def _verify_import_origin_file(
    origin: Path,
    trusted_root: Path,
    context: str,
) -> Path:
    """민감 module origin이 real trusted root 아래 regular file인지 확인한다."""
    action = "required_imports"
    _require_real_directory(trusted_root, action, f"{context} root")
    try:
        resolved_root = trusted_root.resolve(strict=True)
        resolved_origin = origin.resolve(strict=True)
        metadata = resolved_origin.stat()
    except OSError as exc:
        raise EnvironmentSetupError(
            action, f"{context} origin cannot be resolved: {origin}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        _raise(action, f"{context} origin must be a regular file: {origin}")
    try:
        resolved_origin.relative_to(resolved_root)
    except ValueError:
        _raise(action, f"{context} origin escapes trusted root: {origin}")
    return resolved_origin


def _run_candidate_import_cuda_checks(
    preflight: EnvironmentPreflight,
    candidate: Path,
    child_environment: Mapping[str, str],
    runner: Callable[..., Any],
) -> Optional[str]:
    """required imports의 origin과 CUDA policy를 model load 없이 검사한다."""
    python = candidate / "bin" / "python"
    checkout = _expected_source_checkout(preflight, "required_imports")
    source_import_root = (
        checkout / "interfuser"
        if preflight.definition.model_id == "interfuser"
        else None
    )
    if source_import_root is not None:
        _require_real_directory(
            source_import_root, "required_imports", "InterFuser vendored package root"
        )
    imports_code = (
        "import importlib, json, pathlib, sys; names = "
        f"{preflight.definition.required_imports!r}; "
        "source_root = pathlib.Path(sys.argv[1]) if sys.argv[1] else None; "
        "sys.path.insert(0, str(source_root)) if source_root is not None else None; "
        "modules = [importlib.import_module(name) for name in names]; "
        "print(json.dumps([getattr(module, '__file__', None) for module in modules]))"
    )
    import_result = _run_setup_command(
        runner,
        (
            str(python),
            "-B",
            "-c",
            imports_code,
            str(source_import_root) if source_import_root is not None else "",
        ),
        "required_imports",
        cwd=candidate,
        env=child_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    try:
        origins = json.loads(
            _setup_stdout(import_result, "required_imports", "required import origins")
        )
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise EnvironmentSetupError(
            "required_imports", "required imports returned invalid origin JSON"
        ) from exc
    if (
        not isinstance(origins, list)
        or len(origins) != len(preflight.definition.required_imports)
        or any(not isinstance(origin, str) or not origin for origin in origins)
    ):
        _raise("required_imports", "required imports returned invalid origins")
    origin_paths = tuple(
        _canonical_absolute_posix_path(
            origin, f"required import {name} origin", "required_imports"
        )
        for name, origin in zip(preflight.definition.required_imports, origins)
    )
    if preflight.definition.model_id == "interfuser":
        if "timm" not in preflight.definition.required_imports:
            _raise("required_imports", "InterFuser requires vendored timm import")
        timm_index = preflight.definition.required_imports.index("timm")
        try:
            origin_paths[timm_index].relative_to(source_import_root / "timm")
        except ValueError:
            _raise("required_imports", "InterFuser timm origin escapes pinned checkout")
        _verify_import_origin_file(
            origin_paths[timm_index],
            source_import_root / "timm",
            "InterFuser timm",
        )
    if preflight.definition.model_id == "sparsedrive_v2":
        sparse_origins = tuple(
            origin_paths[preflight.definition.required_imports.index(module)]
            for module in _SPARSE_EXTENSION_MODULES
        )
        sparse_ops = checkout / _SPARSE_OPS
        for origin in sparse_origins:
            try:
                origin.relative_to(sparse_ops)
            except ValueError:
                _raise(
                    "required_imports",
                    "SparseDrive extension origin escapes attested ops directory",
                )
        try:
            project_paths = tuple(
                origin.relative_to(preflight.root) for origin in sparse_origins
            )
        except ValueError:
            _raise("required_imports", "SparseDrive extension origin escapes project root")
        selected = _validate_sparse_generated_paths(
            preflight, checkout, project_paths, "required_imports"
        )
        for origin in sparse_origins:
            _verify_import_origin_file(
                origin, sparse_ops, "SparseDrive extension"
            )
        if selected != _existing_sparse_generated_paths(preflight, checkout):
            _raise("required_imports", "SparseDrive extension origin is not the attested pair")
    cuda_code = (
        "import torch; print('1' if torch.cuda.is_available() else '0')"
    )
    cuda_result = _run_setup_command(
        runner,
        (str(python), "-B", "-c", cuda_code),
        "cuda_runtime",
        cwd=candidate,
        env=child_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    cuda_value = _setup_stdout(
        cuda_result, "cuda_runtime", "candidate torch CUDA probe"
    ).strip()
    if cuda_value not in {"0", "1"}:
        _raise("cuda_runtime", "candidate torch CUDA probe returned invalid output")
    runtime_warning: Optional[str] = None
    if cuda_value == "0":
        if preflight.definition.cuda_policy in {"required", "extension_required"}:
            _raise(
                "cuda_runtime",
                f"candidate torch CUDA is required by {preflight.definition.model_id}",
            )
        runtime_warning = (
            "candidate torch CUDA is not available; continuing because "
            f"{preflight.definition.model_id} declares cuda_policy=recommended"
        )
    return runtime_warning


def _run_selected_model_load(
    preflight: EnvironmentPreflight,
    candidate: Path,
    child_environment: Mapping[str, str],
    runner: Callable[..., Any],
) -> None:
    """선택 profile의 실제 model wrapper load를 별도 strict 단계로 실행한다."""
    python = candidate / "bin" / "python"
    profile_path = preflight.root / "src" / "Validator_pkg" / "config" / "profile.json"
    model_code = (
        "import pathlib, sys; "
        "from validator_inference.profile_store import load_selected_profile; "
        "from validator_inference.models.select import build_selected_model; "
        "root = pathlib.Path(sys.argv[1]); profile = pathlib.Path(sys.argv[2]); "
        "selected = load_selected_profile(profile, sys.argv[3]); "
        "build_selected_model(selected, root).load()"
    )
    _run_setup_command(
        runner,
        (
            str(python),
            "-B",
            "-c",
            model_code,
            str(preflight.root),
            str(profile_path),
            preflight.definition.model_id,
        ),
        "model_load",
        cwd=preflight.root,
        env=child_environment,
        timeout=_MODEL_LOAD_TIMEOUT_SECONDS,
    )


def _run_candidate_runtime_checks(
    preflight: EnvironmentPreflight,
    candidate: Path,
    child_environment: Mapping[str, str],
    runner: Callable[..., Any],
) -> Optional[str]:
    """required imports/CUDA를 확인한 뒤 strict selected-model load를 실행한다."""
    runtime_warning = _run_candidate_import_cuda_checks(
        preflight, candidate, child_environment, runner
    )
    _run_selected_model_load(preflight, candidate, child_environment, runner)
    return runtime_warning


def _installed_packages_sha256(
    preflight: EnvironmentPreflight,
    candidate: Path,
    child_environment: Mapping[str, str],
    runner: Callable[..., Any],
) -> str:
    """exact candidate의 uv freeze를 sorted LF로 정규화해 SHA-256을 반환한다."""
    result = _run_setup_command(
        runner,
        (
            str(preflight.uv_path),
            "pip",
            "freeze",
            "--python",
            str(candidate / "bin" / "python"),
            "--strict",
            "--no-python-downloads",
        ),
        "package_freeze",
        cwd=preflight.root,
        env=child_environment,
        capture_output=True,
        timeout=_SETUP_COMMAND_TIMEOUT_SECONDS,
    )
    output = _setup_stdout(result, "package_freeze", "uv pip freeze")
    lines = sorted(line.strip() for line in output.splitlines() if line.strip())
    normalized = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True)
class _DeepVerificationResult:
    """deep verification의 runtime warning과 안정된 input fingerprint를 보관한다."""

    runtime_warning: Optional[str]
    fingerprints: tuple[AssetFingerprint, ...]


def _fingerprint_content_identity(
    fingerprints: Sequence[AssetFingerprint],
) -> tuple[tuple[Path, str], ...]:
    """metadata drift와 독립적인 ordered path/SHA identity를 반환한다."""
    return tuple((fingerprint.path, fingerprint.sha256) for fingerprint in fingerprints)


def _deep_verify_environment(
    preflight: EnvironmentPreflight,
    environment_dir: Path,
    state: EnvironmentState,
    catkin_python: Path,
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
) -> _DeepVerificationResult:
    """fast-valid current의 overlay/assets/interpreter/model/freeze를 deep 검증한다."""
    if state.preflight_ids != _SETUP_PREFLIGHT_IDS:
        _raise(
            "reuse_verification",
            "environment state preflight identifiers do not match the current contract",
        )
    metadata = _call_asset_verifier(preflight, dependencies)
    checkout = _source_checkout_from_metadata(preflight, metadata)
    generated_paths = _validate_state_asset_contract(
        preflight,
        checkout,
        state,
        action="reuse_verification",
    )
    allowed_source_paths = _source_relative_generated_paths(
        preflight,
        checkout,
        generated_paths,
        "reuse_verification",
    )
    _validate_source_checkout(
        preflight,
        metadata,
        dependencies,
        runner,
        allowed_generated_paths=allowed_source_paths,
    )
    fingerprints = _fingerprint_environment_inputs(
        preflight, checkout, generated_paths
    )
    if _fingerprint_content_identity(fingerprints) != _fingerprint_content_identity(
        state.assets
    ):
        _raise("reuse_verification", "runtime asset fingerprint or SHA changed")
    overlay_sha256 = verify_ros_overlay(
        environment_dir / ".ros-overlay",
        environment_dir / ROS_OVERLAY_MANIFEST_FILENAME,
    )
    if overlay_sha256 != state.ros_overlay_sha256:
        _raise("reuse_verification", "ROS overlay SHA changed")
    python = _require_executable_file(
        environment_dir / "bin" / "python", "candidate Python"
    )
    _validate_candidate_python_version(preflight, python, runner)
    child_environment = _build_candidate_child_environment(
        preflight, environment_dir, catkin_python
    )
    _run_ros_smoke(
        preflight, python, child_environment, dependencies, runner
    )
    runtime_warning = _run_candidate_runtime_checks(
        preflight, environment_dir, child_environment, runner
    )
    freeze_sha256 = _installed_packages_sha256(
        preflight, environment_dir, child_environment, runner
    )
    if freeze_sha256 != state.installed_packages_sha256:
        _raise("reuse_verification", "installed package freeze SHA changed")
    final_metadata = _call_asset_verifier(preflight, dependencies)
    final_checkout = _source_checkout_from_metadata(preflight, final_metadata)
    if final_checkout != checkout:
        _raise("reuse_verification", "source checkout changed during verification")
    _validate_source_checkout(
        preflight,
        final_metadata,
        dependencies,
        runner,
        allowed_generated_paths=allowed_source_paths,
    )
    final_fingerprints = _fingerprint_environment_inputs(
        preflight, checkout, generated_paths
    )
    if _fingerprint_content_identity(
        final_fingerprints
    ) != _fingerprint_content_identity(state.assets):
        _raise(
            "reuse_verification",
            "runtime or generated artifact changed during verification",
        )
    if final_fingerprints != fingerprints:
        _raise(
            "reuse_verification",
            "runtime or generated artifact changed during verification",
        )
    return _DeepVerificationResult(runtime_warning, final_fingerprints)


def _merge_setup_warnings(
    preflight_warnings: Tuple[str, ...], runtime_warning: Optional[str]
) -> Tuple[str, ...]:
    """host/runtime CUDA warning을 의미상 중복 없이 안정된 순서로 합친다."""
    warnings = list(preflight_warnings)
    if runtime_warning is not None and not any("CUDA" in warning for warning in warnings):
        warnings.append(runtime_warning)
    return tuple(warnings)


def _revalidate_setup_inputs(
    preflight: EnvironmentPreflight,
    dependencies: EnvironmentSetupDependencies,
    runner: Callable[..., Any],
    checkout: Path,
    generated_paths: Sequence[Path],
    expected_fingerprints: tuple[AssetFingerprint, ...],
) -> tuple[AssetFingerprint, ...]:
    """source/runtime/generated snapshot이 setup baseline과 같은지 다시 확인한다."""
    metadata = _call_asset_verifier(preflight, dependencies)
    observed_checkout = _source_checkout_from_metadata(preflight, metadata)
    if observed_checkout != checkout:
        _raise("setup_revalidation", "source checkout changed during setup")
    allowed_source_paths = _source_relative_generated_paths(
        preflight,
        checkout,
        generated_paths,
        "setup_revalidation",
    )
    _validate_source_checkout(
        preflight,
        metadata,
        dependencies,
        runner,
        allowed_generated_paths=allowed_source_paths,
    )
    fingerprints = _fingerprint_environment_inputs(
        preflight, checkout, generated_paths
    )
    if fingerprints != expected_fingerprints:
        _raise(
            "setup_revalidation",
            "runtime or generated artifact changed during setup",
        )
    return fingerprints


def _create_environment_state(
    preflight: EnvironmentPreflight,
    overlay_sha256: str,
    installed_packages_sha256: str,
    assets: tuple[AssetFingerprint, ...],
    completed_at: str,
) -> EnvironmentState:
    """검증 완료 값만으로 canonical EnvironmentState를 구성한다."""
    return EnvironmentState(
        schema=ENVIRONMENT_STATE_SCHEMA,
        model_id=preflight.definition.model_id,
        python_version=preflight.definition.python_version,
        python_platform=EXPECTED_PYTHON_PLATFORM,
        uv_version=EXPECTED_UV_VERSION,
        manifest_sha256=sha256_file(preflight.manifest.manifest_path),
        lock_sha256=preflight.definition.requirements_lock_sha256,
        ros_overlay_sha256=overlay_sha256,
        source_commit=preflight.definition.source_commit,
        installed_packages_sha256=installed_packages_sha256,
        python_executable=Path("bin") / "python",
        assets=assets,
        completed_at=completed_at,
        preflight_ids=_SETUP_PREFLIGHT_IDS,
    )


def setup_environment(
    project_root: Path,
    model_id: str,
    recreate: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    *,
    dependencies: Optional[EnvironmentSetupDependencies] = None,
) -> EnvironmentOperationResult:
    """모델별 candidate를 완전 검증한 뒤 state-last 방식으로 원자 활성화한다."""
    selected_dependencies = dependencies or EnvironmentSetupDependencies()
    if not isinstance(selected_dependencies, EnvironmentSetupDependencies):
        _raise("setup", "dependencies must be EnvironmentSetupDependencies")
    if type(recreate) is not bool:
        _raise("setup", "recreate must be a boolean")
    preflight = preflight_environment(
        project_root,
        model_id,
        runner=runner,
        dependencies=selected_dependencies.preflight_dependencies,
    )
    with model_operation_lock(preflight.paths):
        catkin_python = _build_catkin_messages(
            preflight, selected_dependencies, runner
        )
        manifest_sha256 = sha256_file(preflight.manifest.manifest_path)
        active = inspect_active_environment(
            preflight.paths, preflight.definition, manifest_sha256
        )
        if not recreate:
            if active.valid:
                if active.environment_dir is None or active.state is None:
                    _raise("reuse_verification", "active inspection lost environment state")
                verification = _deep_verify_environment(
                    preflight,
                    active.environment_dir,
                    active.state,
                    catkin_python,
                    selected_dependencies,
                    runner,
                )
                return EnvironmentOperationResult(
                    environment_dir=active.environment_dir,
                    state=active.state,
                    warnings=_merge_setup_warnings(
                        preflight.warnings, verification.runtime_warning
                    ),
                    changed=False,
                )

        diffusion_transaction = _begin_diffusion_derivative_transaction(
            preflight, active.state
        )
        try:
            prepare_local_runtime_assets(preflight)
            _call_asset_fetcher(preflight, selected_dependencies)
            if diffusion_transaction is not None:
                diffusion_transaction.capture_output()
            asset_metadata = _call_asset_verifier(preflight, selected_dependencies)
            checkout = _source_checkout_from_metadata(preflight, asset_metadata)
            sparse_baseline: Optional[_SparseGeneratedBaseline] = None
            if preflight.definition.model_id == "diffusiondrive_v2":
                initial_generated_paths = (
                    _diffusion_generated_path(preflight, checkout),
                )
            elif preflight.definition.model_id == "sparsedrive_v2":
                initial_generated_paths = _existing_sparse_generated_paths(
                    preflight, checkout
                )
                sparse_baseline = _authenticate_existing_sparse_generated_paths(
                    preflight,
                    checkout,
                    initial_generated_paths,
                    active.state,
                )
            else:
                initial_generated_paths = ()
            initial_allowed_source_paths = _source_relative_generated_paths(
                preflight,
                checkout,
                initial_generated_paths,
                "source_verification",
            )
            _validate_source_checkout(
                preflight,
                asset_metadata,
                selected_dependencies,
                runner,
                allowed_generated_paths=initial_allowed_source_paths,
            )
            runtime_fingerprints = fingerprint_runtime_assets(preflight)
            initial_generated_fingerprints = (
                _fingerprint_generated_artifacts(preflight, checkout)
                if preflight.definition.model_id == "diffusiondrive_v2"
                else ()
            )
            try:
                build_id = selected_dependencies.build_id()
            except (RuntimeError, ValueError, TypeError) as exc:
                raise EnvironmentSetupError(
                    "environment_create", "build ID failed"
                ) from exc
            candidate = create_candidate(preflight.paths, build_id)
            if diffusion_transaction is not None:
                diffusion_transaction.bind_candidate(candidate)
            _run_setup_command(
                runner,
                build_uv_venv_argv(
                    preflight.uv_path, preflight.definition, candidate
                ),
                "environment_create",
                cwd=preflight.root,
            )
            _run_setup_command(
                runner,
                build_uv_sync_argv(
                    preflight.uv_path, preflight.definition, candidate
                ),
                "environment_sync",
                cwd=preflight.root,
            )
            candidate_python = _require_executable_file(
                candidate / "bin" / "python", "candidate Python"
            )
            _validate_candidate_python_version(preflight, candidate_python, runner)
        except BaseException as error:
            if diffusion_transaction is not None:
                try:
                    diffusion_transaction.rollback()
                except BaseException as rollback_error:
                    raise rollback_error from error
                finally:
                    diffusion_transaction.close()
            raise
        sparse_transaction: Optional[_SparseExtensionTransaction] = None
        cleanup_warnings: list[str] = []
        try:
            sparse_transaction = _begin_sparse_extension_transaction(
                preflight, checkout, candidate, sparse_baseline
            )
            sparse_generated_fingerprints = _run_sparse_setup_action(
                preflight,
                candidate_python,
                selected_dependencies,
                runner,
                transaction=sparse_transaction,
            )
            overlay = candidate / ".ros-overlay"
            build_ros_overlay(
                selected_dependencies.preflight_dependencies.ros_python_root,
                overlay,
            )
            overlay_sha256 = verify_ros_overlay(
                overlay, candidate / ROS_OVERLAY_MANIFEST_FILENAME
            )
            child_environment = _build_candidate_child_environment(
                preflight, candidate, catkin_python
            )
            _run_ros_smoke(
                preflight,
                candidate_python,
                child_environment,
                selected_dependencies,
                runner,
            )
            runtime_warning = _run_candidate_runtime_checks(
                preflight, candidate, child_environment, runner
            )
            installed_packages_sha256 = _installed_packages_sha256(
                preflight, candidate, child_environment, runner
            )
            expected_fingerprints = (
                runtime_fingerprints
                + initial_generated_fingerprints
                + sparse_generated_fingerprints
            )
            generated_paths = tuple(
                fingerprint.path
                for fingerprint in (
                    initial_generated_fingerprints + sparse_generated_fingerprints
                )
            )
            final_fingerprints = _revalidate_setup_inputs(
                preflight,
                selected_dependencies,
                runner,
                checkout,
                generated_paths,
                expected_fingerprints,
            )
            try:
                completed_at = selected_dependencies.utc_now()
            except (RuntimeError, ValueError, TypeError) as exc:
                raise EnvironmentSetupError(
                    "state_write", "UTC timestamp failed"
                ) from exc
            state = _create_environment_state(
                preflight,
                overlay_sha256,
                installed_packages_sha256,
                final_fingerprints,
                completed_at,
            )
            state_path = candidate / ENVIRONMENT_STATE_FILENAME
            try:
                write_environment_state(state_path, state)
                observed_state = read_environment_state(candidate)
            except (EnvironmentStateError, OSError) as exc:
                raise EnvironmentSetupError("state_write", str(exc)) from exc
            if observed_state != state:
                _raise("state_write", "environment state readback mismatch")
            try:
                _revalidate_setup_inputs(
                    preflight,
                    selected_dependencies,
                    runner,
                    checkout,
                    generated_paths,
                    expected_fingerprints,
                )
            except BaseException:
                try:
                    state_path.unlink()
                    _fsync_directory(
                        candidate, "state_write", "candidate directory"
                    )
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise EnvironmentSetupError(
                        "state_write", "cannot remove failed candidate state"
                    ) from exc
                raise
            for transaction in (sparse_transaction, diffusion_transaction):
                if transaction is not None:
                    transaction.prepare_commit()
            activate_candidate(preflight.paths, candidate)
            for transaction in (sparse_transaction, diffusion_transaction):
                if transaction is not None:
                    transaction.prepare_commit()
        except BaseException as error:
            rollback_error: Optional[BaseException] = None
            for transaction in (sparse_transaction, diffusion_transaction):
                if transaction is None:
                    continue
                try:
                    transaction.rollback()
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc
            if rollback_error is not None:
                raise rollback_error from error
            raise
        else:
            for transaction in (sparse_transaction, diffusion_transaction):
                if transaction is not None:
                    try:
                        transaction.finalize_commit()
                    except EnvironmentSetupError as exc:
                        cleanup_warnings.append(
                            "generated artifact backup cleanup deferred: " + str(exc)
                        )
        finally:
            for transaction in (sparse_transaction, diffusion_transaction):
                if transaction is not None:
                    transaction.close()
        return EnvironmentOperationResult(
            environment_dir=candidate,
            state=state,
            warnings=(
                _merge_setup_warnings(preflight.warnings, runtime_warning)
                + tuple(cleanup_warnings)
            ),
            changed=True,
        )


def verify_environment(
    project_root: Path,
    model_id: str,
    runner: Callable[..., Any] = subprocess.run,
    *,
    dependencies: Optional[EnvironmentSetupDependencies] = None,
) -> EnvironmentOperationResult:
    """active environment를 full hash/load/freeze 검증하고 metadata drift만 갱신한다."""
    selected_dependencies = dependencies or EnvironmentSetupDependencies()
    if not isinstance(selected_dependencies, EnvironmentSetupDependencies):
        _raise("verify_environment", "dependencies must be EnvironmentSetupDependencies")
    preflight = preflight_environment(
        project_root,
        model_id,
        runner=runner,
        dependencies=selected_dependencies.preflight_dependencies,
    )
    with _model_operation_lock(
        preflight.paths, "verify_environment", create_root=False
    ):
        manifest_sha256 = sha256_file(preflight.manifest.manifest_path)
        active = inspect_active_environment_for_verification(
            preflight.paths, preflight.definition, manifest_sha256
        )
        if not active.valid or active.environment_dir is None or active.state is None:
            _raise("verify_environment", active.reason)
        catkin_python = _validate_catkin_messages_read_only(
            preflight, selected_dependencies, runner
        )
        verification = _deep_verify_environment(
            preflight,
            active.environment_dir,
            active.state,
            catkin_python,
            selected_dependencies,
            runner,
        )
        refreshed = replace(active.state, assets=verification.fingerprints)
        changed = refreshed != active.state
        if changed:
            state_path = active.environment_dir / ENVIRONMENT_STATE_FILENAME
            try:
                write_environment_state(state_path, refreshed)
                observed = read_environment_state(active.environment_dir)
            except (EnvironmentStateError, OSError) as exc:
                raise EnvironmentSetupError("verify_environment", str(exc)) from exc
            if observed != refreshed:
                _raise("verify_environment", "refreshed environment state readback mismatch")
            final = inspect_active_environment(
                preflight.paths, preflight.definition, manifest_sha256
            )
            if (
                not final.valid
                or final.environment_dir != active.environment_dir
                or final.state != refreshed
            ):
                _raise("verify_environment", "environment changed during state refresh")
        return EnvironmentOperationResult(
            environment_dir=active.environment_dir,
            state=refreshed,
            warnings=_merge_setup_warnings(
                preflight.warnings, verification.runtime_warning
            ),
            changed=changed,
        )


def prune_environment_builds(
    project_root: Path, model_id: str
) -> tuple[Path, ...]:
    """host/profile probe 없이 검증된 model path의 inactive builds만 제거한다."""
    if model_id not in SUPPORTED_MODEL_IDS:
        _raise("prune_environment_builds", f"unsupported model_id: {model_id}")
    root = _canonical_absolute_posix_path(
        project_root, "project root", "prune_environment_builds"
    )
    _require_real_directory(root, "prune_environment_builds", "project root")
    return prune_inactive_builds(environment_paths(root, model_id))


def preflight_run_environment(
    project_root: Path,
    model_id: str,
    runner: Callable[..., Any] = subprocess.run,
    *,
    dependencies: Optional[EnvironmentSetupDependencies] = None,
) -> EnvironmentOperationResult:
    """install/hash/model-load 없이 현재 environment의 실제 runtime readiness를 검사한다."""
    selected_dependencies = dependencies or EnvironmentSetupDependencies()
    if not isinstance(selected_dependencies, EnvironmentSetupDependencies):
        _raise("runtime_preflight", "dependencies must be EnvironmentSetupDependencies")
    preflight = preflight_environment(
        project_root,
        model_id,
        runner=runner,
        dependencies=selected_dependencies.preflight_dependencies,
    )
    with _model_operation_lock(
        preflight.paths, "runtime_preflight", create_root=False
    ):
        manifest_sha256 = sha256_file(preflight.manifest.manifest_path)
        active = inspect_active_environment(
            preflight.paths, preflight.definition, manifest_sha256
        )
        if not active.valid or active.environment_dir is None or active.state is None:
            _raise("runtime_preflight", active.reason)
        if active.state.preflight_ids != _SETUP_PREFLIGHT_IDS:
            _raise("runtime_preflight", "environment preflight identifiers are stale")
        checkout = _expected_source_checkout(preflight, "runtime_preflight")
        generated_paths = _validate_state_asset_contract(
            preflight,
            checkout,
            active.state,
            action="runtime_preflight",
        )
        allowed_source_paths = _source_relative_generated_paths(
            preflight,
            checkout,
            generated_paths,
            "runtime_preflight",
        )
        metadata = {
            "source": {
                "checkout": str(checkout),
                "observed_commit": preflight.definition.source_commit,
            }
        }
        _validate_source_checkout(
            preflight,
            metadata,
            selected_dependencies,
            runner,
            allowed_generated_paths=allowed_source_paths,
        )
        catkin_python = _validate_catkin_messages_read_only(
            preflight, selected_dependencies, runner
        )
        python = _require_executable_file(
            active.environment_dir / "bin" / "python", "candidate Python"
        )
        _validate_candidate_python_version(preflight, python, runner)
        child_environment = _build_candidate_child_environment(
            preflight, active.environment_dir, catkin_python
        )
        _run_target_ros_import_smoke(
            python, active.environment_dir, child_environment, runner
        )
        runtime_warning = _run_candidate_import_cuda_checks(
            preflight, active.environment_dir, child_environment, runner
        )
        return EnvironmentOperationResult(
            environment_dir=active.environment_dir,
            state=active.state,
            warnings=_merge_setup_warnings(preflight.warnings, runtime_warning),
            changed=False,
        )
