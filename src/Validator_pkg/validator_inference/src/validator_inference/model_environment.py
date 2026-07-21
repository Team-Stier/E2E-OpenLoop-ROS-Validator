"""
모델별 Python 환경 manifest를 읽고 재현 가능한 명령 계약을 구성한다.
입력: 추적된 환경 JSON, project root, 모델 ID와 후보 환경 경로.
출력(pub): 없음. 검증된 환경 정의, 경로, hash와 uv argv를 반환한다.
구독(sub): 없음.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit


SUPPORTED_MODEL_IDS = (
    "interfuser",
    "tfv6",
    "diffusiondrive_v2",
    "sparsedrive_v2",
)
ENVIRONMENT_STATE_FILENAME = ".validator-environment.json"
ROS_OVERLAY_MANIFEST_FILENAME = ".ros-overlay-manifest.json"
ENVIRONMENT_STATE_SCHEMA = 1
EXPECTED_PYTHON_PLATFORM = "x86_64-manylinux_2_31"
EXPECTED_UV_VERSION = "0.11.14"
_TOP_LEVEL_KEYS = {
    "schema",
    "uv_version",
    "python_platform",
    "exclude_newer",
    "default_index",
    "index_strategy",
    "models",
}
_MODEL_KEYS = {
    "python_version",
    "requirements_input",
    "requirements_input_sha256",
    "requirements_lock",
    "requirements_lock_sha256",
    "indexes",
    "torch_backend",
    "cuda_policy",
    "source_commit",
    "required_imports",
    "setup_actions",
}
_CUDA_POLICIES = {"required", "recommended", "extension_required"}
_PATCH_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMPORT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_TORCH_BACKEND = re.compile(r"cu[0-9]{3}\Z")
_SETUP_ACTION_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SETUP_ACTIONS = {"build_sparsedrive_cuda_extensions"}
_PREFLIGHT_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_STATE_KEYS = {
    "schema",
    "model_id",
    "python_version",
    "python_platform",
    "uv_version",
    "manifest_sha256",
    "lock_sha256",
    "ros_overlay_sha256",
    "source_commit",
    "installed_packages_sha256",
    "python_executable",
    "assets",
    "completed_at",
    "preflight_ids",
}
_ASSET_KEYS = {"path", "sha256", "size", "mtime_ns"}


class EnvironmentConfigError(ValueError):
    """추적된 환경 정의가 지원 계약을 위반할 때 발생한다."""


class EnvironmentStateError(ValueError):
    """로컬 환경 state 파일이 손상되거나 불완전할 때 발생한다."""


@dataclass(frozen=True)
class ModelEnvironmentDefinition:
    """한 모델의 interpreter, lock, index와 사전 준비 계약을 보관한다."""

    model_id: str
    python_version: str
    requirements_input: Path
    requirements_input_sha256: str
    requirements_lock: Path
    requirements_lock_sha256: str
    indexes: tuple[str, ...]
    torch_backend: str
    cuda_policy: str
    source_commit: str
    required_imports: tuple[str, ...]
    setup_actions: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentManifest:
    """공통 resolver 설정과 네 모델의 불변 환경 정의를 보관한다."""

    schema: int
    uv_version: str
    python_platform: str
    exclude_newer: str
    default_index: str
    index_strategy: str
    models: Mapping[str, ModelEnvironmentDefinition]
    manifest_path: Path


@dataclass(frozen=True)
class EnvironmentPaths:
    """한 모델이 소유하는 root, versioned builds와 active symlink 경로다."""

    model_root: Path
    builds: Path
    current: Path


@dataclass(frozen=True)
class AssetFingerprint:
    """검증한 runtime asset의 경로, hash, 크기와 수정 시각을 보관한다."""

    path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class EnvironmentState:
    """완료된 모델 환경의 interpreter, 정의 hash와 검증 결과를 보관한다."""

    schema: int
    model_id: str
    python_version: str
    python_platform: str
    uv_version: str
    manifest_sha256: str
    lock_sha256: str
    ros_overlay_sha256: str
    source_commit: str
    installed_packages_sha256: str
    python_executable: Path
    assets: tuple[AssetFingerprint, ...]
    completed_at: str
    preflight_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActiveEnvironmentCheck:
    """active 환경의 사용 가능 여부와 한 가지 진단 이유를 보관한다."""

    valid: bool
    reason: str
    environment_dir: Optional[Path]
    state: Optional[EnvironmentState]


def _reject_duplicate_state_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """state JSON object의 중복 key를 손상으로 분류해 거부한다."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EnvironmentStateError(f"duplicate key: {key}")
        result[key] = value
    return result


def _require_state_object(value: Any, context: str) -> dict[str, Any]:
    """state 값이 JSON object인지 확인한다."""
    if not isinstance(value, dict):
        raise EnvironmentStateError(f"{context} must be an object")
    return value


def _require_state_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    """state object의 누락 key와 알려지지 않은 key를 거부한다."""
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise EnvironmentStateError(
            f"{context} missing keys: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise EnvironmentStateError(
            f"{context} unknown keys: {', '.join(sorted(unknown))}"
        )


def _require_state_string(value: Any, context: str) -> str:
    """state 값이 비어 있지 않은 문자열인지 확인한다."""
    if not isinstance(value, str) or not value:
        raise EnvironmentStateError(f"{context} must be a nonempty string")
    return value


def _require_state_sha256(value: Any, context: str) -> str:
    """state digest가 lowercase SHA-256 형식인지 확인한다."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EnvironmentStateError(
            f"{context} must be a lowercase 64-digit SHA-256"
        )
    return value


def _require_safe_state_path(value: Any, context: str) -> Path:
    """state의 repository 상대 POSIX 경로가 탈출하거나 변형되지 않게 한다."""
    raw_path = _require_state_string(value, context)
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in raw_path
        or relative.as_posix() != raw_path
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw_path)
    ):
        raise EnvironmentStateError(f"{context} must be a safe relative POSIX path")
    return Path(*relative.parts)


def _parse_asset_fingerprint(value: Any, index: int) -> AssetFingerprint:
    """한 asset fingerprint object를 엄격히 검증한다."""
    context = f"state.assets[{index}]"
    payload = _require_state_object(value, context)
    _require_state_exact_keys(payload, _ASSET_KEYS, context)
    size = payload["size"]
    mtime_ns = payload["mtime_ns"]
    if type(size) is not int or size < 0:
        raise EnvironmentStateError("state.assets.size must be a nonnegative integer")
    if type(mtime_ns) is not int or mtime_ns < 0:
        raise EnvironmentStateError("state.assets.mtime_ns must be a nonnegative integer")
    return AssetFingerprint(
        path=_require_safe_state_path(payload["path"], "state.assets.path"),
        sha256=_require_state_sha256(payload["sha256"], "state.assets.sha256"),
        size=size,
        mtime_ns=mtime_ns,
    )


def _parse_environment_state(value: Any) -> EnvironmentState:
    """decoded JSON object를 검증된 불변 환경 state로 변환한다."""
    payload = _require_state_object(value, "state")
    _require_state_exact_keys(payload, _STATE_KEYS, "state")
    if type(payload["schema"]) is not int or payload["schema"] != ENVIRONMENT_STATE_SCHEMA:
        raise EnvironmentStateError("state schema must be integer 1")

    model_id = _require_state_string(payload["model_id"], "state.model_id")
    if model_id not in SUPPORTED_MODEL_IDS:
        raise EnvironmentStateError("state.model_id is unsupported")
    python_version = _require_state_string(
        payload["python_version"], "state.python_version"
    )
    if not _PATCH_VERSION.fullmatch(python_version):
        raise EnvironmentStateError("state.python_version must be an exact patch version")
    python_platform = _require_state_string(
        payload["python_platform"], "state.python_platform"
    )
    if python_platform != EXPECTED_PYTHON_PLATFORM:
        raise EnvironmentStateError(
            f"state.python_platform must be {EXPECTED_PYTHON_PLATFORM}"
        )
    uv_version = _require_state_string(payload["uv_version"], "state.uv_version")
    if uv_version != EXPECTED_UV_VERSION:
        raise EnvironmentStateError(f"state.uv_version must be {EXPECTED_UV_VERSION}")

    source_commit = payload["source_commit"]
    if not isinstance(source_commit, str) or not _GIT_SHA.fullmatch(source_commit):
        raise EnvironmentStateError(
            "state.source_commit must be a lowercase 40-digit SHA"
        )
    completed_at = _require_state_string(payload["completed_at"], "state.completed_at")
    if not _UTC_TIMESTAMP.fullmatch(completed_at):
        raise EnvironmentStateError("state.completed_at must be a UTC second timestamp")
    try:
        datetime.strptime(completed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise EnvironmentStateError(
            "state.completed_at must be a valid UTC second timestamp"
        ) from exc

    assets_value = payload["assets"]
    if not isinstance(assets_value, list):
        raise EnvironmentStateError("state.assets must be an array")
    assets = tuple(
        _parse_asset_fingerprint(asset, index)
        for index, asset in enumerate(assets_value)
    )
    asset_paths = [asset.path.as_posix() for asset in assets]
    if len(set(asset_paths)) != len(asset_paths):
        raise EnvironmentStateError("state contains a duplicate asset path")

    preflight_value = payload["preflight_ids"]
    if not isinstance(preflight_value, list):
        raise EnvironmentStateError("state.preflight_ids must be an array")
    if any(
        not isinstance(identifier, str) or not _PREFLIGHT_ID.fullmatch(identifier)
        for identifier in preflight_value
    ):
        raise EnvironmentStateError(
            "state.preflight_ids must contain lowercase identifiers"
        )
    if len(set(preflight_value)) != len(preflight_value):
        raise EnvironmentStateError("state.preflight_ids must not contain duplicates")

    return EnvironmentState(
        schema=ENVIRONMENT_STATE_SCHEMA,
        model_id=model_id,
        python_version=python_version,
        python_platform=python_platform,
        uv_version=uv_version,
        manifest_sha256=_require_state_sha256(
            payload["manifest_sha256"], "state.manifest_sha256"
        ),
        lock_sha256=_require_state_sha256(
            payload["lock_sha256"], "state.lock_sha256"
        ),
        ros_overlay_sha256=_require_state_sha256(
            payload["ros_overlay_sha256"], "state.ros_overlay_sha256"
        ),
        source_commit=source_commit,
        installed_packages_sha256=_require_state_sha256(
            payload["installed_packages_sha256"],
            "state.installed_packages_sha256",
        ),
        python_executable=_require_safe_state_path(
            payload["python_executable"], "state.python_executable"
        ),
        assets=assets,
        completed_at=completed_at,
        preflight_ids=tuple(preflight_value),
    )


def _environment_state_payload(state: EnvironmentState) -> dict[str, Any]:
    """환경 state를 deterministic JSON용 기본 자료형으로 변환한다."""
    return {
        "schema": state.schema,
        "model_id": state.model_id,
        "python_version": state.python_version,
        "python_platform": state.python_platform,
        "uv_version": state.uv_version,
        "manifest_sha256": state.manifest_sha256,
        "lock_sha256": state.lock_sha256,
        "ros_overlay_sha256": state.ros_overlay_sha256,
        "source_commit": state.source_commit,
        "installed_packages_sha256": state.installed_packages_sha256,
        "python_executable": state.python_executable.as_posix(),
        "assets": [
            {
                "path": asset.path.as_posix(),
                "sha256": asset.sha256,
                "size": asset.size,
                "mtime_ns": asset.mtime_ns,
            }
            for asset in state.assets
        ],
        "completed_at": state.completed_at,
        "preflight_ids": list(state.preflight_ids),
    }


def read_environment_state(environment_dir: Path) -> EnvironmentState:
    """선택 build의 고정 state 파일을 읽어 엄격히 검증한다."""
    state_path = Path(environment_dir) / ENVIRONMENT_STATE_FILENAME
    try:
        metadata = state_path.lstat()
    except OSError as exc:
        raise EnvironmentStateError(f"cannot read environment state: {state_path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise EnvironmentStateError(
            f"environment state must be a regular non-symlink file: {state_path}"
        )
    try:
        payload = json.loads(
            state_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_state_keys,
        )
    except EnvironmentStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EnvironmentStateError(f"cannot read environment state: {state_path}") from exc
    return _parse_environment_state(payload)


def write_environment_state(path: Path, state: EnvironmentState) -> None:
    """검증한 state JSON을 sibling 임시 파일에서 원자적으로 교체한다."""
    state_path = Path(path)
    payload = _environment_state_payload(state)
    parsed = _parse_environment_state(payload)
    if parsed != state:
        raise EnvironmentStateError("environment state is not canonically serializable")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{state_path.name}.", suffix=".tmp", dir=str(state_path.parent)
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            descriptor_open = False
            target.write(serialized)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, state_path)
        directory_descriptor = os.open(
            str(state_path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object의 중복 key를 허용하지 않고 일반 dict로 변환한다."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EnvironmentConfigError(f"duplicate key: {key}")
        result[key] = value
    return result


def _require_object(value: Any, context: str) -> dict[str, Any]:
    """값이 JSON object인지 확인하고 typed dict로 반환한다."""
    if not isinstance(value, dict):
        raise EnvironmentConfigError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    """JSON object의 누락 key와 알려지지 않은 key를 각각 거부한다."""
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise EnvironmentConfigError(
            f"{context} missing keys: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise EnvironmentConfigError(
            f"{context} unknown keys: {', '.join(sorted(unknown))}"
        )


def _require_string(value: Any, context: str) -> str:
    """값이 비어 있지 않은 문자열인지 확인한다."""
    if not isinstance(value, str) or not value:
        raise EnvironmentConfigError(f"{context} must be a nonempty string")
    return value


def _require_string_tuple(
    value: Any, context: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    """JSON 문자열 배열을 중복 없는 tuple로 정규화한다."""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise EnvironmentConfigError(f"{context} must be a string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise EnvironmentConfigError(f"{context} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise EnvironmentConfigError(f"{context} must not contain duplicates")
    return tuple(value)


def _validate_index(value: str, context: str) -> None:
    """package index가 credential 없는 고정 HTTPS URL인지 검사한다."""
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value):
        raise EnvironmentConfigError(
            f"{context} contains an ASCII whitespace or control character"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise EnvironmentConfigError(
            f"{context} contains a malformed index URL"
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EnvironmentConfigError(f"{context} contains a malformed index URL")


def _resolve_requirement_path(
    package_root: Path,
    value: Any,
    context: str,
    expected_relative: str,
) -> Path:
    """package root 상대 requirement 경로의 탈출을 막고 존재를 확인한다."""
    raw_path = _require_string(value, context)
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in raw_path
        or relative.as_posix() != raw_path
        or raw_path != expected_relative
    ):
        raise EnvironmentConfigError(f"{context} must be a safe package-relative path")
    resolved = package_root.joinpath(*relative.parts)
    if not resolved.is_file():
        raise EnvironmentConfigError(f"{context} does not exist: {resolved}")
    resolved = resolved.resolve()
    try:
        resolved.relative_to(package_root.resolve())
    except ValueError as exc:
        raise EnvironmentConfigError(
            f"{context} must resolve beneath the package root"
        ) from exc
    return resolved


def _require_sha256(value: Any, context: str) -> str:
    """manifest digest가 정확한 lowercase SHA-256인지 확인한다."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EnvironmentConfigError(
            f"{context} must be a lowercase 64-digit SHA-256"
        )
    return value


def _parse_model_definition(
    package_root: Path,
    default_index: str,
    model_id: str,
    value: Any,
) -> ModelEnvironmentDefinition:
    """한 model object를 엄격히 검증해 typed 환경 정의로 변환한다."""
    payload = _require_object(value, f"models.{model_id}")
    _require_exact_keys(payload, _MODEL_KEYS, f"models.{model_id}")

    python_version = _require_string(
        payload["python_version"], f"models.{model_id}.python_version"
    )
    if not _PATCH_VERSION.fullmatch(python_version):
        raise EnvironmentConfigError(
            f"models.{model_id}.python_version must be an exact patch version"
        )
    cuda_policy = _require_string(
        payload["cuda_policy"], f"models.{model_id}.cuda_policy"
    )
    if cuda_policy not in _CUDA_POLICIES:
        raise EnvironmentConfigError(f"models.{model_id}.cuda_policy is unsupported")

    source_commit = payload["source_commit"]
    if not isinstance(source_commit, str) or not _GIT_SHA.fullmatch(source_commit):
        raise EnvironmentConfigError(
            f"models.{model_id}.source_commit must be a lowercase 40-digit SHA"
        )

    indexes = _require_string_tuple(payload["indexes"], f"models.{model_id}.indexes")
    for index in indexes:
        _validate_index(index, f"models.{model_id}.indexes")
    if (
        len(indexes) != 2
        or indexes[1] != default_index
        or urlsplit(indexes[0]).hostname != "download.pytorch.org"
    ):
        raise EnvironmentConfigError(
            f"models.{model_id}.indexes must contain official CUDA then default index"
        )
    torch_backend = _require_string(
        payload["torch_backend"], f"models.{model_id}.torch_backend"
    )
    if (
        not _TORCH_BACKEND.fullmatch(torch_backend)
        or urlsplit(indexes[0]).path != f"/whl/{torch_backend}"
    ):
        raise EnvironmentConfigError(
            f"models.{model_id}.torch_backend must match its CUDA index"
        )

    required_imports = _require_string_tuple(
        payload["required_imports"], f"models.{model_id}.required_imports"
    )
    if any(not _IMPORT_NAME.fullmatch(name) for name in required_imports):
        raise EnvironmentConfigError(
            f"models.{model_id}.required_imports contains an invalid module name"
        )
    setup_actions = _require_string_tuple(
        payload["setup_actions"],
        f"models.{model_id}.setup_actions",
        allow_empty=True,
    )
    if any(not _SETUP_ACTION_NAME.fullmatch(action) for action in setup_actions):
        raise EnvironmentConfigError(
            f"models.{model_id}.setup_actions contains an invalid identifier"
        )
    unsupported_actions = set(setup_actions) - _SETUP_ACTIONS
    if unsupported_actions:
        raise EnvironmentConfigError(
            f"models.{model_id}.setup_actions contains an unsupported action"
        )

    requirements_input = _resolve_requirement_path(
        package_root,
        payload["requirements_input"],
        f"models.{model_id}.requirements_input",
        f"requirements/{model_id}.in",
    )
    requirements_lock = _resolve_requirement_path(
        package_root,
        payload["requirements_lock"],
        f"models.{model_id}.requirements_lock",
        f"requirements/{model_id}.lock",
    )
    requirements_input_sha256 = _require_sha256(
        payload["requirements_input_sha256"],
        f"models.{model_id}.requirements_input_sha256",
    )
    requirements_lock_sha256 = _require_sha256(
        payload["requirements_lock_sha256"],
        f"models.{model_id}.requirements_lock_sha256",
    )
    if sha256_file(requirements_input) != requirements_input_sha256:
        raise EnvironmentConfigError(
            f"models.{model_id}.requirements_input_sha256 does not match file content"
        )
    if sha256_file(requirements_lock) != requirements_lock_sha256:
        raise EnvironmentConfigError(
            f"models.{model_id}.requirements_lock_sha256 does not match file content"
        )
    return ModelEnvironmentDefinition(
        model_id=model_id,
        python_version=python_version,
        requirements_input=requirements_input,
        requirements_input_sha256=requirements_input_sha256,
        requirements_lock=requirements_lock,
        requirements_lock_sha256=requirements_lock_sha256,
        indexes=indexes,
        torch_backend=torch_backend,
        cuda_policy=cuda_policy,
        source_commit=source_commit,
        required_imports=required_imports,
        setup_actions=setup_actions,
    )


def load_environment_manifest(path: Path) -> EnvironmentManifest:
    """추적된 JSON을 읽어 exact 공통·모델 환경 계약을 검증한다."""
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except EnvironmentConfigError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentConfigError(f"cannot read environment manifest: {path}") from exc

    root = _require_object(payload, "manifest")
    _require_exact_keys(root, _TOP_LEVEL_KEYS, "manifest")
    if type(root["schema"]) is not int or root["schema"] != 1:
        raise EnvironmentConfigError("manifest schema must be integer 1")

    exact_common = {
        "uv_version": "0.11.14",
        "python_platform": "x86_64-manylinux_2_31",
        "exclude_newer": "2026-07-21T00:00:00Z",
        "default_index": "https://pypi.org/simple",
        "index_strategy": "first-index",
    }
    for key, expected in exact_common.items():
        actual = _require_string(root[key], f"manifest.{key}")
        if actual != expected:
            raise EnvironmentConfigError(f"manifest.{key} must be {expected}")
    _validate_index(root["default_index"], "manifest.default_index")

    models_payload = _require_object(root["models"], "manifest.models")
    if set(models_payload) != set(SUPPORTED_MODEL_IDS):
        raise EnvironmentConfigError(
            "manifest model IDs must be exactly: " + ", ".join(SUPPORTED_MODEL_IDS)
        )
    package_root = manifest_path.parent.parent
    models = {
        model_id: _parse_model_definition(
            package_root, root["default_index"], model_id, models_payload[model_id]
        )
        for model_id in SUPPORTED_MODEL_IDS
    }
    return EnvironmentManifest(
        schema=1,
        uv_version=root["uv_version"],
        python_platform=root["python_platform"],
        exclude_newer=root["exclude_newer"],
        default_index=root["default_index"],
        index_strategy=root["index_strategy"],
        models=MappingProxyType(models),
        manifest_path=manifest_path,
    )


def select_model_environment(
    manifest: EnvironmentManifest, model_id: str
) -> ModelEnvironmentDefinition:
    """검증된 manifest에서 지원 model ID의 환경 정의를 선택한다."""
    try:
        return manifest.models[model_id]
    except (KeyError, TypeError) as exc:
        raise EnvironmentConfigError(f"unsupported model_id: {model_id}") from exc


def environment_paths(project_root: Path, model_id: str) -> EnvironmentPaths:
    """project의 .venvs 아래에서 한 모델의 root, builds와 current를 만든다."""
    if model_id not in SUPPORTED_MODEL_IDS:
        raise EnvironmentConfigError(f"unsupported model_id: {model_id}")
    model_root = Path(project_root).resolve() / ".venvs" / model_id
    return EnvironmentPaths(model_root, model_root / "builds", model_root / "current")


def sha256_file(path: Path) -> str:
    """파일 전체를 streaming 방식으로 읽어 lowercase SHA-256을 반환한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_uv_venv_argv(
    uv: Path, definition: ModelEnvironmentDefinition, candidate: Path
) -> tuple[str, ...]:
    """정확한 managed CPython으로 candidate를 만드는 uv argv를 반환한다."""
    return (
        str(uv),
        "venv",
        "--no-project",
        "--managed-python",
        "--python",
        definition.python_version,
        str(candidate),
    )


def build_uv_sync_argv(
    uv: Path, definition: ModelEnvironmentDefinition, candidate: Path
) -> tuple[str, ...]:
    """hash lock을 재해석 없이 candidate에 동기화하는 uv argv를 반환한다."""
    return (
        str(uv),
        "pip",
        "sync",
        "--python",
        str(Path(candidate) / "bin" / "python"),
        "--require-hashes",
        "--strict",
        "--no-python-downloads",
        "--default-index",
        definition.indexes[1],
        "--index-strategy",
        "first-index",
        "--torch-backend",
        definition.torch_backend,
        str(definition.requirements_lock),
    )


def _invalid_active_environment(
    reason: str,
    environment_dir: Optional[Path] = None,
    state: Optional[EnvironmentState] = None,
) -> ActiveEnvironmentCheck:
    """active 환경의 한 가지 실패 이유를 표준 invalid 결과로 만든다."""
    return ActiveEnvironmentCheck(False, reason, environment_dir, state)


def _path_contains_symlink(root: Path, relative: Path) -> bool:
    """root 아래 상대 경로 구성 요소 중 symlink가 있는지 확인한다."""
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def inspect_active_environment(
    paths: EnvironmentPaths,
    definition: ModelEnvironmentDefinition,
    manifest_sha256: str,
) -> ActiveEnvironmentCheck:
    """active build와 state를 변경 없이 빠른 fingerprint 방식으로 검사한다."""
    model_root = Path(paths.model_root)
    builds = Path(paths.builds)
    current = Path(paths.current)
    if (
        not model_root.is_absolute()
        or model_root.name != definition.model_id
        or model_root.parent.name != ".venvs"
        or builds != model_root / "builds"
        or current != model_root / "current"
        or model_root.resolve() != model_root
    ):
        return _invalid_active_environment("environment path layout is invalid")
    project_root = model_root.parent.parent
    if not project_root.is_dir():
        return _invalid_active_environment("project root directory is missing")
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        return _invalid_active_environment("expected manifest SHA-256 is invalid")

    if not current.is_symlink():
        if current.exists():
            return _invalid_active_environment("current must be a symlink")
        return _invalid_active_environment("current environment is missing")
    try:
        environment_dir = current.resolve(strict=True)
    except (OSError, RuntimeError):
        return _invalid_active_environment("current symlink is dangling")

    if not builds.is_dir() or builds.is_symlink():
        return _invalid_active_environment("builds directory is invalid")
    builds_resolved = builds.resolve()
    if environment_dir.parent != builds_resolved:
        if builds_resolved in environment_dir.parents:
            return _invalid_active_environment(
                "current must select an immediate build child", environment_dir
            )
        return _invalid_active_environment(
            "current must select a build inside builds", environment_dir
        )
    if not environment_dir.is_dir():
        return _invalid_active_environment(
            "selected build is not a directory", environment_dir
        )

    python = environment_dir / "bin" / "python"
    if not python.exists():
        return _invalid_active_environment("selected bin/python is missing", environment_dir)
    if not python.is_file():
        return _invalid_active_environment(
            "selected bin/python is not a regular file", environment_dir
        )
    if not os.access(str(python), os.X_OK):
        return _invalid_active_environment(
            "selected bin/python is not executable", environment_dir
        )

    try:
        state_value = read_environment_state(environment_dir)
    except EnvironmentStateError as exc:
        return _invalid_active_environment(
            f"invalid environment state: {exc}", environment_dir
        )
    expected_values = (
        (state_value.schema, ENVIRONMENT_STATE_SCHEMA, "state schema is stale"),
        (state_value.model_id, definition.model_id, "state model is stale"),
        (
            state_value.python_version,
            definition.python_version,
            "environment Python version is stale",
        ),
        (
            state_value.python_platform,
            EXPECTED_PYTHON_PLATFORM,
            "state Python platform is stale",
        ),
        (state_value.uv_version, EXPECTED_UV_VERSION, "state uv version is stale"),
        (
            state_value.manifest_sha256,
            manifest_sha256,
            "environment manifest digest is stale",
        ),
        (
            state_value.lock_sha256,
            definition.requirements_lock_sha256,
            "environment lock digest is stale",
        ),
        (
            state_value.source_commit,
            definition.source_commit,
            "environment source commit is stale",
        ),
    )
    for actual, expected, reason in expected_values:
        if actual != expected:
            return _invalid_active_environment(reason, environment_dir, state_value)
    if state_value.python_executable != Path("bin") / "python":
        return _invalid_active_environment(
            "state python_executable must be bin/python",
            environment_dir,
            state_value,
        )

    overlay_manifest = environment_dir / ROS_OVERLAY_MANIFEST_FILENAME
    if not overlay_manifest.exists():
        return _invalid_active_environment(
            "ROS overlay manifest is missing", environment_dir, state_value
        )
    if overlay_manifest.is_symlink() or not overlay_manifest.is_file():
        return _invalid_active_environment(
            "ROS overlay manifest is not a regular file",
            environment_dir,
            state_value,
        )
    try:
        overlay_sha256 = sha256_file(overlay_manifest)
    except OSError:
        return _invalid_active_environment(
            "ROS overlay manifest cannot be read", environment_dir, state_value
        )
    if overlay_sha256 != state_value.ros_overlay_sha256:
        return _invalid_active_environment(
            "ROS overlay manifest digest is stale", environment_dir, state_value
        )

    project_root_resolved = project_root.resolve()
    for fingerprint in state_value.assets:
        relative = fingerprint.path
        if _path_contains_symlink(project_root, relative):
            return _invalid_active_environment(
                f"asset path contains a symlink: {relative.as_posix()}",
                environment_dir,
                state_value,
            )
        asset_path = project_root.joinpath(*relative.parts)
        try:
            metadata = asset_path.lstat()
        except OSError:
            return _invalid_active_environment(
                f"asset is missing: {relative.as_posix()}",
                environment_dir,
                state_value,
            )
        if not stat.S_ISREG(metadata.st_mode):
            return _invalid_active_environment(
                f"asset is not a regular file: {relative.as_posix()}",
                environment_dir,
                state_value,
            )
        try:
            asset_path.resolve(strict=True).relative_to(project_root_resolved)
        except (OSError, ValueError):
            return _invalid_active_environment(
                f"asset escapes project root: {relative.as_posix()}",
                environment_dir,
                state_value,
            )
        if metadata.st_size != fingerprint.size:
            return _invalid_active_environment(
                f"asset size changed: {relative.as_posix()}",
                environment_dir,
                state_value,
            )
        if metadata.st_mtime_ns != fingerprint.mtime_ns:
            return _invalid_active_environment(
                f"asset mtime_ns changed: {relative.as_posix()}",
                environment_dir,
                state_value,
            )

    return ActiveEnvironmentCheck(True, "ready", environment_dir, state_value)


def _require_absolute_directory(path: Path, context: str) -> Path:
    """process 환경 입력이 존재하는 절대 directory인지 확인한다."""
    value = Path(path)
    if not value.is_absolute():
        raise EnvironmentStateError(f"{context} must be an absolute path")
    if not value.is_dir():
        raise EnvironmentStateError(f"{context} must be an existing directory")
    return value


def _is_activation_variable(name: str) -> bool:
    """상속하면 안 되는 Python, virtualenv와 Conda 활성화 변수를 식별한다."""
    return (
        name in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "_CE_CONDA", "_CE_M"}
        or name.startswith("VIRTUAL_ENV")
        or name.startswith("_OLD_VIRTUAL_")
        or name.startswith("CONDA_")
        or name.startswith("_CONDA")
        or name.startswith("__CONDA")
    )


def _path_entry_key(value: str) -> str:
    """PATH 항목의 중복과 제거 대상을 비교할 정규화 key를 만든다."""
    return os.path.normcase(os.path.normpath(os.path.realpath(value)))


def build_model_process_environment(
    inherited: Mapping[str, str],
    environment_dir: Path,
    ros_overlay: Path,
    catkin_python_path: Path,
    project_root: Path,
) -> dict[str, str]:
    """선택 모델 interpreter만 사용하는 격리된 child process 환경을 만든다."""
    selected_environment = _require_absolute_directory(
        environment_dir, "environment_dir"
    )
    selected_overlay = _require_absolute_directory(ros_overlay, "ros_overlay")
    selected_catkin = _require_absolute_directory(
        catkin_python_path, "catkin_python_path"
    )
    selected_project = _require_absolute_directory(project_root, "project_root")
    selected_bin = selected_environment / "bin"
    selected_python = selected_bin / "python"
    if not selected_python.exists() or not selected_python.is_file():
        raise EnvironmentStateError("selected bin/python must be a regular file")
    if not os.access(str(selected_python), os.X_OK):
        raise EnvironmentStateError("selected bin/python must be executable")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in inherited.items()
    ):
        raise EnvironmentStateError("inherited environment must contain strings")

    foreign_bins: set[str] = set()
    for name, value in inherited.items():
        if name == "VIRTUAL_ENV" or name.startswith("CONDA_PREFIX"):
            if value:
                foreign_bins.add(_path_entry_key(str(Path(value) / "bin")))
    selected_bin_key = _path_entry_key(str(selected_bin))
    retained_path: list[str] = []
    retained_keys: set[str] = set()
    for entry in inherited.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        key = _path_entry_key(entry)
        if key == selected_bin_key or key in foreign_bins or key in retained_keys:
            continue
        retained_path.append(entry)
        retained_keys.add(key)

    result = {
        key: value
        for key, value in inherited.items()
        if key != "PATH" and not _is_activation_variable(key)
    }
    result.update(
        {
            "PATH": os.pathsep.join((str(selected_bin), *retained_path)),
            "VIRTUAL_ENV": str(selected_environment),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(selected_overlay), str(selected_catkin))
            ),
            "TORCH_HOME": str(selected_project / "models" / "shared" / "torch-cache"),
        }
    )
    return result
