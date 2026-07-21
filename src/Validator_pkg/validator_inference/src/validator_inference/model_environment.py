"""
모델별 Python 환경 manifest를 읽고 재현 가능한 명령 계약을 구성한다.
입력: 추적된 환경 JSON, project root, 모델 ID와 후보 환경 경로.
출력(pub): 없음. 검증된 환경 정의, 경로, hash와 uv argv를 반환한다.
구독(sub): 없음.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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
    "requirements_lock",
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
_IMPORT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_TORCH_BACKEND = re.compile(r"cu[0-9]{3}\Z")
_SETUP_ACTION_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SETUP_ACTIONS = {"build_sparsedrive_cuda_extensions"}


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
    requirements_lock: Path
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
    return ModelEnvironmentDefinition(
        model_id=model_id,
        python_version=python_version,
        requirements_input=requirements_input,
        requirements_lock=requirements_lock,
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
