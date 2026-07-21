"""모델별 Python 환경 manifest와 lock 계약을 회귀 검사한다."""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Callable

import pytest

import validator_inference.model_environment as model_environment
from validator_inference.model_environment import (
    ActiveEnvironmentCheck,
    AssetFingerprint,
    EnvironmentConfigError,
    EnvironmentManifest,
    EnvironmentPaths,
    EnvironmentState,
    EnvironmentStateError,
    ModelEnvironmentDefinition,
    build_model_process_environment,
    build_uv_sync_argv,
    build_uv_venv_argv,
    environment_paths,
    inspect_active_environment,
    load_environment_manifest,
    read_environment_state,
    select_model_environment,
    sha256_file,
    write_environment_state,
)


PACKAGE_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PACKAGE_ROOT / "config" / "model_environments.json"
PACKAGE_RELATIVE_ROOT = Path("src/Validator_pkg/validator_inference")
EXPECTED = {
    "interfuser": {
        "python": "3.8.20",
        "torch": "torch==2.4.1+cu121",
        "torchvision": "torchvision==0.19.1+cu121",
        "cuda_policy": "recommended",
        "cuda_index": "https://download.pytorch.org/whl/cu121",
        "torch_backend": "cu121",
        "required_imports": {
            "torch",
            "torchvision",
            "numpy",
            "cv2",
            "timm",
            "imgaug",
            "skimage",
        },
        "setup_actions": set(),
    },
    "tfv6": {
        "python": "3.10.20",
        "torch": "torch==2.5.0+cu124",
        "torchvision": "torchvision==0.20.0+cu124",
        "cuda_policy": "required",
        "cuda_index": "https://download.pytorch.org/whl/cu124",
        "torch_backend": "cu124",
        "required_imports": {
            "torch",
            "torchvision",
            "torchaudio",
            "numpy",
            "timm",
            "transformers",
            "torchmetrics",
        },
        "setup_actions": set(),
    },
    "diffusiondrive_v2": {
        "python": "3.9.25",
        "torch": "torch==2.0.1+cu118",
        "torchvision": "torchvision==0.15.2+cu118",
        "cuda_policy": "recommended",
        "cuda_index": "https://download.pytorch.org/whl/cu118",
        "torch_backend": "cu118",
        "required_imports": {
            "torch",
            "torchvision",
            "diffusers",
            "einops",
            "nuplan",
        },
        "setup_actions": set(),
    },
    "sparsedrive_v2": {
        "python": "3.9.25",
        "torch": "torch==2.0.1+cu118",
        "torchvision": "torchvision==0.15.2+cu118",
        "cuda_policy": "extension_required",
        "cuda_index": "https://download.pytorch.org/whl/cu118",
        "torch_backend": "cu118",
        "required_imports": {
            "torch",
            "torchvision",
            "nuplan",
            "deformable_aggregation_ext",
            "deformable_aggregation_with_depth_ext",
        },
        "setup_actions": {"build_sparsedrive_cuda_extensions"},
    },
}
COMMON_INPUTS = {
    "PyYAML==6.0.3",
    "rospkg==1.6.0",
    "catkin-pkg==1.0.0",
    "netifaces==0.11.0",
    "python-dateutil==2.9.0.post0",
}


@pytest.fixture
def manifest() -> EnvironmentManifest:
    """저장소가 추적하는 실제 환경 manifest를 읽는다."""
    return load_environment_manifest(MANIFEST_PATH)


@pytest.fixture
def manifest_payload() -> dict[str, Any]:
    """오류 주입 검사를 위해 실제 manifest JSON을 새 객체로 읽는다."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest_fixture(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """독립 requirement 파일을 갖는 임시 package manifest를 만든다."""
    package_root = tmp_path / "validator_inference"
    config_dir = package_root / "config"
    requirements = package_root / "requirements"
    config_dir.mkdir(parents=True)
    requirements.mkdir()
    models = payload.get("models")
    for model_id in EXPECTED:
        input_path = requirements / f"{model_id}.in"
        lock_path = requirements / f"{model_id}.lock"
        input_path.write_text("fixture input\n", encoding="utf-8")
        lock_path.write_text("fixture lock\n", encoding="utf-8")
        if isinstance(models, dict) and isinstance(models.get(model_id), dict):
            model = models[model_id]
            model["requirements_input_sha256"] = sha256_file(input_path)
            model["requirements_lock_sha256"] = sha256_file(lock_path)
    path = config_dir / "model_environments.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mutated_manifest(
    tmp_path: Path,
    payload: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    """payload 한 곳을 변경해 독립적인 invalid manifest를 기록한다."""
    mutate(payload)
    return _write_manifest_fixture(tmp_path, payload)


def _requirement_blocks(lock_text: str) -> list[str]:
    """uv lock의 logical requirement를 묶고 exact SHA-256 option을 검증한다."""
    hash_option = re.compile(r"--hash=sha256:[0-9a-f]{64}\Z")
    blocks: list[str] = []
    current: list[str] = []
    continuing = False
    saw_index_url = False

    def finish_block() -> None:
        """현재 physical line들을 한 requirement argv로 검증해 추가한다."""
        logical = " ".join(
            line.strip()[:-1].rstrip()
            if line.strip().endswith("\\")
            else line.strip()
            for line in current
        )
        tokens = shlex.split(logical, comments=True)
        hash_tokens = [token for token in tokens if hash_option.fullmatch(token)]
        unexpected_options = [
            token
            for token in tokens
            if token.startswith("--") and not hash_option.fullmatch(token)
        ]
        assert not unexpected_options, (
            "unexpected requirement option; every option must be a valid SHA-256 "
            "hash option: " + ", ".join(unexpected_options)
        )
        assert hash_tokens, "installable entry lacks a valid SHA-256 hash option"
        blocks.append("\n".join(current))

    for line in lock_text.splitlines():
        stripped = line.strip()
        if continuing:
            assert line[:1].isspace(), "unexpected unindented continuation content"
            current.append(line)
            continuing = stripped.endswith("\\")
            if not continuing:
                finish_block()
                current = []
            continue
        if not line or line.startswith("#"):
            continue
        if line[0].isspace():
            assert stripped.startswith("#"), "unexpected indented lock content"
            continue
        if line.startswith("--"):
            assert not saw_index_url, "unexpected duplicate lock directive"
            assert line == "--index-url https://pypi.org/simple", (
                "unexpected lock directive: " + line
            )
            saw_index_url = True
            continue
        current = [line]
        continuing = stripped.endswith("\\")
        if not continuing:
            finish_block()
            current = []
    assert not continuing, "unexpected end of continued requirement"
    return blocks


def _assert_compile_command_contract(
    command: list[str], input_path: str, expected_values: dict[str, str]
) -> None:
    """uv compile argv를 strict flag grammar로 해석해 exact 계약과 대조한다."""
    valueless_flags = {
        "--generate-hashes",
        "--emit-index-url",
        "--emit-index-annotation",
    }
    assert command[:3] == ["uv", "pip", "compile"]
    parsed_values: dict[str, str] = {}
    parsed_valueless: set[str] = set()
    positional: list[str] = []
    index = 3
    while index < len(command):
        token = command[index]
        if token in expected_values:
            assert token not in parsed_values, f"duplicate valued flag: {token}"
            assert index + 1 < len(command), f"missing value for flag: {token}"
            parsed_values[token] = command[index + 1]
            index += 2
        elif token in valueless_flags:
            assert token not in parsed_valueless, f"duplicate valueless flag: {token}"
            parsed_valueless.add(token)
            index += 1
        elif token.startswith("--"):
            raise AssertionError(f"unknown compile flag: {token}")
        else:
            positional.append(token)
            index += 1

    assert positional == [input_path], "compile command must have one exact input path"
    assert parsed_values == expected_values
    assert parsed_valueless == valueless_flags


def _valid_environment_state(
    project_root: Path,
    environment_dir: Path,
    definition: ModelEnvironmentDefinition,
    manifest_sha256: str = "a" * 64,
) -> EnvironmentState:
    """검사 가능한 build와 asset을 만들고 그 fingerprint state를 반환한다."""
    python = environment_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)
    overlay_manifest = environment_dir / ".ros-overlay-manifest.json"
    overlay_manifest.write_text('{"packages": []}\n', encoding="utf-8")
    asset = project_root / "models" / definition.model_id / "weights.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"runtime asset")
    metadata = asset.stat()
    return EnvironmentState(
        schema=1,
        model_id=definition.model_id,
        python_version=definition.python_version,
        python_platform="x86_64-manylinux_2_31",
        uv_version="0.11.14",
        manifest_sha256=manifest_sha256,
        lock_sha256=definition.requirements_lock_sha256,
        ros_overlay_sha256=sha256_file(overlay_manifest),
        source_commit=definition.source_commit,
        installed_packages_sha256="b" * 64,
        python_executable=Path("bin/python"),
        assets=(
            AssetFingerprint(
                path=Path("models") / definition.model_id / "weights.bin",
                sha256=sha256_file(asset),
                size=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
            ),
        ),
        completed_at="2026-07-21T12:34:56Z",
        preflight_ids=("python_imports", "runtime_assets"),
    )


def _active_environment_fixture(
    tmp_path: Path, definition: ModelEnvironmentDefinition
) -> tuple[EnvironmentPaths, Path, EnvironmentState]:
    """current symlink가 선택한 정상 build와 state를 구성한다."""
    paths = environment_paths(tmp_path, definition.model_id)
    environment_dir = paths.builds / "build-001"
    environment_dir.mkdir(parents=True)
    state = _valid_environment_state(tmp_path, environment_dir, definition)
    write_environment_state(environment_dir / ".validator-environment.json", state)
    paths.current.symlink_to(Path("builds") / environment_dir.name)
    return paths, environment_dir, state


def _state_payload(state: EnvironmentState) -> dict[str, Any]:
    """state writer의 JSON을 변형 가능한 payload로 읽는다."""
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


def test_environment_state_roundtrip_is_deterministic_and_uses_posix_paths(
    tmp_path: Path, manifest: EnvironmentManifest
):
    """state JSON이 안정된 정렬·들여쓰기·개행과 POSIX 경로로 왕복되는지 검사한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    state = _valid_environment_state(
        tmp_path, environment_dir, manifest.models["interfuser"]
    )
    state_path = environment_dir / ".validator-environment.json"

    write_environment_state(state_path, state)
    first = state_path.read_bytes()
    write_environment_state(state_path, read_environment_state(environment_dir))

    assert read_environment_state(environment_dir) == state
    assert state_path.read_bytes() == first
    assert first == (json.dumps(_state_payload(state), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    assert os.stat(state_path).st_mode & 0o777 == 0o600


def test_environment_state_write_replaces_atomically(
    tmp_path: Path, manifest: EnvironmentManifest, monkeypatch: pytest.MonkeyPatch
):
    """state writer가 같은 directory의 임시 파일을 os.replace로 선택하는지 검사한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    state = _valid_environment_state(
        tmp_path, environment_dir, manifest.models["interfuser"]
    )
    state_path = environment_dir / ".validator-environment.json"
    state_path.write_text("old\n", encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: Any, target: Any) -> None:
        """atomic 교체 인자를 기록한 뒤 실제 교체를 수행한다."""
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(model_environment.os, "replace", recording_replace)
    write_environment_state(state_path, state)

    assert len(replacements) == 1
    temporary, target = replacements[0]
    assert temporary.parent == state_path.parent
    assert target == state_path
    assert not temporary.exists()
    assert read_environment_state(environment_dir) == state


def test_environment_state_write_cleans_temporary_file_after_replace_failure(
    tmp_path: Path, manifest: EnvironmentManifest, monkeypatch: pytest.MonkeyPatch
):
    """atomic replace 실패 시 기존 state와 directory 내용을 보존하는지 검사한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    state = _valid_environment_state(
        tmp_path, environment_dir, manifest.models["interfuser"]
    )
    state_path = environment_dir / ".validator-environment.json"
    state_path.write_text("old state\n", encoding="utf-8")
    before = set(environment_dir.iterdir())

    def fail_replace(source: Any, target: Any) -> None:
        """교체 단계의 운영체제 오류를 재현한다."""
        raise OSError("replace failed")

    monkeypatch.setattr(model_environment.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_environment_state(state_path, state)

    assert state_path.read_text(encoding="utf-8") == "old state\n"
    assert set(environment_dir.iterdir()) == before


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda payload: payload.pop("schema"), "missing keys"),
        (lambda payload: payload.update(extra=True), "unknown keys"),
        (lambda payload: payload.update(schema=True), "schema"),
        (lambda payload: payload.update(schema=2), "schema"),
        (lambda payload: payload.update(model_id="unknown"), "model_id"),
        (lambda payload: payload.update(python_version="3.8"), "python_version"),
        (lambda payload: payload.update(python_platform="linux"), "python_platform"),
        (lambda payload: payload.update(uv_version="0.11.15"), "uv_version"),
        (lambda payload: payload.update(manifest_sha256="A" * 64), "manifest_sha256"),
        (lambda payload: payload.update(lock_sha256="0" * 63), "lock_sha256"),
        (lambda payload: payload.update(ros_overlay_sha256=12), "ros_overlay_sha256"),
        (lambda payload: payload.update(source_commit="g" * 40), "source_commit"),
        (
            lambda payload: payload.update(installed_packages_sha256="x"),
            "installed_packages_sha256",
        ),
        (lambda payload: payload.update(python_executable="/bin/python"), "python_executable"),
        (
            lambda payload: payload.update(python_executable="bin/../../python"),
            "python_executable",
        ),
        (lambda payload: payload.update(completed_at="2026-02-30T00:00:00Z"), "completed_at"),
        (lambda payload: payload.update(preflight_ids=["ok", "ok"]), "preflight_ids"),
        (lambda payload: payload.update(preflight_ids=["bad id"]), "preflight_ids"),
        (lambda payload: payload.update(assets={}), "assets"),
        (
            lambda payload: payload["assets"][0].update(path="../weights.bin"),
            "assets.path",
        ),
        (
            lambda payload: payload["assets"].append(dict(payload["assets"][0])),
            "duplicate asset path",
        ),
        (lambda payload: payload["assets"][0].update(sha256="z" * 64), "assets.sha256"),
        (lambda payload: payload["assets"][0].update(size=True), "assets.size"),
        (lambda payload: payload["assets"][0].update(mtime_ns=-1), "assets.mtime_ns"),
        (lambda payload: payload["assets"][0].update(extra=True), "unknown keys"),
    ),
)
def test_environment_state_rejects_malformed_payloads(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
):
    """state의 schema, 형식, ID, digest와 상대 경로를 엄격히 검증한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    state = _valid_environment_state(
        tmp_path, environment_dir, manifest.models["interfuser"]
    )
    payload = _state_payload(state)
    mutate(payload)
    (environment_dir / ".validator-environment.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(EnvironmentStateError, match=match):
        read_environment_state(environment_dir)


def test_environment_state_rejects_duplicate_json_keys(tmp_path: Path):
    """JSON decoder 단계에서 중복 state key도 손상으로 거부하는지 검사한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    (environment_dir / ".validator-environment.json").write_text(
        '{"schema": 1, "schema": 1}\n', encoding="utf-8"
    )

    with pytest.raises(EnvironmentStateError, match="duplicate key: schema"):
        read_environment_state(environment_dir)


def test_environment_state_normalizes_invalid_utf8(tmp_path: Path):
    """UTF-8로 decode할 수 없는 state도 EnvironmentStateError로 정규화한다."""
    environment_dir = tmp_path / "build"
    environment_dir.mkdir()
    (environment_dir / ".validator-environment.json").write_bytes(b"{\xff}\n")

    with pytest.raises(EnvironmentStateError, match="cannot read environment state"):
        read_environment_state(environment_dir)


def test_inspect_active_environment_reports_invalid_utf8_state(
    tmp_path: Path, manifest: EnvironmentManifest
):
    """active 검사가 잘못된 UTF-8 state를 예외 없이 사용 불가로 보고한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, _ = _active_environment_fixture(tmp_path, definition)
    (environment_dir / ".validator-environment.json").write_bytes(b"{\xff}\n")

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert check == ActiveEnvironmentCheck(
        False,
        f"invalid environment state: cannot read environment state: "
        f"{environment_dir / '.validator-environment.json'}",
        environment_dir,
        None,
    )


def test_inspect_active_environment_accepts_matching_state_without_hashing_assets(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    monkeypatch: pytest.MonkeyPatch,
):
    """빠른 active 검사가 asset 본문 hash 없이 fingerprint만 확인하는지 검사한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, state = _active_environment_fixture(tmp_path, definition)
    asset_path = tmp_path / state.assets[0].path
    hashed: list[Path] = []
    real_sha256_file = sha256_file

    def recording_sha256_file(path: Path) -> str:
        """overlay hash 호출만 허용하고 asset deep hash 호출을 기록한다."""
        hashed.append(Path(path))
        if Path(path) == asset_path:
            raise AssertionError("active inspection must not hash a runtime asset")
        return real_sha256_file(path)

    monkeypatch.setattr(model_environment, "sha256_file", recording_sha256_file)

    assert inspect_active_environment(paths, definition, "a" * 64) == (
        ActiveEnvironmentCheck(True, "ready", environment_dir.resolve(), state)
    )
    assert hashed == [environment_dir / ".ros-overlay-manifest.json"]


@pytest.mark.parametrize(
    "kind,reason",
    (
        ("missing", "current"),
        ("regular", "symlink"),
        ("dangling", "dangling"),
        ("outside", "builds"),
        ("nested", "immediate"),
    ),
)
def test_inspect_active_environment_rejects_invalid_current_selection(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    kind: str,
    reason: str,
):
    """current가 없거나 안전한 immediate build symlink가 아니면 거부하는지 검사한다."""
    definition = manifest.models["interfuser"]
    paths = environment_paths(tmp_path, definition.model_id)
    paths.model_root.mkdir(parents=True)
    if kind == "regular":
        paths.current.write_text("not a symlink\n", encoding="utf-8")
    elif kind == "dangling":
        paths.current.symlink_to(Path("builds") / "missing")
    elif kind == "outside":
        outside = tmp_path / "outside-build"
        outside.mkdir()
        paths.current.symlink_to(outside)
    elif kind == "nested":
        nested = paths.builds / "group" / "build-001"
        nested.mkdir(parents=True)
        paths.current.symlink_to(Path("builds") / "group" / "build-001")

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert reason in check.reason


@pytest.mark.parametrize(
    "kind",
    ("missing", "directory", "not-executable"),
)
def test_inspect_active_environment_rejects_unusable_selected_python(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    kind: str,
):
    """선택 build의 bin/python이 일반 실행 파일이 아니면 거부하는지 검사한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, _ = _active_environment_fixture(tmp_path, definition)
    python = environment_dir / "bin" / "python"
    python.unlink()
    if kind == "directory":
        python.mkdir()
    elif kind == "not-executable":
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o600)

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert "bin/python" in check.reason


@pytest.mark.parametrize(
    "change,reason",
    (
        (lambda state: replace(state, model_id="tfv6"), "model"),
        (lambda state: replace(state, python_version="3.8.19"), "Python"),
        (lambda state: replace(state, manifest_sha256="c" * 64), "manifest"),
        (lambda state: replace(state, lock_sha256="c" * 64), "lock"),
        (lambda state: replace(state, source_commit="c" * 40), "source"),
        (
            lambda state: replace(state, python_executable=Path("bin/python3")),
            "python_executable",
        ),
    ),
)
def test_inspect_active_environment_rejects_stale_definition_state(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    change: Callable[[EnvironmentState], EnvironmentState],
    reason: str,
):
    """모델·Python·definition digest·source가 달라진 active state를 거부한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, state = _active_environment_fixture(tmp_path, definition)
    write_environment_state(
        environment_dir / ".validator-environment.json", change(state)
    )

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert reason in check.reason


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema", 2),
        ("python_platform", "x86_64-manylinux_2_30"),
        ("uv_version", "0.11.13"),
    ),
)
def test_inspect_active_environment_converts_malformed_state_to_invalid_result(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    field: str,
    value: Any,
):
    """지원하지 않는 schema·platform·uv state가 정상 run에서 예외를 내지 않는지 검사한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, state = _active_environment_fixture(tmp_path, definition)
    payload = _state_payload(state)
    payload[field] = value
    (environment_dir / ".validator-environment.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert "state" in check.reason


@pytest.mark.parametrize("kind", ("missing", "directory", "mutated"))
def test_inspect_active_environment_rejects_invalid_overlay_manifest(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    kind: str,
):
    """ROS overlay manifest가 없거나 일반 파일이 아니거나 바뀌면 거부한다."""
    definition = manifest.models["interfuser"]
    paths, environment_dir, _ = _active_environment_fixture(tmp_path, definition)
    overlay_manifest = environment_dir / ".ros-overlay-manifest.json"
    overlay_manifest.unlink()
    if kind == "directory":
        overlay_manifest.mkdir()
    elif kind == "mutated":
        overlay_manifest.write_text('{"packages": ["changed"]}\n', encoding="utf-8")

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert "overlay manifest" in check.reason


@pytest.mark.parametrize(
    "kind,reason",
    (
        ("missing", "missing"),
        ("directory", "regular"),
        ("symlink", "symlink"),
        ("size", "size"),
        ("mtime", "mtime"),
    ),
)
def test_inspect_active_environment_rejects_stale_or_unsafe_assets(
    tmp_path: Path,
    manifest: EnvironmentManifest,
    kind: str,
    reason: str,
):
    """asset의 존재·형식·size·nanosecond mtime fingerprint를 검사한다."""
    definition = manifest.models["interfuser"]
    paths, _, state = _active_environment_fixture(tmp_path, definition)
    asset = tmp_path / state.assets[0].path
    if kind == "missing":
        asset.unlink()
    elif kind == "directory":
        asset.unlink()
        asset.mkdir()
    elif kind == "symlink":
        target = tmp_path / "other-weights.bin"
        target.write_bytes(b"runtime asset")
        os.utime(target, ns=(state.assets[0].mtime_ns, state.assets[0].mtime_ns))
        asset.unlink()
        asset.symlink_to(target)
    elif kind == "size":
        asset.write_bytes(b"changed asset size")
    elif kind == "mtime":
        changed = state.assets[0].mtime_ns + 1_000_000
        os.utime(asset, ns=(changed, changed))

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert reason in check.reason


def test_inspect_active_environment_rejects_symlinked_asset_parent(
    tmp_path: Path, manifest: EnvironmentManifest
):
    """repository-relative asset 경로의 중간 directory symlink도 거부한다."""
    definition = manifest.models["interfuser"]
    paths, _, state = _active_environment_fixture(tmp_path, definition)
    models = tmp_path / "models"
    relocated = tmp_path / "relocated-models"
    models.rename(relocated)
    models.symlink_to(relocated, target_is_directory=True)

    check = inspect_active_environment(paths, definition, "a" * 64)

    assert not check.valid
    assert "symlink" in check.reason


def test_inspect_active_environment_rejects_untrusted_path_layout(
    tmp_path: Path, manifest: EnvironmentManifest
):
    """project/.venvs/MODEL에서 안전하게 유도할 수 없는 path 묶음을 거부한다."""
    definition = manifest.models["interfuser"]
    model_root = tmp_path / ".venvs" / definition.model_id
    wrong_paths = EnvironmentPaths(
        model_root=model_root,
        builds=tmp_path / "foreign-builds",
        current=model_root / "current",
    )

    check = inspect_active_environment(wrong_paths, definition, "a" * 64)

    assert not check.valid
    assert "layout" in check.reason


def test_build_model_process_environment_sanitizes_python_activation_state(
    tmp_path: Path,
):
    """child process 환경이 선택 venv와 ROS path만 사용하고 ROS 변수는 보존한다."""
    project_root = tmp_path.resolve()
    environment_dir = project_root / ".venvs" / "interfuser" / "builds" / "build-1"
    selected_bin = environment_dir / "bin"
    selected_bin.mkdir(parents=True)
    python = selected_bin / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o700)
    ros_overlay = environment_dir / ".ros-overlay"
    catkin_python_path = project_root / "devel" / "lib" / "python3" / "dist-packages"
    ros_overlay.mkdir()
    catkin_python_path.mkdir(parents=True)
    foreign = project_root / "foreign venv"
    inherited = {
        "HOME": "/home/operator",
        "PATH": os.pathsep.join(
            (
                str(foreign / "bin"),
                "/usr/local/bin",
                str(selected_bin),
                "/usr/bin",
                "/usr/local/bin",
                "",
                "/bin",
            )
        ),
        "PYTHONPATH": "/untrusted/python:/another/python",
        "PYTHONHOME": "/foreign/python",
        "PYTHONUSERBASE": "/foreign/userbase",
        "PYTHONNOUSERSITE": "0",
        "VIRTUAL_ENV": str(foreign),
        "VIRTUAL_ENV_PROMPT": "(foreign) ",
        "CONDA_PREFIX": "/opt/conda/envs/foreign",
        "CONDA_DEFAULT_ENV": "foreign",
        "CONDA_SHLVL": "2",
        "CONDA_EXE": "/opt/conda/bin/conda",
        "_CE_CONDA": "conda",
        "_CE_M": "-m",
        "TORCH_HOME": "/foreign/torch-cache",
        "ROS_MASTER_URI": "http://127.0.0.1:11311",
        "ROS_IP": "127.0.0.1",
        "CUSTOM_RUNTIME_FLAG": "keep-me",
    }

    actual = build_model_process_environment(
        inherited,
        environment_dir,
        ros_overlay,
        catkin_python_path,
        project_root,
    )

    assert actual == {
        "HOME": "/home/operator",
        "PATH": os.pathsep.join(
            (str(selected_bin), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "VIRTUAL_ENV": str(environment_dir),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(ros_overlay), str(catkin_python_path))),
        "TORCH_HOME": str(project_root / "models" / "shared" / "torch-cache"),
        "ROS_MASTER_URI": "http://127.0.0.1:11311",
        "ROS_IP": "127.0.0.1",
        "CUSTOM_RUNTIME_FLAG": "keep-me",
    }


@pytest.mark.parametrize(
    "invalid",
    (
        "relative-environment",
        "relative-overlay",
        "relative-catkin",
        "relative-project",
        "missing-python",
        "non-executable-python",
        "missing-overlay",
        "missing-catkin",
    ),
)
def test_build_model_process_environment_rejects_invalid_paths(
    tmp_path: Path, invalid: str
):
    """child 환경을 만들기 전에 모든 입력 절대 경로와 필수 파일·directory를 검증한다."""
    project_root = tmp_path.resolve()
    environment_dir = project_root / "environment"
    python = environment_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o700)
    ros_overlay = project_root / "overlay"
    catkin_python_path = project_root / "catkin"
    ros_overlay.mkdir()
    catkin_python_path.mkdir()
    if invalid == "relative-environment":
        environment_dir = Path("environment")
    elif invalid == "relative-overlay":
        ros_overlay = Path("overlay")
    elif invalid == "relative-catkin":
        catkin_python_path = Path("catkin")
    elif invalid == "relative-project":
        project_root = Path("project")
    elif invalid == "missing-python":
        python.unlink()
    elif invalid == "non-executable-python":
        python.chmod(0o600)
    elif invalid == "missing-overlay":
        ros_overlay.rmdir()
    elif invalid == "missing-catkin":
        catkin_python_path.rmdir()

    with pytest.raises(EnvironmentStateError):
        build_model_process_environment(
            {}, environment_dir, ros_overlay, catkin_python_path, project_root
        )


def test_manifest_defines_exact_common_contract(manifest: EnvironmentManifest):
    """공통 resolver 계약이 요청한 exact 값으로 고정되는지 검사한다."""
    assert manifest.schema == 1
    assert manifest.uv_version == "0.11.14"
    assert manifest.python_platform == "x86_64-manylinux_2_31"
    assert manifest.exclude_newer == "2026-07-21T00:00:00Z"
    assert manifest.default_index == "https://pypi.org/simple"
    assert manifest.index_strategy == "first-index"


def test_manifest_defines_exact_supported_models(manifest: EnvironmentManifest):
    """manifest가 네 모델의 Python, PyTorch, CUDA 계약만 정의하는지 검사한다."""
    assert set(manifest.models) == set(EXPECTED)
    for model_id, expected in EXPECTED.items():
        definition = manifest.models[model_id]
        requirement_text = definition.requirements_input.read_text(encoding="utf-8")

        assert definition.model_id == model_id
        assert definition.python_version == expected["python"]
        assert definition.cuda_policy == expected["cuda_policy"]
        assert definition.torch_backend == expected["torch_backend"]
        assert set(definition.required_imports) == expected["required_imports"]
        assert set(definition.setup_actions) == expected["setup_actions"]
        assert definition.indexes == (
            expected["cuda_index"],
            "https://pypi.org/simple",
        )
        assert expected["torch"] in requirement_text.splitlines()
        assert expected["torchvision"] in requirement_text.splitlines()
        assert COMMON_INPUTS <= set(requirement_text.splitlines())
        assert definition.requirements_input.is_file()
        assert definition.requirements_lock.is_file()
        assert definition.requirements_input_sha256 == sha256_file(
            definition.requirements_input
        )
        assert definition.requirements_lock_sha256 == sha256_file(
            definition.requirements_lock
        )

    tfv6_text = manifest.models["tfv6"].requirements_input.read_text(encoding="utf-8")
    assert "torchaudio==2.5.0+cu124" in tfv6_text.splitlines()
    assert {
        "deformable_aggregation_ext",
        "deformable_aggregation_with_depth_ext",
    } <= set(manifest.models["sparsedrive_v2"].required_imports)
    assert manifest.models["sparsedrive_v2"].setup_actions == (
        "build_sparsedrive_cuda_extensions",
    )


def test_environment_value_objects_are_frozen(manifest: EnvironmentManifest):
    """환경 계약과 이후 state용 value object가 모두 불변 dataclass인지 검사한다."""
    values = (
        manifest,
        manifest.models["interfuser"],
        EnvironmentPaths(Path(".venvs/interfuser"), Path("builds"), Path("current")),
        AssetFingerprint(Path("asset.bin"), "0" * 64, 1, 2),
        EnvironmentState(
            1,
            "interfuser",
            "3.8.20",
            "x86_64-manylinux_2_31",
            "0.11.14",
            "0" * 64,
            "1" * 64,
            "2" * 64,
            "3" * 40,
            "4" * 64,
            Path("bin/python"),
            (),
            "2026-07-21T00:00:00Z",
            (),
        ),
        ActiveEnvironmentCheck(False, "missing", None, None),
    )

    for value in values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, next(iter(value.__dict__)), "changed")

    assert issubclass(EnvironmentConfigError, ValueError)
    assert issubclass(EnvironmentStateError, ValueError)


@pytest.mark.parametrize(
    "scope,key,value",
    (
        ("top", "surprise", True),
        ("model", "surprise", True),
    ),
)
def test_manifest_rejects_unknown_keys(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    scope: str,
    key: str,
    value: Any,
):
    """top-level과 model object의 미지원 key를 모두 거부하는지 검사한다."""
    def mutate(payload: dict[str, Any]) -> None:
        target = payload if scope == "top" else payload["models"]["interfuser"]
        target[key] = value

    path = _mutated_manifest(tmp_path, manifest_payload, mutate)
    with pytest.raises(EnvironmentConfigError, match="unknown keys"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda payload: payload.update(schema=2), "schema"),
        (lambda payload: payload.update(schema="1"), "schema"),
        (lambda payload: payload.update(models=[]), "models"),
        (lambda payload: payload.pop("uv_version"), "missing keys"),
        (lambda payload: payload["models"].pop("tfv6"), "model IDs"),
        (
            lambda payload: payload["models"].update(
                extra=payload["models"]["interfuser"]
            ),
            "model IDs",
        ),
    ),
)
def test_manifest_rejects_wrong_schema_and_model_set(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    match: str,
):
    """schema 형식과 정확한 model ID 집합을 강제하는지 검사한다."""
    path = _mutated_manifest(tmp_path, manifest_payload, mutate)
    with pytest.raises(EnvironmentConfigError, match=match):
        load_environment_manifest(path)


@pytest.mark.parametrize("bad_version", ("3.8", "3.8.x", "v3.8.20", "3.08.20", 3.8))
def test_manifest_rejects_malformed_exact_python_versions(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_version: Any,
):
    """CPython 선택 값이 정규화한 semantic patch version인지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            python_version=bad_version
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="python_version"):
        load_environment_manifest(path)


@pytest.mark.parametrize("bad_sha", ("abc", "g" * 40, "A" * 40, 123))
def test_manifest_rejects_malformed_source_sha(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_sha: Any,
):
    """source commit이 정확한 lowercase SHA-1 형식인지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            source_commit=bad_sha
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="source_commit"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "field,bad_sha",
    (
        ("requirements_input_sha256", "abc"),
        ("requirements_input_sha256", "g" * 64),
        ("requirements_input_sha256", "A" * 64),
        ("requirements_lock_sha256", "0" * 63),
        ("requirements_lock_sha256", 123),
    ),
)
def test_manifest_rejects_malformed_requirement_digests(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    field: str,
    bad_sha: Any,
):
    """requirement digest가 정확한 lowercase SHA-256 형식인지 검사한다."""
    path = _write_manifest_fixture(tmp_path, manifest_payload)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["interfuser"][field] = bad_sha
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        EnvironmentConfigError, match="must be a lowercase 64-digit SHA-256"
    ):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "field,filename",
    (
        ("requirements_input_sha256", "interfuser.in"),
        ("requirements_lock_sha256", "interfuser.lock"),
    ),
)
def test_manifest_rejects_stale_requirement_digests(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    field: str,
    filename: str,
):
    """manifest digest와 실제 requirement bytes가 다르면 즉시 거부하는지 검사한다."""
    path = _write_manifest_fixture(tmp_path, manifest_payload)
    (path.parent.parent / "requirements" / filename).write_text(
        "changed after hashing\n", encoding="utf-8"
    )

    with pytest.raises(EnvironmentConfigError, match=f"{field} does not match"):
        load_environment_manifest(path)


@pytest.mark.parametrize("bad_policy", ("optional", "REQUIRED", "", None, []))
def test_manifest_rejects_unknown_cuda_policy(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_policy: Any,
):
    """세 가지 실행 가능 CUDA policy 외 값을 거부하는지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            cuda_policy=bad_policy
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="cuda_policy"):
        load_environment_manifest(path)


@pytest.mark.parametrize("bad_backend", ("cpu", "CU121", "cu12", "cu121x", None))
def test_manifest_rejects_unknown_or_mismatched_torch_backend(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_backend: Any,
):
    """torch backend 이름이 CUDA index suffix와 정확히 일치하는지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            torch_backend=bad_backend
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="torch_backend"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "bad_import",
    ("torch vision", " torch", "torch ", "torch-module", "torch..nn"),
)
def test_manifest_rejects_invalid_required_import_syntax(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_import: str,
):
    """required import가 공백 없는 Python module identifier인지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            required_imports=["torch", bad_import]
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="required_imports"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "bad_action",
    (
        "download_assets",
        " build_sparsedrive_cuda_extensions",
        "build_sparsedrive_cuda_extensions ",
        "build sparsedrive cuda extensions",
    ),
)
def test_manifest_rejects_invalid_or_unknown_setup_actions(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_action: str,
):
    """setup action이 identifier 문법과 executor allowlist를 모두 따르는지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            setup_actions=[bad_action]
        ),
    )
    with pytest.raises(EnvironmentConfigError, match="setup_actions"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "field,bad_path",
    (
        ("requirements_input", "/tmp/interfuser.in"),
        ("requirements_input", "../requirements/interfuser.in"),
        ("requirements_input", "requirements/../interfuser.in"),
        ("requirements_lock", "/tmp/interfuser.lock"),
        ("requirements_lock", "requirements/../../interfuser.lock"),
    ),
)
def test_manifest_rejects_absolute_and_traversing_requirement_paths(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    field: str,
    bad_path: str,
):
    """requirement artifact 경로가 package root 밖으로 나가지 않는지 검사한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            **{field: bad_path}
        ),
    )
    with pytest.raises(EnvironmentConfigError, match=field):
        load_environment_manifest(path)


def test_manifest_rejects_requirement_symlink_escape(
    tmp_path: Path, manifest_payload: dict[str, Any]
):
    """정상 상대 경로의 symlink도 package root 밖을 가리키면 거부하는지 검사한다."""
    path = _write_manifest_fixture(tmp_path, manifest_payload)
    requirement = path.parent.parent / "requirements" / "interfuser.in"
    outside = tmp_path / "outside.in"
    outside.write_text("outside\n", encoding="utf-8")
    requirement.unlink()
    requirement.symlink_to(outside)

    with pytest.raises(EnvironmentConfigError, match="package root"):
        load_environment_manifest(path)


@pytest.mark.parametrize(
    "indexes",
    (
        "https://download.pytorch.org/whl/cu121",
        [],
        ["http://download.pytorch.org/whl/cu121", "https://pypi.org/simple"],
        ["https://pypi.org/simple", "https://download.pytorch.org/whl/cu121"],
        ["https://user:password@example.com/simple", "https://pypi.org/simple"],
        ["not-a-url", "https://pypi.org/simple"],
        [
            "https://download.pytorch.org:bogus/whl/cu121",
            "https://pypi.org/simple",
        ],
        [
            "https://download.pytorch.org:8443/whl/cu121",
            "https://pypi.org/simple",
        ],
    ),
)
def test_manifest_rejects_malformed_or_reordered_indexes(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    indexes: Any,
):
    """공식 CUDA first index와 PyPI secondary index 순서를 강제한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(indexes=indexes),
    )
    with pytest.raises(EnvironmentConfigError, match="indexes"):
        load_environment_manifest(path)


@pytest.mark.parametrize("bad_character", ("\n", "\t", "\x00", "\x1f", "\x7f"))
def test_manifest_rejects_ascii_whitespace_and_controls_in_index_urls(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    bad_character: str,
):
    """URL parser가 제거할 수 있는 ASCII 공백과 제어 문자도 먼저 거부한다."""
    path = _mutated_manifest(
        tmp_path,
        manifest_payload,
        lambda payload: payload["models"]["interfuser"].update(
            indexes=[
                "https://download.pytorch.org/whl/" + bad_character + "cu121",
                "https://pypi.org/simple",
            ]
        ),
    )
    with pytest.raises(
        EnvironmentConfigError, match="ASCII whitespace or control character"
    ):
        load_environment_manifest(path)


@pytest.mark.parametrize("field", ("requirements_input", "requirements_lock"))
def test_manifest_rejects_missing_requirement_artifacts(
    tmp_path: Path,
    manifest_payload: dict[str, Any],
    field: str,
):
    """manifest에 기록한 input 또는 lock 파일 누락을 즉시 거부하는지 검사한다."""
    path = _write_manifest_fixture(tmp_path, manifest_payload)
    missing = path.parent.parent / manifest_payload["models"]["interfuser"][field]
    missing.unlink()
    with pytest.raises(EnvironmentConfigError, match="does not exist"):
        load_environment_manifest(path)


def test_select_model_environment_rejects_unknown_model(manifest: EnvironmentManifest):
    """선택 API가 manifest 밖 model ID를 명확히 거부하는지 검사한다."""
    assert select_model_environment(manifest, "tfv6") is manifest.models["tfv6"]
    with pytest.raises(EnvironmentConfigError, match="unsupported model_id"):
        select_model_environment(manifest, "unknown")


def test_environment_paths_are_repository_owned_and_validate_model_id(tmp_path: Path):
    """환경 경로가 .venvs/MODEL 아래 builds와 current로 고정되는지 검사한다."""
    paths = environment_paths(tmp_path, "interfuser")
    assert paths.model_root == tmp_path.resolve() / ".venvs" / "interfuser"
    assert paths.builds == paths.model_root / "builds"
    assert paths.current == paths.model_root / "current"

    for model_id in ("unknown", "../interfuser", "/tmp/interfuser", ""):
        with pytest.raises(EnvironmentConfigError, match="unsupported model_id"):
            environment_paths(tmp_path, model_id)


def test_sha256_file_hashes_binary_content(tmp_path: Path):
    """파일 hash가 binary byte 전체를 SHA-256으로 계산하는지 검사한다."""
    path = tmp_path / "payload.bin"
    path.write_bytes(b"validator\x00environment\n")
    assert sha256_file(path) == (
        "252fe01b5e0b929e36272fd908260c4658dc1d4832e949beef17d8181a53e401"
    )


def test_uv_argv_builders_return_exact_argument_tuples(
    manifest: EnvironmentManifest, tmp_path: Path
):
    """venv와 sync 명령이 shell 문자열 없이 정확한 argv 순서를 갖는지 검사한다."""
    definition = manifest.models["interfuser"]
    candidate = tmp_path / "candidate with spaces"
    uv = Path("/opt/tools/uv with spaces")

    assert build_uv_venv_argv(uv, definition, candidate) == (
        str(uv),
        "venv",
        "--no-project",
        "--managed-python",
        "--python",
        "3.8.20",
        str(candidate),
    )
    assert build_uv_sync_argv(uv, definition, candidate) == (
        str(uv),
        "pip",
        "sync",
        "--python",
        str(candidate / "bin" / "python"),
        "--require-hashes",
        "--strict",
        "--no-python-downloads",
        "--default-index",
        "https://pypi.org/simple",
        "--index-strategy",
        "first-index",
        "--torch-backend",
        "cu121",
        str(definition.requirements_lock),
    )


def test_requirement_inputs_pin_model_specific_runtime_contracts(
    manifest: EnvironmentManifest,
):
    """upstream별 핵심 runtime package가 direct input에 exact pin으로 남는지 검사한다."""
    inputs = {
        model_id: set(definition.requirements_input.read_text(encoding="utf-8").splitlines())
        for model_id, definition in manifest.models.items()
    }
    assert {
        "numpy==1.24.4",
        "pillow==10.4.0",
        "imgaug==0.4.0",
        "scikit-image==0.21.0",
        "scipy==1.10.1",
        "shapely==2.0.7",
        "tqdm==4.69.0",
    } <= inputs["interfuser"]
    assert "opencv-python==4.13.0.92" not in inputs["interfuser"]
    assert {
        "numpy==1.26.*",
        "timm==1.0.19",
        "transformers==4.46.3",
        "torchmetrics==0.11.*",
    } <= inputs["tfv6"]
    assert "protobuf==4.25.3" in inputs["diffusiondrive_v2"]
    assert any(line.startswith("diffusers==") for line in inputs["diffusiondrive_v2"])
    assert any(line.startswith("einops==") for line in inputs["diffusiondrive_v2"])
    assert "protobuf==3.20.*" in inputs["sparsedrive_v2"]


def test_interfuser_input_documents_pinned_upstream_and_runtime_scope(
    manifest: EnvironmentManifest,
):
    """InterFuser input이 검증한 source와 의도적인 runtime 범위를 기록하는지 검사한다."""
    header = manifest.models["interfuser"].requirements_input.read_text(
        encoding="utf-8"
    ).split("\n\n", 1)[0]
    assert (
        "https://github.com/opendilab/InterFuser/tree/"
        "f0be8ea294c49a59febaa032dd6da40b475e3657"
    ) in header
    assert "training" in header.lower()
    assert "simulator" in header.lower()
    assert "leaderboard" in header.lower()
    assert "intentionally" in header.lower()


def test_lock_compile_commands_match_manifest_contract(
    manifest: EnvironmentManifest,
):
    """각 uv compile header를 argv로 해석해 manifest resolver 계약과 대조한다."""
    for model_id, definition in manifest.models.items():
        lines = definition.requirements_lock.read_text(encoding="utf-8").splitlines()
        commands = [
            shlex.split(line[len("#    ") :])
            for line in lines
            if line.startswith("#    uv pip compile ")
        ]
        assert len(commands) == 1
        command = commands[0]
        input_path = str(PACKAGE_RELATIVE_ROOT / f"requirements/{model_id}.in")
        output_path = str(PACKAGE_RELATIVE_ROOT / f"requirements/{model_id}.lock")
        _assert_compile_command_contract(
            command,
            input_path,
            {
                "--output-file": output_path,
                "--python-version": definition.python_version,
                "--python-platform": manifest.python_platform,
                "--default-index": manifest.default_index,
                "--index-strategy": manifest.index_strategy,
                "--torch-backend": definition.torch_backend,
                "--exclude-newer": manifest.exclude_newer,
            },
        )


def test_compile_command_parser_rejects_swapped_valued_flag_values():
    """같은 token 집합이어도 valued flag의 값이 바뀌면 거부하는지 검사한다."""
    expected_values = {
        "--output-file": "requirements/model.lock",
        "--python-version": "3.8.20",
        "--python-platform": "x86_64-manylinux_2_31",
        "--default-index": "https://pypi.org/simple",
        "--index-strategy": "first-index",
        "--torch-backend": "cu121",
        "--exclude-newer": "2026-07-21T00:00:00Z",
    }
    command = [
        "uv",
        "pip",
        "compile",
        "requirements/model.in",
        "--output-file",
        "requirements/model.lock",
        "--python-version",
        "x86_64-manylinux_2_31",
        "--python-platform",
        "3.8.20",
        "--default-index",
        "https://pypi.org/simple",
        "--index-strategy",
        "first-index",
        "--torch-backend",
        "cu121",
        "--exclude-newer",
        "2026-07-21T00:00:00Z",
        "--generate-hashes",
        "--emit-index-url",
        "--emit-index-annotation",
    ]

    with pytest.raises(AssertionError):
        _assert_compile_command_contract(
            command, "requirements/model.in", expected_values
        )


@pytest.mark.parametrize(
    "hash_content",
    (
        "--requirement-option=--hash=sha256:" + "a" * 64,
        "--hash=sha256:" + "a" * 63,
        "--hash=sha256:" + "g" * 64,
        "# --hash=sha256:" + "a" * 64,
    ),
    ids=("fake", "short", "nonhex", "comment-only"),
)
def test_requirement_block_parser_rejects_invalid_hash_options(hash_content: str):
    """hash처럼 보이는 substring이나 comment를 installable hash로 세지 않는다."""
    lock_text = "example==1.0.0 \\\n    " + hash_content + "\n"
    with pytest.raises(AssertionError, match="valid SHA-256 hash option"):
        _requirement_blocks(lock_text)


@pytest.mark.parametrize(
    "lock_text",
    (
        "    --hash=sha256:" + "a" * 64 + "\n",
        "--find-links https://example.com/packages\n",
        "example==1.0.0 \\\n    --no-deps\n",
        "example==1.0.0 \\\n"
        "    --evil=--hash=sha256:" + "a" * 64 + " \\\n"
        "    --hash=sha256:" + "b" * 64 + "\n",
    ),
    ids=(
        "orphan-indentation",
        "unexpected-directive",
        "unexpected-continuation",
        "misleading-hash-option",
    ),
)
def test_requirement_block_parser_rejects_unexpected_content(lock_text: str):
    """installable block 밖의 들여쓰기와 미지원 directive를 거부하는지 검사한다."""
    with pytest.raises(AssertionError, match="unexpected"):
        _requirement_blocks(lock_text)


def test_generated_locks_are_pinned_hashed_and_index_annotated(
    manifest: EnvironmentManifest,
):
    """모든 uv lock installable entry가 pin과 SHA-256 hash를 갖는지 검사한다."""
    for definition in manifest.models.values():
        text = definition.requirements_lock.read_text(encoding="utf-8")
        assert "This file was autogenerated by uv" in text
        assert "--index-url https://pypi.org/simple" in text
        assert "--index-strategy first-index" in text.splitlines()[1]
        assert "--torch-backend " + definition.torch_backend in text.splitlines()[1]
        assert "# from " + definition.indexes[0] in text
        assert "--generate-hashes" in text

        blocks = _requirement_blocks(text)
        assert blocks
        for block in blocks:
            first_line = block.splitlines()[0]
            assert "==" in first_line or " @ https://" in first_line, first_line
            assert "git+" not in block
            assert " @ file:" not in block
            assert "-e " not in block


def test_nuplan_archive_and_hash_are_exact(manifest: EnvironmentManifest):
    """Diffusion/Sparse lock이 immutable nuPlan archive와 검증 hash를 공유하는지 검사한다."""
    archive = (
        "nuplan-devkit @ https://github.com/motional/nuplan-devkit/archive/"
        "ce3c323af01c0d7ec5672f7832ef53f9c679aab0.tar.gz"
    )
    digest = "5ab521c1af439902fd8e3df2619da75a03ade5814b35cd37c4c920fd45455e62"
    for model_id in ("diffusiondrive_v2", "sparsedrive_v2"):
        definition = manifest.models[model_id]
        input_text = definition.requirements_input.read_text(encoding="utf-8")
        lock_text = definition.requirements_lock.read_text(encoding="utf-8")
        assert archive in input_text.splitlines()
        assert archive in lock_text
        assert "--hash=sha256:" + digest in lock_text
