"""모델별 Python 환경 manifest와 lock 계약을 회귀 검사한다."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest

from validator_inference.model_environment import (
    ActiveEnvironmentCheck,
    AssetFingerprint,
    EnvironmentConfigError,
    EnvironmentManifest,
    EnvironmentPaths,
    EnvironmentState,
    EnvironmentStateError,
    ModelEnvironmentDefinition,
    build_uv_sync_argv,
    build_uv_venv_argv,
    environment_paths,
    load_environment_manifest,
    select_model_environment,
    sha256_file,
)


PACKAGE_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PACKAGE_ROOT / "config" / "model_environments.json"
EXPECTED = {
    "interfuser": {
        "python": "3.8.20",
        "torch": "torch==2.4.1+cu121",
        "torchvision": "torchvision==0.19.1+cu121",
        "cuda_policy": "recommended",
        "cuda_index": "https://download.pytorch.org/whl/cu121",
        "torch_backend": "cu121",
    },
    "tfv6": {
        "python": "3.10.20",
        "torch": "torch==2.5.0+cu124",
        "torchvision": "torchvision==0.20.0+cu124",
        "cuda_policy": "required",
        "cuda_index": "https://download.pytorch.org/whl/cu124",
        "torch_backend": "cu124",
    },
    "diffusiondrive_v2": {
        "python": "3.9.25",
        "torch": "torch==2.0.1+cu118",
        "torchvision": "torchvision==0.15.2+cu118",
        "cuda_policy": "recommended",
        "cuda_index": "https://download.pytorch.org/whl/cu118",
        "torch_backend": "cu118",
    },
    "sparsedrive_v2": {
        "python": "3.9.25",
        "torch": "torch==2.0.1+cu118",
        "torchvision": "torchvision==0.15.2+cu118",
        "cuda_policy": "extension_required",
        "cuda_index": "https://download.pytorch.org/whl/cu118",
        "torch_backend": "cu118",
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
    for model_id in EXPECTED:
        (requirements / f"{model_id}.in").write_text("fixture\n", encoding="utf-8")
        (requirements / f"{model_id}.lock").write_text("fixture\n", encoding="utf-8")
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
    """uv lock에서 각 installable requirement와 hash continuation을 묶는다."""
    blocks: list[str] = []
    current: list[str] = []
    for line in lock_text.splitlines():
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line[0].isspace():
            if current:
                current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
        current = [line]
    if current:
        blocks.append("\n".join(current))
    return blocks


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
        assert definition.indexes == (
            expected["cuda_index"],
            "https://pypi.org/simple",
        )
        assert expected["torch"] in requirement_text.splitlines()
        assert expected["torchvision"] in requirement_text.splitlines()
        assert COMMON_INPUTS <= set(requirement_text.splitlines())
        assert definition.requirements_input.is_file()
        assert definition.requirements_lock.is_file()

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
            assert "--hash=sha256:" in block, first_line
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
