"""모델 환경 candidate와 순수 Python ROS overlay 보안 계약을 검사한다."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
import typing
from typing import Any, Callable

import pytest

from validator_inference import environment_setup
from validator_inference.environment_setup import (
    PURE_ROS_PACKAGES,
    EnvironmentOperationResult,
    EnvironmentPreflight,
    EnvironmentPreflightDependencies,
    EnvironmentSetupDependencies,
    EnvironmentSetupError,
    RosOverlayFile,
    RosOverlayManifest,
    activate_candidate,
    build_ros_overlay,
    create_candidate,
    fingerprint_runtime_assets,
    preflight_environment,
    prepare_local_runtime_assets,
    prune_inactive_builds,
    setup_environment,
    verify_ros_overlay,
)
from validator_inference.model_environment import (
    ENVIRONMENT_STATE_FILENAME,
    ROS_OVERLAY_MANIFEST_FILENAME,
    AssetFingerprint,
    EnvironmentState,
    EnvironmentPaths,
    build_uv_sync_argv,
    build_uv_venv_argv,
    environment_paths,
    read_environment_state,
    sha256_file,
)


EXPECTED_ROS_PACKAGES = (
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


def _fake_ros_tree(root: Path) -> Path:
    """14개 허용 package와 일부 nested Python module을 만든다."""
    source = root / "ros-dist-packages"
    for package in EXPECTED_ROS_PACKAGES:
        package_root = source / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            f"PACKAGE = {package!r}\n", encoding="utf-8"
        )
    (source / "rospy" / "client.py").write_text("VALUE = 1\n", encoding="utf-8")
    message_root = source / "std_msgs" / "msg"
    message_root.mkdir()
    (message_root / "__init__.py").write_text(
        "from ._Header import Header\n", encoding="utf-8"
    )
    (message_root / "_Header.py").write_text("class Header:\n    pass\n", encoding="utf-8")
    return source


def _build_overlay(tmp_path: Path) -> tuple[Path, Path, RosOverlayManifest]:
    """정상 fake source로 overlay와 manifest를 구성한다."""
    source = _fake_ros_tree(tmp_path)
    destination = tmp_path / "candidate" / ".ros-overlay"
    manifest = build_ros_overlay(source, destination)
    manifest_path = destination.parent / ROS_OVERLAY_MANIFEST_FILENAME
    return source, destination, manifest


def _manifest_payload(manifest_path: Path) -> dict[str, Any]:
    """manifest JSON을 보안 변형 테스트용 dict로 읽는다."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest_payload(manifest_path: Path, payload: dict[str, Any]) -> None:
    """변형한 manifest payload를 JSON으로 기록한다."""
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_public_api_and_exact_ros_allowlist_are_stable(tmp_path: Path):
    """Task 4a 공개 API와 정확한 14-package allowlist를 고정한다."""
    assert PURE_ROS_PACKAGES == EXPECTED_ROS_PACKAGES
    assert issubclass(EnvironmentSetupError, RuntimeError)
    assert all(
        callable(value)
        for value in (
            create_candidate,
            build_ros_overlay,
            verify_ros_overlay,
            activate_candidate,
            prune_inactive_builds,
            environment_setup.model_operation_lock,
            environment_setup.verify_environment,
            environment_setup.prune_environment_builds,
            environment_setup.preflight_run_environment,
        )
    )

    record = RosOverlayFile(
        source_path=(tmp_path / "source.py").resolve(),
        destination_path=Path("rospy/source.py"),
        sha256="0" * 64,
    )
    manifest = RosOverlayManifest(
        schema=1,
        packages=("rospy",),
        files=(record,),
        manifest_path=(tmp_path / ROS_OVERLAY_MANIFEST_FILENAME).resolve(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.sha256 = "1" * 64  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.schema = 2  # type: ignore[misc]


def test_environment_setup_error_records_action():
    """domain 오류가 실패 action과 원인을 함께 노출한다."""
    error = EnvironmentSetupError("build_ros_overlay", "native extension found")
    assert error.action == "build_ros_overlay"
    assert str(error) == "build_ros_overlay: native extension found"


def test_create_candidate_makes_unique_immediate_build_directory(tmp_path: Path):
    """candidate를 움직이지 않을 최종 builds 경로에 한 번만 만든다."""
    paths = environment_paths(tmp_path, "interfuser")

    candidate = create_candidate(paths, "20260721T120000Z-abc123")

    assert candidate == paths.builds / "20260721T120000Z-abc123"
    assert candidate.is_dir()
    assert not candidate.is_symlink()
    with pytest.raises(EnvironmentSetupError, match="already exists"):
        create_candidate(paths, "20260721T120000Z-abc123")


@pytest.mark.parametrize(
    "build_id",
    ("", ".", "..", "nested/build", "/absolute", "back\\slash", " hidden", "a\n"),
)
def test_create_candidate_rejects_unsafe_build_ids(tmp_path: Path, build_id: str):
    """build ID가 경로 탈출이나 숨은 component를 만들 수 없게 한다."""
    paths = environment_paths(tmp_path, "interfuser")
    with pytest.raises(EnvironmentSetupError, match="build_id"):
        create_candidate(paths, build_id)
    assert not paths.builds.exists()


@pytest.mark.parametrize("component", ("venvs", "model", "builds"))
def test_create_candidate_rejects_symlinked_path_components(
    tmp_path: Path, component: str
):
    """candidate 상위 path의 기존 symlink를 따라 workspace 밖에 쓰지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    outside = tmp_path / "outside"
    outside.mkdir()
    if component == "venvs":
        paths.model_root.parent.symlink_to(outside, target_is_directory=True)
    elif component == "model":
        paths.model_root.parent.mkdir()
        paths.model_root.symlink_to(outside, target_is_directory=True)
    else:
        paths.model_root.mkdir(parents=True)
        paths.builds.symlink_to(outside, target_is_directory=True)

    with pytest.raises(EnvironmentSetupError, match="symlink"):
        create_candidate(paths, "build-1")
    assert not (outside / "build-1").exists()


def test_create_candidate_rejects_non_directory_and_untrusted_layout(tmp_path: Path):
    """기존 일반 파일과 model root에서 유도되지 않은 paths를 거부한다."""
    paths = environment_paths(tmp_path, "interfuser")
    paths.model_root.parent.mkdir()
    paths.model_root.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(EnvironmentSetupError, match="directory"):
        create_candidate(paths, "build-1")

    wrong_root = tmp_path / ".venvs" / "tfv6"
    wrong = EnvironmentPaths(
        model_root=wrong_root,
        builds=tmp_path / "foreign-builds",
        current=wrong_root / "current",
    )
    with pytest.raises(EnvironmentSetupError, match="layout"):
        create_candidate(wrong, "build-1")


def test_build_ros_overlay_copies_only_python_and_writes_sorted_manifest(
    tmp_path: Path,
):
    """허용 package의 Python source만 복사하고 file provenance를 정렬 기록한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / "README.txt").write_text("ignored\n", encoding="utf-8")
    (source / "rospy" / "cache.pyc").write_bytes(b"ignored")
    cache = source / "rospy" / "__pycache__"
    cache.mkdir()
    (cache / "client.cpython-38.pyc").write_bytes(b"ignored")
    destination = tmp_path / "candidate" / ".ros-overlay"

    manifest = build_ros_overlay(source, destination)

    manifest_path = destination.parent / ROS_OVERLAY_MANIFEST_FILENAME
    copied = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert copied == sorted(record.destination_path.as_posix() for record in manifest.files)
    assert "rospy/client.py" in copied
    assert not any("pycache" in path or path.endswith(".pyc") for path in copied)
    assert manifest.schema == 1
    assert manifest.packages == EXPECTED_ROS_PACKAGES
    assert manifest.manifest_path == manifest_path.resolve()
    assert tuple(record.destination_path.as_posix() for record in manifest.files) == tuple(
        sorted(copied)
    )
    for record in manifest.files:
        assert record.source_path.is_absolute()
        assert record.source_path == source / record.destination_path
        assert record.sha256 == sha256_file(destination / record.destination_path)

    payload = _manifest_payload(manifest_path)
    assert list(payload) == ["files", "packages", "schema"]
    assert payload["packages"] == list(EXPECTED_ROS_PACKAGES)
    assert [entry["destination"] for entry in payload["files"]] == sorted(copied)
    assert manifest_path.read_bytes().endswith(b"\n")


def test_build_ros_overlay_is_byte_deterministic(tmp_path: Path):
    """같은 source에서 만든 두 manifest가 destination 위치와 무관하게 동일하다."""
    source = _fake_ros_tree(tmp_path)
    first = tmp_path / "first" / ".ros-overlay"
    second = tmp_path / "second" / ".ros-overlay"

    build_ros_overlay(source, first)
    build_ros_overlay(source, second)

    assert (first.parent / ROS_OVERLAY_MANIFEST_FILENAME).read_bytes() == (
        second.parent / ROS_OVERLAY_MANIFEST_FILENAME
    ).read_bytes()


@pytest.mark.parametrize(
    "packages",
    (
        EXPECTED_ROS_PACKAGES[:-1],
        tuple(reversed(EXPECTED_ROS_PACKAGES)),
        (*EXPECTED_ROS_PACKAGES[:-1], "evil"),
    ),
)
def test_build_ros_overlay_requires_exact_allowlist(
    tmp_path: Path, packages: tuple[str, ...]
):
    """subset, reorder, 미선언 package를 exact ROS allowlist로 허용하지 않는다."""
    source = _fake_ros_tree(tmp_path)
    if "evil" in packages:
        evil = source / "evil"
        evil.mkdir()
        (evil / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(EnvironmentSetupError, match="exact"):
        build_ros_overlay(
            source,
            tmp_path / "candidate" / ".ros-overlay",
            packages=packages,
        )


@pytest.mark.parametrize(
    "kind",
    (
        "source-trailing",
        "source-dot",
        "source-duplicate",
        "destination-parent",
        "destination-backslash",
        "destination-control",
    ),
)
def test_build_ros_overlay_requires_canonical_absolute_paths(
    tmp_path: Path, kind: str
):
    """build root 인자의 alias·separator·control 변형을 canonical path로 거부한다."""
    source = _fake_ros_tree(tmp_path)
    destination = tmp_path / "candidate" / ".ros-overlay"
    source_arg: str | Path = source
    destination_arg: str | Path = destination
    if kind == "source-trailing":
        source_arg = str(source) + "/"
    elif kind == "source-dot":
        source_arg = f"{source.parent}/./{source.name}"
    elif kind == "source-duplicate":
        source_arg = str(source).replace("/ros-dist-packages", "//ros-dist-packages")
    elif kind == "destination-parent":
        destination_arg = f"{tmp_path}/candidate/nested/../.ros-overlay"
    elif kind == "destination-backslash":
        destination_arg = f"{tmp_path}/candidate/bad\\name/.ros-overlay"
    else:
        destination_arg = f"{tmp_path}/candidate/bad\nname/.ros-overlay"

    with pytest.raises(EnvironmentSetupError, match="canonical"):
        build_ros_overlay(source_arg, destination_arg)  # type: ignore[arg-type]


@pytest.mark.parametrize("filename", ("bad\\name.py", "bad\nname.py"))
def test_build_ros_overlay_rejects_noncanonical_source_entries(
    tmp_path: Path, filename: str
):
    """자기 생성 manifest에 비정규 source path가 기록되기 전에 source entry를 거부한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / filename).write_text("VALUE = 1\n", encoding="utf-8")
    destination = tmp_path / "candidate" / ".ros-overlay"

    with pytest.raises(EnvironmentSetupError, match="canonical"):
        build_ros_overlay(source, destination)
    assert not destination.exists()


def test_build_ros_overlay_requires_package_init_module(tmp_path: Path):
    """각 선언 ROS package가 import 가능한 일반 __init__.py를 포함해야 한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / "__init__.py").unlink()

    with pytest.raises(EnvironmentSetupError, match="__init__"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


@pytest.mark.parametrize(
    "statement",
    (
        "import rosgraph.missing_module\n",
        "from std_msgs.msg.missing_module import Missing\n",
        "from .missing_relative import Missing\n",
    ),
)
def test_build_ros_overlay_rejects_missing_selected_package_imports(
    tmp_path: Path, statement: str
):
    """선택 package 내부 absolute·relative import의 누락 module closure를 거부한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / "client.py").write_text(statement, encoding="utf-8")

    with pytest.raises(EnvironmentSetupError, match="import closure"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


def test_build_ros_overlay_accepts_external_optional_and_exported_symbol_imports(
    tmp_path: Path,
):
    """allowlist 밖 optional import와 from-dot exported symbol은 module로 오판하지 않는다."""
    source = _fake_ros_tree(tmp_path)
    with (source / "rospy" / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write("exported_symbol = 1\n")
    with (source / "rosgraph" / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write("exported_symbol = 1\n")
    (source / "rospy" / "client.py").write_text(
        "import rosnode\n"
        "import rosservice\n"
        "import em\n"
        "from . import exported_symbol\n"
        "from . import client\n"
        "from rosgraph import exported_symbol\n",
        encoding="utf-8",
    )

    manifest = build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")

    assert any(record.destination_path == Path("rospy/client.py") for record in manifest.files)


@pytest.mark.parametrize(
    "statement",
    ("from rosgraph import missing_export\n", "from . import missing_export\n"),
)
def test_build_ros_overlay_rejects_missing_ambiguous_import_aliases(
    tmp_path: Path, statement: str
):
    """submodule도 base export도 아닌 from-import alias를 closure 누락으로 거부한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / "client.py").write_text(statement, encoding="utf-8")

    with pytest.raises(EnvironmentSetupError, match="export"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


def test_build_ros_overlay_accepts_static_symbol_exports(tmp_path: Path):
    """top-level 정의·할당·import binding을 base module static export로 해석한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rosgraph" / "provider.py").write_text(
        "def provided():\n    pass\n", encoding="utf-8"
    )
    (source / "rosgraph" / "__init__.py").write_text(
        "def function_export():\n    pass\n"
        "class ClassExport:\n    pass\n"
        "assigned_export = 1\n"
        "annotated_export: int = 1\n"
        "import math as imported_export\n"
        "from math import pi as external_explicit_export\n"
        "from .provider import provided as explicit_export\n",
        encoding="utf-8",
    )
    (source / "rospy" / "client.py").write_text(
        "from rosgraph import function_export\n"
        "from rosgraph import ClassExport\n"
        "from rosgraph import assigned_export\n"
        "from rosgraph import annotated_export\n"
        "from rosgraph import imported_export\n"
        "from rosgraph import external_explicit_export\n"
        "from rosgraph import explicit_export\n",
        encoding="utf-8",
    )

    manifest = build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")

    assert any(
        record.destination_path == Path("rosgraph/provider.py")
        for record in manifest.files
    )


def test_build_ros_overlay_resolves_recursive_local_and_selected_wildcards(
    tmp_path: Path,
):
    """local·selected wildcard re-export chain에서 static export를 재귀 탐색한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rosgraph" / "__init__.py").write_text(
        "from .local_exports import *\n", encoding="utf-8"
    )
    (source / "rosgraph" / "local_exports.py").write_text(
        "from rosgraph.selected_exports import *\n", encoding="utf-8"
    )
    (source / "rosgraph" / "selected_exports.py").write_text(
        "wildcard_export = 1\n", encoding="utf-8"
    )
    (source / "rospy" / "client.py").write_text(
        "from rosgraph import wildcard_export\n", encoding="utf-8"
    )

    manifest = build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")

    assert manifest.packages == EXPECTED_ROS_PACKAGES


def test_build_ros_overlay_resolves_export_through_wildcard_cycle(tmp_path: Path):
    """wildcard cycle 안의 실제 static export는 cycle guard 중에도 찾는다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rosgraph" / "__init__.py").write_text(
        "from .cycle_a import *\n", encoding="utf-8"
    )
    (source / "rosgraph" / "cycle_a.py").write_text(
        "from .cycle_b import *\n", encoding="utf-8"
    )
    (source / "rosgraph" / "cycle_b.py").write_text(
        "from .cycle_a import *\ncycle_export = 1\n", encoding="utf-8"
    )
    (source / "rospy" / "client.py").write_text(
        "from rosgraph import cycle_export\n", encoding="utf-8"
    )

    manifest = build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")

    assert manifest.packages == EXPECTED_ROS_PACKAGES


def test_build_ros_overlay_rejects_missing_export_after_wildcard_cycle(tmp_path: Path):
    """wildcard re-export cycle을 종료하고 명백히 없는 export를 거부한다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rosgraph" / "__init__.py").write_text(
        "from .cycle import *\n", encoding="utf-8"
    )
    (source / "rosgraph" / "cycle.py").write_text(
        "from rosgraph import *\n", encoding="utf-8"
    )
    (source / "rospy" / "client.py").write_text(
        "from rosgraph import absent_after_cycle\n", encoding="utf-8"
    )

    with pytest.raises(EnvironmentSetupError, match="export"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


@pytest.mark.parametrize(
    "provider",
    (
        "from external_runtime_package import *\n",
        "def __getattr__(name):\n    return name\n",
        "if runtime_condition:\n    def __getattr__(name):\n        return name\n",
    ),
)
def test_build_ros_overlay_defers_unresolved_dynamic_exports(
    tmp_path: Path, provider: str
):
    """외부 wildcard와 module __getattr__ export는 runtime 판정으로 넘긴다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rosgraph" / "__init__.py").write_text(provider, encoding="utf-8")
    (source / "rospy" / "client.py").write_text(
        "from rosgraph import dynamic_export\n", encoding="utf-8"
    )

    manifest = build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")

    assert manifest.packages == EXPECTED_ROS_PACKAGES


def test_build_ros_overlay_rejects_python_syntax_errors(tmp_path: Path):
    """복사 대상 Python source가 AST parse되지 않으면 candidate를 만들지 않는다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / "client.py").write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(EnvironmentSetupError, match="syntax"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


@pytest.mark.parametrize("suffix", (".so", ".cpython-38-x86_64-linux-gnu.so", ".pyd", ".dylib", ".dll"))
def test_build_ros_overlay_rejects_native_extensions(tmp_path: Path, suffix: str):
    """허용 package 안의 Python ABI 또는 native library를 overlay에 넣지 않는다."""
    source = _fake_ros_tree(tmp_path)
    (source / "rospy" / f"native{suffix}").write_bytes(b"binary")
    destination = tmp_path / "candidate" / ".ros-overlay"

    with pytest.raises(EnvironmentSetupError, match="native"):
        build_ros_overlay(source, destination)
    assert not (destination.parent / ROS_OVERLAY_MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("kind", ("file", "directory", "package"))
def test_build_ros_overlay_rejects_source_symlinks(tmp_path: Path, kind: str):
    """source package tree의 어떤 symlink도 copy2가 따라가지 못하게 한다."""
    source = _fake_ros_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 2\n", encoding="utf-8")
    if kind == "file":
        (source / "rospy" / "linked.py").symlink_to(outside)
    elif kind == "directory":
        outside_dir = tmp_path / "outside-package"
        outside_dir.mkdir()
        (outside_dir / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
        (source / "rospy" / "linked").symlink_to(outside_dir, target_is_directory=True)
    else:
        real_package = source / "real-rospy"
        (source / "rospy").rename(real_package)
        (source / "rospy").symlink_to(real_package, target_is_directory=True)

    with pytest.raises(EnvironmentSetupError, match="symlink"):
        build_ros_overlay(source, tmp_path / "candidate" / ".ros-overlay")


@pytest.mark.parametrize("kind", ("missing", "empty", "source-file", "existing-destination"))
def test_build_ros_overlay_rejects_invalid_roots_and_packages(
    tmp_path: Path, kind: str
):
    """source root와 모든 선언 package가 새 overlay를 구성할 수 있어야 한다."""
    source = _fake_ros_tree(tmp_path)
    destination = tmp_path / "candidate" / ".ros-overlay"
    if kind == "missing":
        (source / "rospy" / "__init__.py").unlink()
        (source / "rospy" / "client.py").unlink()
        (source / "rospy").rmdir()
    elif kind == "empty":
        for path in (source / "rospy").iterdir():
            path.unlink()
    elif kind == "source-file":
        for path in sorted(source.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        source.rmdir()
        source.write_text("not a directory\n", encoding="utf-8")
    else:
        destination.mkdir(parents=True)

    with pytest.raises(EnvironmentSetupError):
        build_ros_overlay(source, destination)


def test_verify_ros_overlay_returns_manifest_digest(tmp_path: Path):
    """정상 overlay 전체를 검증한 뒤 manifest 파일 자체 SHA256을 반환한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    manifest_path = manifest.manifest_path

    assert verify_ros_overlay(destination, manifest_path) == sha256_file(manifest_path)


def test_verify_ros_overlay_hashes_open_file_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """verify가 검증 뒤 path를 sha256_file로 다시 열지 않고 같은 fd를 hash한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    expected = sha256_file(manifest.manifest_path)

    def reject_path_hash(path: Path) -> str:
        """verify 중 path 기반 reopen을 즉시 실패시킨다."""
        raise AssertionError(f"path hash attempted: {path}")

    monkeypatch.setattr(environment_setup, "sha256_file", reject_path_hash)

    assert verify_ros_overlay(destination, manifest.manifest_path) == expected


def test_verify_ros_overlay_rejects_fifo_swapped_before_final_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """manifest final component가 open 직전 FIFO로 바뀌어도 block 없이 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        """manifest final open 직전에 regular file을 FIFO로 교체한다."""
        nonlocal swapped
        if path == manifest.manifest_path.name and dir_fd is not None and not swapped:
            swapped = True
            manifest.manifest_path.unlink()
            os.mkfifo(str(manifest.manifest_path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(environment_setup.os, "open", swapping_open)

    with pytest.raises(EnvironmentSetupError, match="regular"):
        verify_ros_overlay(destination, manifest.manifest_path)
    assert swapped


def test_verify_ros_overlay_rejects_file_symlink_swap_before_final_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """hash 대상 final component의 symlink swap을 O_NOFOLLOW open으로 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    target = destination / "rospy" / "client.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(target.read_bytes())
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        """destination client.py final open 직전에 동일 내용 symlink로 교체한다."""
        nonlocal swapped
        if path == target.name and dir_fd is not None and not swapped:
            parent = Path(os.readlink(f"/proc/self/fd/{dir_fd}"))
            if parent == target.parent:
                swapped = True
                target.unlink()
                target.symlink_to(outside)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(environment_setup.os, "open", swapping_open)

    with pytest.raises(EnvironmentSetupError, match="symlink"):
        verify_ros_overlay(destination, manifest.manifest_path)
    assert swapped


def test_verify_ros_overlay_requires_fixed_sibling_manifest_path(tmp_path: Path):
    """검증할 manifest를 overlay 형제 고정 이름 밖에서 주입하지 못하게 한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    foreign = tmp_path / "foreign-manifest.json"
    foreign.write_bytes(manifest.manifest_path.read_bytes())

    with pytest.raises(EnvironmentSetupError, match="manifest path"):
        verify_ros_overlay(destination, foreign)


@pytest.mark.parametrize(
    "mutation,reason",
    (
        ("modified", "hash"),
        ("missing", "missing"),
        ("extra-file", "extra"),
        ("extra-directory", "extra"),
        ("native", "native"),
        ("file-symlink", "symlink"),
        ("directory-symlink", "symlink"),
    ),
)
def test_verify_ros_overlay_rejects_tree_changes(
    tmp_path: Path, mutation: str, reason: str
):
    """manifest에 없는 추가 entry와 누락·변조·symlink를 모두 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    target = destination / "rospy" / "client.py"
    if mutation == "modified":
        target.write_text("VALUE = 999\n", encoding="utf-8")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra-file":
        (destination / "rospy" / "extra.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "extra-directory":
        (destination / "rospy" / "extra").mkdir()
    elif mutation == "native":
        (destination / "rospy" / "native.so").write_bytes(b"binary")
    elif mutation == "file-symlink":
        target.unlink()
        target.symlink_to(destination / "rosgraph" / "__init__.py")
    else:
        (destination / "rospy" / "linked").symlink_to(
            destination / "rosgraph", target_is_directory=True
        )

    with pytest.raises(EnvironmentSetupError, match=reason):
        verify_ros_overlay(destination, manifest.manifest_path)


@pytest.mark.parametrize("target", ("destination", "manifest", "package"))
def test_verify_ros_overlay_rejects_symlinked_boundary_paths(
    tmp_path: Path, target: str
):
    """overlay root, manifest, package directory의 symlink 경계를 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    manifest_path = manifest.manifest_path
    if target == "destination":
        relocated = destination.parent / "relocated-overlay"
        destination.rename(relocated)
        destination.symlink_to(relocated, target_is_directory=True)
    elif target == "manifest":
        relocated = manifest_path.with_name("relocated-manifest.json")
        manifest_path.rename(relocated)
        manifest_path.symlink_to(relocated)
    else:
        package = destination / "rospy"
        relocated = destination / "relocated-rospy"
        package.rename(relocated)
        package.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(EnvironmentSetupError, match="symlink"):
        verify_ros_overlay(destination, manifest_path)


def _invalid_manifest_cases(
    source_root: Path,
) -> tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...]:
    """strict JSON/parser 경계를 변형하는 함수 목록을 반환한다."""

    def unknown_top(value: dict[str, Any]) -> None:
        """알 수 없는 top-level key를 추가한다."""
        value["unexpected"] = True

    def wrong_schema(value: dict[str, Any]) -> None:
        """지원하지 않는 schema로 바꾼다."""
        value["schema"] = 2

    def duplicate_package(value: dict[str, Any]) -> None:
        """package 이름을 중복한다."""
        value["packages"].append(value["packages"][0])

    def unsafe_package(value: dict[str, Any]) -> None:
        """package component에 경로 separator를 넣는다."""
        value["packages"][0] = "../rospy"

    def unknown_file_key(value: dict[str, Any]) -> None:
        """file record에 알 수 없는 key를 추가한다."""
        value["files"][0]["unexpected"] = True

    def unsafe_destination(value: dict[str, Any]) -> None:
        """destination을 overlay 밖 상대 경로로 바꾼다."""
        value["files"][0]["destination"] = "../escape.py"

    def absolute_destination(value: dict[str, Any]) -> None:
        """destination을 절대 경로로 바꾼다."""
        value["files"][0]["destination"] = "/tmp/escape.py"

    def relative_source(value: dict[str, Any]) -> None:
        """source provenance를 상대 경로로 바꾼다."""
        value["files"][0]["source"] = "relative/source.py"

    def mismatched_source(value: dict[str, Any]) -> None:
        """source suffix가 destination과 다른 파일을 가리키게 한다."""
        value["files"][0]["source"] = str((source_root / "other.py").resolve())

    def invalid_sha(value: dict[str, Any]) -> None:
        """file hash를 SHA256이 아닌 값으로 바꾼다."""
        value["files"][0]["sha256"] = "not-a-sha"

    def duplicate_destination(value: dict[str, Any]) -> None:
        """동일 destination record를 중복한다."""
        value["files"].append(dict(value["files"][0]))

    def undeclared_package_file(value: dict[str, Any]) -> None:
        """file destination의 package를 선언 목록 밖으로 바꾼다."""
        value["files"][0]["destination"] = "evil/source.py"

    return (
        ("unknown", unknown_top, "unknown"),
        ("schema", wrong_schema, "schema"),
        ("duplicate-package", duplicate_package, "duplicate"),
        ("unsafe-package", unsafe_package, "package"),
        ("unknown-file", unknown_file_key, "unknown"),
        ("unsafe-destination", unsafe_destination, "destination"),
        ("absolute-destination", absolute_destination, "destination"),
        ("relative-source", relative_source, "source"),
        ("mismatched-source", mismatched_source, "source"),
        ("sha", invalid_sha, "SHA"),
        ("duplicate-destination", duplicate_destination, "duplicate"),
        ("undeclared-package", undeclared_package_file, "package"),
    )


def test_verify_ros_overlay_rejects_strict_manifest_violations(tmp_path: Path):
    """unknown field, schema, path 탈출, 중복, 잘못된 hash를 각각 거부한다."""
    source, destination, manifest = _build_overlay(tmp_path)
    original = _manifest_payload(manifest.manifest_path)
    for _name, mutate, reason in _invalid_manifest_cases(source):
        payload = json.loads(json.dumps(original))
        mutate(payload)
        _write_manifest_payload(manifest.manifest_path, payload)
        with pytest.raises(EnvironmentSetupError, match=reason):
            verify_ros_overlay(destination, manifest.manifest_path)
        _write_manifest_payload(manifest.manifest_path, original)


@pytest.mark.parametrize(
    "packages",
    (
        list(EXPECTED_ROS_PACKAGES[:-1]),
        list(reversed(EXPECTED_ROS_PACKAGES)),
        [*EXPECTED_ROS_PACKAGES[:-1], "evil"],
    ),
)
def test_verify_ros_overlay_requires_exact_manifest_allowlist(
    tmp_path: Path, packages: list[str]
):
    """manifest package 목록도 exact 14-package 순서와 값에서 벗어나지 못한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    payload = _manifest_payload(manifest.manifest_path)
    payload["packages"] = packages
    _write_manifest_payload(manifest.manifest_path, payload)

    with pytest.raises(EnvironmentSetupError, match="exact"):
        verify_ros_overlay(destination, manifest.manifest_path)


@pytest.mark.parametrize(
    "variant",
    ("dot", "parent", "duplicate", "trailing", "backslash", "control"),
)
def test_verify_ros_overlay_rejects_noncanonical_manifest_source_paths(
    tmp_path: Path, variant: str
):
    """manifest source absolute path의 모든 lexical alias와 control 변형을 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    payload = _manifest_payload(manifest.manifest_path)
    source = payload["files"][0]["source"]
    if variant == "dot":
        changed = source.replace("/actionlib_msgs/", "/actionlib_msgs/./")
    elif variant == "parent":
        changed = source.replace(
            "/actionlib_msgs/", "/actionlib_msgs/../actionlib_msgs/"
        )
    elif variant == "duplicate":
        changed = source.replace("/actionlib_msgs/", "//actionlib_msgs/")
    elif variant == "trailing":
        changed = source + "/"
    elif variant == "backslash":
        changed = source.replace("/actionlib_msgs/", "/actionlib_msgs\\")
    else:
        changed = source.replace("/actionlib_msgs/", "/actionlib_msgs\n/")
    payload["files"][0]["source"] = changed
    _write_manifest_payload(manifest.manifest_path, payload)

    with pytest.raises(EnvironmentSetupError, match="canonical"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_rejects_duplicate_json_keys(tmp_path: Path):
    """JSON decoder가 덮어쓸 duplicate key를 손상 manifest로 분류한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    manifest.manifest_path.write_text(
        '{"schema":1,"schema":1,"packages":[],"files":[]}\n', encoding="utf-8"
    )
    with pytest.raises(EnvironmentSetupError, match="duplicate"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_rejects_fifo_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """manifest FIFO를 read_bytes로 열어 block하기 전에 lstat 형식으로 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    manifest.manifest_path.unlink()
    os.mkfifo(str(manifest.manifest_path))
    real_read_bytes = Path.read_bytes

    def reject_fifo_read(path: Path) -> bytes:
        """FIFO read 시도를 즉시 노출하고 일반 파일은 기존 구현으로 읽는다."""
        if path == manifest.manifest_path:
            raise AssertionError("FIFO read attempted")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_fifo_read)

    with pytest.raises(EnvironmentSetupError, match="regular"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_normalizes_recursive_json(tmp_path: Path):
    """decoder recursion 한계를 넘는 manifest를 domain 오류로 정규화한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    manifest.manifest_path.write_text(
        "[" * 2000 + "0" + "]" * 2000 + "\n", encoding="utf-8"
    )

    with pytest.raises(EnvironmentSetupError, match="JSON"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_rejects_oversized_manifest_bytes(tmp_path: Path):
    """JSON whitespace도 포함한 manifest 전체 byte 수를 read 전에 제한한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    encoded = manifest.manifest_path.read_bytes()
    padding = environment_setup._MAX_MANIFEST_BYTES - len(encoded) + 1
    manifest.manifest_path.write_bytes(b" " * padding + encoded)

    with pytest.raises(EnvironmentSetupError, match="manifest size limit"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_rejects_too_many_flat_records(tmp_path: Path):
    """깊지 않은 JSON도 file record 상한을 넘으면 collection 생성 전에 거부한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    payload = _manifest_payload(manifest.manifest_path)
    record = payload["files"][0]
    payload["files"] = [
        dict(record) for _index in range(environment_setup._MAX_MANIFEST_RECORDS + 1)
    ]
    _write_manifest_payload(manifest.manifest_path, payload)

    with pytest.raises(EnvironmentSetupError, match="record limit"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_rejects_oversized_path_strings(tmp_path: Path):
    """canonical 형태여도 manifest provenance path string 길이를 제한한다."""
    _, destination, manifest = _build_overlay(tmp_path)
    payload = _manifest_payload(manifest.manifest_path)
    suffix = payload["files"][0]["destination"]
    payload["files"][0]["source"] = (
        "/" + "a" * environment_setup._MAX_PATH_STRING_LENGTH + "/" + suffix
    )
    _write_manifest_payload(manifest.manifest_path, payload)

    with pytest.raises(EnvironmentSetupError, match="path length limit"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_verify_ros_overlay_normalizes_parser_memory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """JSON parser의 MemoryError를 process 예외 대신 manifest domain 오류로 바꾼다."""
    _, destination, manifest = _build_overlay(tmp_path)

    def fail_json_loads(*_args: object, **_kwargs: object) -> object:
        """JSON decode allocation 실패를 주입한다."""
        raise MemoryError("injected parser allocation failure")

    monkeypatch.setattr(environment_setup.json, "loads", fail_json_loads)

    with pytest.raises(EnvironmentSetupError, match="JSON"):
        verify_ros_overlay(destination, manifest.manifest_path)


def test_activate_candidate_uses_relative_atomic_current_link(tmp_path: Path):
    """검증된 immediate candidate를 상대 symlink로 선택한다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "build-1")

    activate_candidate(paths, candidate)

    assert paths.current.is_symlink()
    assert Path(os.readlink(str(paths.current))) == Path("builds/build-1")
    assert paths.current.resolve() == candidate.resolve()


@pytest.mark.parametrize("kind", ("outside", "nested", "symlink", "missing"))
def test_activate_candidate_rejects_invalid_candidate(tmp_path: Path, kind: str):
    """builds immediate real directory가 아닌 candidate를 선택하지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    paths.builds.mkdir(parents=True)
    if kind == "outside":
        candidate = tmp_path / "outside"
        candidate.mkdir()
    elif kind == "nested":
        candidate = paths.builds / "group" / "build-1"
        candidate.mkdir(parents=True)
    elif kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        candidate = paths.builds / "build-1"
        candidate.symlink_to(outside, target_is_directory=True)
    else:
        candidate = paths.builds / "missing"

    with pytest.raises(EnvironmentSetupError, match="candidate"):
        activate_candidate(paths, candidate)
    assert not paths.current.exists()
    assert not paths.current.is_symlink()


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_activate_candidate_does_not_replace_non_symlink_current(
    tmp_path: Path, kind: str
):
    """사용자 파일이나 directory인 current를 os.replace로 덮지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "build-1")
    if kind == "file":
        paths.current.write_text("keep\n", encoding="utf-8")
    else:
        paths.current.mkdir()

    with pytest.raises(EnvironmentSetupError, match="current"):
        activate_candidate(paths, candidate)
    assert not paths.current.is_symlink()
    assert paths.current.exists()


def test_activate_candidate_preserves_old_current_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """마지막 atomic replace 실패 시 기존 current와 temp cleanup을 보장한다."""
    paths = environment_paths(tmp_path, "interfuser")
    old = create_candidate(paths, "old")
    new = create_candidate(paths, "new")
    paths.current.symlink_to(Path("builds/old"))
    real_replace = os.replace

    def fail_current_replace(source: str | Path, destination: str | Path) -> None:
        """current 교체만 실패시키고 다른 replace는 실제 구현에 위임한다."""
        if Path(destination) == paths.current:
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(environment_setup.os, "replace", fail_current_replace)

    with pytest.raises(EnvironmentSetupError, match="activate"):
        activate_candidate(paths, new)
    assert Path(os.readlink(str(paths.current))) == Path("builds/old")
    assert paths.current.resolve() == old.resolve()
    assert not list(paths.model_root.glob(".current.*"))


def test_activate_candidate_fsyncs_model_root_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """current atomic replace 뒤 model root directory를 fsync해 이름 변경을 내구화한다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "candidate")
    events: list[str] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        """current replace 순서를 기록하고 실제 교체를 수행한다."""
        if Path(destination) == paths.current:
            events.append("replace")
        real_replace(source, destination)

    def recording_fsync(file_descriptor: int) -> None:
        """directory fsync 순서를 기록하고 실제 fsync를 수행한다."""
        events.append("fsync")
        real_fsync(file_descriptor)

    monkeypatch.setattr(environment_setup.os, "replace", recording_replace)
    monkeypatch.setattr(environment_setup.os, "fsync", recording_fsync)

    activate_candidate(paths, candidate)

    assert events == ["replace", "fsync"]


def test_activate_candidate_cleans_temp_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """replace 뒤 directory fsync 실패를 domain 오류로 바꾸고 temp link를 남기지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "candidate")

    def fail_fsync(_file_descriptor: int) -> None:
        """model root directory durability 실패를 주입한다."""
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(environment_setup.os, "fsync", fail_fsync)

    with pytest.raises(EnvironmentSetupError, match="fsync"):
        activate_candidate(paths, candidate)
    assert Path(os.readlink(str(paths.current))) == Path("builds/candidate")
    assert not list(paths.model_root.glob(".current.*"))


def test_restore_current_snapshot_preserves_unrelated_link_raced_before_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """rollback isolation 직전 교체된 unrelated current를 덮어쓰지 않는다."""
    project = tmp_path / "project"
    project.mkdir()
    paths = environment_paths(project, "tfv6")
    old = create_candidate(paths, "old-build")
    candidate = create_candidate(paths, "candidate-build")
    unrelated = create_candidate(paths, "unrelated-build")
    activate_candidate(paths, candidate)
    real_rename = os.rename
    raced = False

    def replace_before_isolation(
        source: str | bytes,
        destination: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal raced
        if source == "current" and src_dir_fd is not None and not raced:
            raced = True
            paths.current.unlink()
            paths.current.symlink_to(Path("builds") / unrelated.name)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(environment_setup.os, "rename", replace_before_isolation)

    with pytest.raises(EnvironmentSetupError, match="changed|unrelated"):
        environment_setup._restore_current_link_snapshot(
            paths,
            candidate,
            (Path("builds") / old.name).as_posix(),
        )

    assert os.readlink(paths.current) == (
        Path("builds") / unrelated.name
    ).as_posix()


def test_model_operation_lock_is_reentrant_across_lifecycle_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """outer setup lease 안의 activate/prune가 같은 flock을 재진입해 한 번만 획득한다."""
    paths = environment_paths(tmp_path, "interfuser")
    calls: list[int] = []
    real_flock = environment_setup.fcntl.flock

    def recording_flock(file_descriptor: int, operation: int) -> None:
        """실제 flock을 수행하며 outer lease의 acquire/release만 기록한다."""
        calls.append(operation)
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(environment_setup.fcntl, "flock", recording_flock)

    with environment_setup.model_operation_lock(paths):
        active = create_candidate(paths, "active")
        inactive = create_candidate(paths, "inactive")
        activate_candidate(paths, active)
        assert prune_inactive_builds(paths) == (inactive,)

    assert calls == [environment_setup.fcntl.LOCK_EX, environment_setup.fcntl.LOCK_UN]


def test_outer_model_operation_lease_blocks_concurrent_prune(tmp_path: Path):
    """create부터 activate까지 잡은 outer lease 동안 같은 process prune thread를 막는다."""
    paths = environment_paths(tmp_path, "interfuser")
    prune_finished = threading.Event()
    release_prune = threading.Event()
    errors: list[BaseException] = []
    removed: list[tuple[Path, ...]] = []

    def run_prune() -> None:
        """outer lease와 경쟁하는 prune 결과를 수집한다."""
        try:
            removed.append(prune_inactive_builds(paths))
        except BaseException as exc:
            errors.append(exc)
        finally:
            prune_finished.set()
            release_prune.set()

    with environment_setup.model_operation_lock(paths):
        candidate = create_candidate(paths, "candidate")
        prune_thread = threading.Thread(target=run_prune)
        prune_thread.start()
        assert not prune_finished.wait(0.2)
        activate_candidate(paths, candidate)
    assert release_prune.wait(5.0)
    prune_thread.join(5.0)

    assert not prune_thread.is_alive()
    assert errors == []
    assert removed == [()]
    assert paths.current.resolve() == candidate.resolve()


def test_model_operation_lock_blocks_a_separate_process(tmp_path: Path):
    """process-local RLock 밖에서도 lifecycle file flock이 child process를 막는다."""
    paths = environment_paths(tmp_path, "interfuser")
    started = tmp_path / "child-started"
    acquired = tmp_path / "child-acquired"
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "from validator_inference.environment_setup import model_operation_lock\n"
        "from validator_inference.model_environment import environment_paths\n"
        "project, started, acquired = map(Path, sys.argv[1:])\n"
        "started.write_text('started\\n', encoding='utf-8')\n"
        "with model_operation_lock(environment_paths(project, 'interfuser')):\n"
        "    acquired.write_text('acquired\\n', encoding='utf-8')\n"
    )
    process: subprocess.Popen[str] | None = None
    try:
        with environment_setup.model_operation_lock(paths):
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path), str(started), str(acquired)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _attempt in range(100):
                if started.exists():
                    break
                time.sleep(0.01)
            assert started.exists()
            time.sleep(0.1)
            assert not acquired.exists()
        stdout, stderr = process.communicate(timeout=5.0)
        assert process.returncode == 0, stdout + stderr
        assert acquired.is_file()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)


def test_activate_and_prune_share_lifecycle_lock_and_secure_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """activation과 prune가 같은 0600 regular lock을 acquire/release하는지 검사한다."""
    paths = environment_paths(tmp_path, "interfuser")
    active = create_candidate(paths, "active")
    inactive = create_candidate(paths, "inactive")
    calls: list[int] = []
    real_flock = environment_setup.fcntl.flock

    def recording_flock(file_descriptor: int, operation: int) -> None:
        """실제 Linux flock을 수행하면서 operation 순서를 기록한다."""
        calls.append(operation)
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(environment_setup.fcntl, "flock", recording_flock)

    activate_candidate(paths, active)
    assert prune_inactive_builds(paths) == (inactive,)

    assert calls == [
        environment_setup.fcntl.LOCK_EX,
        environment_setup.fcntl.LOCK_UN,
        environment_setup.fcntl.LOCK_EX,
        environment_setup.fcntl.LOCK_UN,
    ]
    lock_path = paths.model_root / ".lifecycle.lock"
    assert lock_path.is_file()
    assert not lock_path.is_symlink()
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.parent == paths.model_root
    assert lock_path.parent != paths.builds


def test_lifecycle_lock_errors_preserve_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """lock 획득 실패를 domain 오류로 바꾸고 current를 변경하지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "candidate")

    def fail_lock(_file_descriptor: int, operation: int) -> None:
        """exclusive acquisition만 injected OSError로 실패시킨다."""
        if operation == environment_setup.fcntl.LOCK_EX:
            raise OSError("injected lock failure")

    monkeypatch.setattr(environment_setup.fcntl, "flock", fail_lock)

    with pytest.raises(EnvironmentSetupError, match="lock"):
        activate_candidate(paths, candidate)
    assert not paths.current.exists()
    assert not paths.current.is_symlink()


def test_lifecycle_lock_rejects_symlink(tmp_path: Path):
    """O_NOFOLLOW lifecycle lock이 model root 밖 symlink target을 열지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "candidate")
    outside = tmp_path / "outside-lock"
    outside.write_text("keep\n", encoding="utf-8")
    (paths.model_root / ".lifecycle.lock").symlink_to(outside)

    with pytest.raises(EnvironmentSetupError, match="lock"):
        activate_candidate(paths, candidate)
    assert outside.read_text(encoding="utf-8") == "keep\n"


def test_lifecycle_lock_prevents_activation_prune_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """activation replace 중 prune가 새 active candidate를 삭제하지 못하게 직렬화한다."""
    paths = environment_paths(tmp_path, "interfuser")
    old = create_candidate(paths, "old")
    new = create_candidate(paths, "new")
    activate_candidate(paths, old)
    replace_entered = threading.Event()
    allow_replace = threading.Event()
    prune_finished = threading.Event()
    errors: list[BaseException] = []
    removed: list[tuple[Path, ...]] = []
    real_replace = os.replace

    def blocking_replace(source: str | Path, destination: str | Path) -> None:
        """current 교체 직전 activation을 lock 안에서 일시 정지한다."""
        if Path(destination) == paths.current:
            replace_entered.set()
            if not allow_replace.wait(5.0):
                raise RuntimeError("timed out waiting to finish activation")
        real_replace(source, destination)

    def run_activate() -> None:
        """thread에서 새 candidate activation 예외를 수집한다."""
        try:
            activate_candidate(paths, new)
        except BaseException as exc:
            errors.append(exc)

    def run_prune() -> None:
        """thread에서 prune 결과와 예외를 수집한다."""
        try:
            removed.append(prune_inactive_builds(paths))
        except BaseException as exc:
            errors.append(exc)
        finally:
            prune_finished.set()

    monkeypatch.setattr(environment_setup.os, "replace", blocking_replace)
    activate_thread = threading.Thread(target=run_activate)
    activate_thread.start()
    assert replace_entered.wait(5.0)
    prune_thread = threading.Thread(target=run_prune)
    prune_thread.start()
    assert not prune_finished.wait(0.2)
    allow_replace.set()
    activate_thread.join(5.0)
    prune_thread.join(5.0)

    assert not activate_thread.is_alive()
    assert not prune_thread.is_alive()
    assert errors == []
    assert removed == [(old,)]
    assert Path(os.readlink(str(paths.current))) == Path("builds/new")
    assert new.is_dir()
    assert not old.exists()


def test_prune_inactive_builds_keeps_active_and_returns_sorted_paths(tmp_path: Path):
    """active candidate를 보존하고 inactive build만 이름 순서대로 지운다."""
    paths = environment_paths(tmp_path, "interfuser")
    second = create_candidate(paths, "b-second")
    active = create_candidate(paths, "m-active")
    first = create_candidate(paths, "a-first")
    paths.current.symlink_to(Path("builds/m-active"))

    removed = prune_inactive_builds(paths)

    assert removed == (first, second)
    assert active.is_dir()
    assert paths.current.resolve() == active.resolve()
    assert not first.exists()
    assert not second.exists()


def test_prune_inactive_builds_handles_missing_current(tmp_path: Path):
    """current가 없으면 모든 정상 inactive candidate를 명시적으로 정리한다."""
    paths = environment_paths(tmp_path, "interfuser")
    candidate = create_candidate(paths, "build-1")

    assert prune_inactive_builds(paths) == (candidate,)
    assert not candidate.exists()


def test_prune_inactive_builds_rejects_inode_replacement_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """scan한 inactive directory가 삭제 전 교체되면 새 inode를 지우지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    victim = create_candidate(paths, "victim")
    displaced = paths.builds / "displaced"
    real_lstat = Path.lstat
    swapped = False

    def swapping_lstat(path: Path) -> os.stat_result:
        """victim의 삭제 전 revalidation 시 directory inode를 교체한다."""
        nonlocal swapped
        if path == victim and not swapped:
            swapped = True
            victim.rename(displaced)
            victim.mkdir()
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", swapping_lstat)

    with pytest.raises(EnvironmentSetupError, match="changed since scan"):
        prune_inactive_builds(paths)
    assert swapped
    assert victim.is_dir()
    assert displaced.is_dir()


@pytest.mark.parametrize("kind", ("symlink", "file", "malformed-current", "dangling-current"))
def test_prune_inactive_builds_rejects_suspicious_entries_before_deleting(
    tmp_path: Path, kind: str
):
    """의심 entry를 발견하면 어떤 정상 build도 먼저 삭제하지 않는다."""
    paths = environment_paths(tmp_path, "interfuser")
    keep = create_candidate(paths, "keep")
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (paths.builds / "suspicious").symlink_to(outside, target_is_directory=True)
    elif kind == "file":
        (paths.builds / "suspicious").write_text("not a build\n", encoding="utf-8")
    elif kind == "malformed-current":
        paths.current.symlink_to(tmp_path / "outside")
    else:
        paths.current.symlink_to(Path("builds/missing"))

    with pytest.raises(EnvironmentSetupError):
        prune_inactive_builds(paths)
    assert keep.is_dir()


def test_prune_inactive_builds_returns_empty_for_absent_builds(tmp_path: Path):
    """아직 model environment가 없으면 filesystem을 만들지 않고 종료한다."""
    paths = environment_paths(tmp_path, "interfuser")
    assert prune_inactive_builds(paths) == ()
    assert not paths.model_root.exists()


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PREFLIGHT_WARNING = (
    "CUDA GPU is not available; continuing because {model_id} declares "
    "cuda_policy=recommended"
)


class _RecordingPreflightRunner:
    """호스트 probe의 argv와 subprocess keyword를 기록하는 fake runner다."""

    def __init__(
        self,
        *,
        uv_output: str = "uv 0.11.14 (x86_64-unknown-linux-gnu)\n",
        nvcc_output: str = "Cuda compilation tools, release 11.8, V11.8.89\n",
        gcc_output: str = "9.4.0\n",
        gxx_output: str = "9.4.0\n",
    ) -> None:
        self.uv_output = uv_output
        self.nvcc_output = nvcc_output
        self.gcc_output = gcc_output
        self.gxx_output = gxx_output
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.failure: BaseException | None = None

    def __call__(self, argv: list[str] | tuple[str, ...], **kwargs: Any):
        assert isinstance(argv, (list, tuple))
        command = tuple(argv)
        self.calls.append((command, dict(kwargs)))
        if self.failure is not None:
            raise self.failure
        executable = Path(command[0]).name
        if executable == "uv" and command[1:] == ("--version",):
            stdout = self.uv_output
        elif executable == "nvcc" and command[1:] == ("--version",):
            stdout = self.nvcc_output
        elif executable == "gcc" and command[1:] == (
            "-dumpfullversion",
            "-dumpversion",
        ):
            stdout = self.gcc_output
        elif executable == "g++" and command[1:] == (
            "-dumpfullversion",
            "-dumpversion",
        ):
            stdout = self.gxx_output
        else:
            raise AssertionError(f"unexpected preflight command: {command!r}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _write_executable(path: Path) -> Path:
    """fake command를 기존 executable regular file로 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _complete_sparse_calibration(profile_path: Path) -> None:
    """SparseDrive 테스트 profile의 세 카메라 calibration을 완성한다."""
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    sparse = next(item for item in payload["profiles"] if item["id"] == "sparsedrive_v2")
    cameras = sparse["static_runtime_inputs"]["camera_calibration"]["cameras"]
    identity = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    for camera in cameras:
        camera["K_original_3x3"] = [
            800.0,
            0.0,
            320.0,
            0.0,
            800.0,
            240.0,
            0.0,
            0.0,
            1.0,
        ]
        camera["distortion_model"] = "plumb_bob"
        camera["distortion_coefficients"] = [0.0] * 5
        camera["T_lidar_from_camera_4x4"] = identity
    profile_path.write_text(json.dumps(payload), encoding="utf-8")


def _preflight_project(tmp_path: Path, *, complete_sparse: bool = True) -> Path:
    """권위 manifest, lock, input과 profile을 가진 최소 project tree를 만든다."""
    root = tmp_path / "project"
    package_root = root / "src" / "Validator_pkg" / "validator_inference"
    package_config = package_root / "config"
    package_config.mkdir(parents=True)
    shutil.copy2(
        _REPOSITORY_ROOT
        / "src"
        / "Validator_pkg"
        / "validator_inference"
        / "config"
        / "model_environments.json",
        package_config / "model_environments.json",
    )
    shutil.copytree(
        _REPOSITORY_ROOT
        / "src"
        / "Validator_pkg"
        / "validator_inference"
        / "requirements",
        package_root / "requirements",
    )
    (package_root / "src" / "validator_inference").mkdir(parents=True)
    (package_root / "src" / "validator_inference" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    profile_root = root / "src" / "Validator_pkg" / "config"
    profile_root.mkdir()
    profile_path = profile_root / "profile.json"
    shutil.copy2(
        _REPOSITORY_ROOT / "src" / "Validator_pkg" / "config" / "profile.json",
        profile_path,
    )
    if complete_sparse:
        _complete_sparse_calibration(profile_path)
    return root


def _preflight_host(
    tmp_path: Path,
    *,
    gpu_available: bool = True,
) -> tuple[EnvironmentPreflightDependencies, _RecordingPreflightRunner, dict[str, str]]:
    """실제 host와 독립적인 Ubuntu/ROS/CUDA probe bundle을 만든다."""
    host = tmp_path / "host"
    ros_root = host / "opt" / "ros" / "noetic"
    ros_python_root = ros_root / "lib" / "python3" / "dist-packages"
    ros_python_root.mkdir(parents=True)
    (ros_root / "setup.bash").write_text("# fake noetic setup\n", encoding="utf-8")
    for package in EXPECTED_ROS_PACKAGES:
        (ros_python_root / package).mkdir()

    command_root = host / "commands"
    commands = {
        "uv": str(_write_executable(command_root / "uv")),
        "catkin_make": str(_write_executable(command_root / "catkin_make")),
        "roscore": str(_write_executable(command_root / "roscore")),
    }
    cuda_root = host / "usr" / "local" / "cuda-11.8"
    nvcc = _write_executable(cuda_root / "bin" / "nvcc")
    gcc = _write_executable(host / "usr" / "bin" / "gcc")
    gxx = _write_executable(host / "usr" / "bin" / "g++")
    environment = {
        "ROS_DISTRO": "noetic",
        "CUDA_HOME": str(cuda_root),
    }
    dependencies = EnvironmentPreflightDependencies(
        platform_system=lambda: "Linux",
        platform_machine=lambda: "x86_64",
        read_os_release=lambda: {"ID": "ubuntu", "VERSION_ID": "20.04"},
        getenv=environment.get,
        which=commands.get,
        gpu_available=lambda: gpu_available,
        ros_root=ros_root,
        ros_python_root=ros_python_root,
        cuda_root=cuda_root,
        nvcc_path=nvcc,
        gcc_path=gcc,
        gxx_path=gxx,
    )
    return dependencies, _RecordingPreflightRunner(), environment


def _assert_probe_call_contract(runner: _RecordingPreflightRunner) -> None:
    """모든 host command가 shell 없는 bounded read probe인지 고정한다."""
    assert runner.calls
    for argv, kwargs in runner.calls:
        assert isinstance(argv, tuple)
        assert kwargs == {
            "check": True,
            "shell": False,
            "capture_output": True,
            "text": True,
            "timeout": 10.0,
        }


def test_preflight_public_result_is_frozen_and_read_only(tmp_path: Path):
    """preflight가 exact 공개 결과를 반환하고 project를 변경하지 않는다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path, gpu_available=False)

    result = preflight_environment(
        root, "interfuser", runner=runner, dependencies=dependencies
    )

    assert tuple(field.name for field in dataclasses.fields(EnvironmentPreflight)) == (
        "root",
        "package_root",
        "manifest",
        "definition",
        "selected_profile",
        "paths",
        "uv_path",
        "warnings",
    )
    assert result.root == root
    assert result.package_root == root / "src" / "Validator_pkg" / "validator_inference"
    assert result.manifest.manifest_path == result.package_root / "config" / "model_environments.json"
    assert result.definition.model_id == "interfuser"
    assert result.selected_profile.model_id == "interfuser"
    assert result.paths == environment_paths(root, "interfuser")
    assert result.uv_path == Path(dependencies.which("uv"))
    assert result.warnings == (_PREFLIGHT_WARNING.format(model_id="interfuser"),)
    assert not (root / ".venvs").exists()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.warnings = ()  # type: ignore[misc]
    _assert_probe_call_contract(runner)


def test_preflight_public_type_hints_resolve_without_eager_profile_import():
    """격리 runtime에서도 공개 result annotation을 runtime에 해석할 수 있다."""
    hints = typing.get_type_hints(EnvironmentPreflight)

    assert tuple(hints) == (
        "root",
        "package_root",
        "manifest",
        "definition",
        "selected_profile",
        "paths",
        "uv_path",
        "warnings",
    )
    assert all(value is not None for value in hints.values())


def test_default_gpu_probe_requires_accessible_driver_and_device_nodes(
    tmp_path: Path,
):
    """proc metadata만 남은 container를 usable CUDA GPU로 오인하지 않는다."""
    driver_root = tmp_path / "proc" / "driver" / "nvidia" / "gpus"
    (driver_root / "0000:01:00.0").mkdir(parents=True)
    device_root = tmp_path / "dev"
    device_root.mkdir()
    control = device_root / "nvidiactl"
    gpu = device_root / "nvidia0"
    control.write_text("control\n", encoding="utf-8")

    assert not environment_setup._default_gpu_available(driver_root, device_root)

    gpu.write_text("gpu\n", encoding="utf-8")
    assert environment_setup._default_gpu_available(driver_root, device_root)

    gpu.chmod(0)
    assert not environment_setup._default_gpu_available(driver_root, device_root)


def test_preflight_rejects_command_not_executable_by_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """mode bit이 있어도 현재 process가 실행할 수 없는 command를 거부한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    uv_path = Path(dependencies.which("uv"))
    real_access = os.access

    def deny_uv(path: str | Path, mode: int) -> bool:
        if Path(path) == uv_path and mode == os.X_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(environment_setup.os, "access", deny_uv)

    with pytest.raises(EnvironmentSetupError, match="executable regular file"):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert runner.calls == []


@pytest.mark.parametrize(
    ("model_id", "gpu_available", "expected_warnings", "expected_commands"),
    (
        ("interfuser", False, 1, ("uv",)),
        ("tfv6", True, 0, ("uv",)),
        ("diffusiondrive_v2", False, 1, ("uv",)),
        ("sparsedrive_v2", True, 0, ("uv", "nvcc", "gcc", "g++")),
    ),
)
def test_preflight_applies_each_model_cuda_policy(
    tmp_path: Path,
    model_id: str,
    gpu_available: bool,
    expected_warnings: int,
    expected_commands: tuple[str, ...],
):
    """네 모델의 required/recommended/extension 정책을 그대로 집행한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(
        tmp_path, gpu_available=gpu_available
    )

    result = preflight_environment(
        root, model_id, runner=runner, dependencies=dependencies
    )

    assert len(result.warnings) == expected_warnings
    assert tuple(Path(call[0][0]).name for call in runner.calls) == expected_commands
    assert not (root / ".venvs").exists()


@pytest.mark.parametrize("model_id", ("tfv6", "sparsedrive_v2"))
def test_preflight_blocks_required_cuda_policies_before_cuda_tool_commands(
    tmp_path: Path, model_id: str
):
    """required 계열은 GPU 부재를 nvcc/build/download 전에 차단한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path, gpu_available=False)

    with pytest.raises(EnvironmentSetupError, match="CUDA GPU is required") as caught:
        preflight_environment(root, model_id, runner=runner, dependencies=dependencies)

    assert caught.value.action == "host_preflight"
    assert tuple(Path(call[0][0]).name for call in runner.calls) == ("uv",)
    assert not (root / ".venvs").exists()


def test_preflight_rejects_unknown_model_without_running_commands(tmp_path: Path):
    """unknown model ID는 loader와 host command보다 먼저 거부한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)

    with pytest.raises(EnvironmentSetupError, match="unsupported model_id") as caught:
        preflight_environment(root, "unknown", runner=runner, dependencies=dependencies)

    assert caught.value.action == "host_preflight"
    assert runner.calls == []


def test_preflight_rejects_unknown_injected_cuda_policy(tmp_path: Path):
    """검증 seam이 반환한 알 수 없는 CUDA 정책도 fail closed 한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    original_select = dependencies.select_definition

    def select_unknown(manifest: Any, model_id: str):
        return dataclasses.replace(
            original_select(manifest, model_id), cuda_policy="future_policy"
        )

    dependencies = dataclasses.replace(
        dependencies, select_definition=select_unknown
    )
    with pytest.raises(EnvironmentSetupError, match="unsupported cuda_policy"):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert runner.calls == []


def test_preflight_rejects_profile_manifest_source_mismatch_before_host_probe(
    tmp_path: Path,
):
    """profile source pin과 environment pin 불일치는 host probe 전에 차단한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    original_load_profile = dependencies.load_profile

    def load_mismatched(path: Path, model_id: str):
        selected = original_load_profile(path, model_id)
        selected.data["source"]["commit"] = "0" * 40
        return selected

    dependencies = dataclasses.replace(dependencies, load_profile=load_mismatched)
    with pytest.raises(EnvironmentSetupError, match="source commit does not match"):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert runner.calls == []


@pytest.mark.parametrize(
    ("dependency_changes", "message"),
    (
        ({"platform_system": lambda: "Darwin"}, "Linux"),
        ({"read_os_release": lambda: {"ID": "debian", "VERSION_ID": "12"}}, "Ubuntu 20.04"),
        ({"read_os_release": lambda: {"ID": "ubuntu", "VERSION_ID": "22.04"}}, "Ubuntu 20.04"),
        ({"platform_machine": lambda: "aarch64"}, "x86_64"),
    ),
)
def test_preflight_rejects_unsupported_host_before_uv(
    tmp_path: Path, dependency_changes: dict[str, Any], message: str
):
    """Linux Ubuntu 20.04 x86_64 외 host는 uv 실행 전에 거부한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    dependencies = dataclasses.replace(dependencies, **dependency_changes)

    with pytest.raises(EnvironmentSetupError, match=message) as caught:
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert caught.value.action == "host_preflight"
    assert runner.calls == []


@pytest.mark.parametrize(
    ("uv_output", "message"),
    (
        ("uv 0.11.13\n", "uv 0.11.14"),
        ("uv 0.11.14rc1\n", "uv 0.11.14"),
        ("unexpected\n", "uv 0.11.14"),
    ),
)
def test_preflight_requires_exact_uv_version(
    tmp_path: Path, uv_output: str, message: str
):
    """uv version probe는 0.11.14 token만 허용한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    runner.uv_output = uv_output

    with pytest.raises(EnvironmentSetupError, match=message):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    _assert_probe_call_contract(runner)
    assert len(runner.calls) == 1


def test_preflight_requires_existing_uv_executable_without_running_runner(
    tmp_path: Path,
):
    """lookup된 uv가 없거나 executable이 아니면 subprocess를 호출하지 않는다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    dependencies = dataclasses.replace(
        dependencies, which=lambda command: None if command == "uv" else dependencies.which(command)
    )

    with pytest.raises(EnvironmentSetupError, match="uv executable"):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert runner.calls == []


@pytest.mark.parametrize(
    ("break_host", "message"),
    (
        ("ros_distro", "ROS_DISTRO=noetic"),
        ("setup", "setup.bash"),
        ("package", "ROS Python package"),
        ("catkin_make", "catkin_make"),
        ("roscore", "roscore"),
    ),
)
def test_preflight_requires_noetic_setup_tools_and_exact_package_dirs(
    tmp_path: Path, break_host: str, message: str
):
    """Noetic environment, tools와 14개 package directory를 모두 검증한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, environment = _preflight_host(tmp_path)
    if break_host == "ros_distro":
        environment["ROS_DISTRO"] = "melodic"
    elif break_host == "setup":
        (dependencies.ros_root / "setup.bash").unlink()
    elif break_host == "package":
        (dependencies.ros_python_root / EXPECTED_ROS_PACKAGES[-1]).rmdir()
    else:
        old_which = dependencies.which
        dependencies = dataclasses.replace(
            dependencies,
            which=lambda command, missing=break_host: (
                None if command == missing else old_which(command)
            ),
        )

    with pytest.raises(EnvironmentSetupError, match=message):
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert tuple(Path(call[0][0]).name for call in runner.calls) == ("uv",)


def test_sparse_calibration_blocks_before_every_subprocess_probe(tmp_path: Path):
    """불완전 Sparse calibration은 uv/nvcc/asset 작업 전 ProfileError를 정규화한다."""
    root = _preflight_project(tmp_path, complete_sparse=False)
    dependencies, runner, _ = _preflight_host(tmp_path)

    with pytest.raises(EnvironmentSetupError, match="camera front requires nonempty") as caught:
        preflight_environment(
            root, "sparsedrive_v2", runner=runner, dependencies=dependencies
        )
    assert caught.value.action == "host_preflight"
    assert runner.calls == []
    assert not (root / ".venvs").exists()


@pytest.mark.parametrize(
    ("blocker", "message", "expected_commands"),
    (
        ("cuda_home", "CUDA_HOME", ("uv",)),
        ("cuda_root", "CUDA Toolkit", ("uv",)),
        ("nvcc", "nvcc", ("uv",)),
        ("nvcc_version", "release 11.8", ("uv", "nvcc")),
        ("gcc", "/usr/bin/gcc", ("uv", "nvcc")),
        ("compiler_version", "compatible", ("uv", "nvcc", "gcc", "g++")),
    ),
)
def test_sparse_host_blockers_run_in_read_only_order(
    tmp_path: Path,
    blocker: str,
    message: str,
    expected_commands: tuple[str, ...],
):
    """Sparse Toolkit/nvcc/compiler blocker를 build나 asset 작업 전에 순서대로 찾는다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, environment = _preflight_host(tmp_path)
    if blocker == "cuda_home":
        environment["CUDA_HOME"] = "/wrong/cuda"
    elif blocker == "cuda_root":
        shutil.rmtree(dependencies.cuda_root)
    elif blocker == "nvcc":
        dependencies.nvcc_path.unlink()
    elif blocker == "nvcc_version":
        runner.nvcc_output = "Cuda compilation tools, release 12.1, V12.1.0\n"
    elif blocker == "gcc":
        dependencies.gcc_path.unlink()
    else:
        runner.gxx_output = "12.1.0\n"

    with pytest.raises(EnvironmentSetupError, match=message):
        preflight_environment(
            root, "sparsedrive_v2", runner=runner, dependencies=dependencies
        )
    assert tuple(Path(call[0][0]).name for call in runner.calls) == expected_commands
    assert not (root / ".venvs").exists()


@pytest.mark.parametrize(
    "failure",
    (
        OSError("exec denied"),
        subprocess.CalledProcessError(1, ["uv", "--version"]),
        subprocess.TimeoutExpired(["uv", "--version"], 10.0),
    ),
)
def test_preflight_normalizes_subprocess_failures(
    tmp_path: Path, failure: BaseException
):
    """runner의 OS/exit/timeout 오류를 deterministic host_preflight로 변환한다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    runner.failure = failure

    with pytest.raises(EnvironmentSetupError, match="uv version probe failed") as caught:
        preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)
    assert caught.value.action == "host_preflight"


def test_preflight_never_invokes_mutating_setup_commands_or_creates_paths(
    tmp_path: Path,
):
    """성공 preflight도 downloader, catkin, uv venv/sync, asset mutator를 호출하지 않는다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path, gpu_available=False)
    before = {
        path.relative_to(root): (path.stat().st_mode, path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
    }

    preflight_environment(root, "interfuser", runner=runner, dependencies=dependencies)

    after = {
        path.relative_to(root): (path.stat().st_mode, path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
    }
    assert before == after
    assert [call[0][1:] for call in runner.calls] == [("--version",)]
    assert not (root / ".venvs").exists()


def test_real_host_interfuser_preflight_is_nonmutating_when_prerequisites_exist():
    """실제 Ubuntu/Noetic/uv tree에서도 InterFuser 점검이 project를 바꾸지 않는다."""
    dependencies = EnvironmentPreflightDependencies()
    release = dependencies.read_os_release()
    uv_path = shutil.which("uv")
    ros_commands = {
        "catkin_make": dependencies.ros_root / "bin" / "catkin_make",
        "roscore": dependencies.ros_root / "bin" / "roscore",
    }
    required_paths = (
        dependencies.ros_root / "setup.bash",
        dependencies.ros_python_root,
        *(dependencies.ros_python_root / package for package in EXPECTED_ROS_PACKAGES),
        *ros_commands.values(),
    )
    if (
        dependencies.platform_system() != "Linux"
        or dependencies.platform_machine() != "x86_64"
        or release.get("ID") != "ubuntu"
        or release.get("VERSION_ID") != "20.04"
        or uv_path is None
        or any(not path.exists() for path in required_paths)
    ):
        pytest.skip("real Ubuntu 20.04, ROS Noetic, and uv prerequisites unavailable")

    command_paths = {
        "uv": uv_path,
        **{name: str(path) for name, path in ros_commands.items()},
    }
    dependencies = dataclasses.replace(
        dependencies,
        getenv=lambda key: "noetic" if key == "ROS_DISTRO" else os.environ.get(key),
        which=command_paths.get,
        gpu_available=lambda: False,
    )
    environment_root = _REPOSITORY_ROOT / ".venvs"
    before = environment_root.lstat() if environment_root.exists() else None

    result = preflight_environment(
        _REPOSITORY_ROOT,
        "interfuser",
        runner=subprocess.run,
        dependencies=dependencies,
    )

    after = environment_root.lstat() if environment_root.exists() else None
    assert result.definition.model_id == "interfuser"
    assert result.warnings == (_PREFLIGHT_WARNING.format(model_id="interfuser"),)
    assert before == after


_INTERFUSER_CHECKPOINT_PATH = Path(
    "models/interfuser/official_carla/interfuser.pth.tar"
)
_INTERFUSER_BACKBONE_PATH = Path(
    "models/shared/torch-cache/hub/checkpoints/resnet50d_ra2-464e36ba.pth"
)
_INTERFUSER_CHECKPOINT_SHA256 = (
    "39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2"
)
_INTERFUSER_BACKBONE_SHA256 = (
    "464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d"
)


def _bytes_sha256(payload: bytes) -> str:
    """작은 asset fixture bytes의 SHA-256을 계산한다."""
    return hashlib.sha256(payload).hexdigest()


def _interfuser_runtime_assets(
    checkpoint_sha256: str,
    backbone_sha256: str,
) -> list[dict[str, Any]]:
    """InterFuser의 두 runtime asset 선언을 fixture hash로 만든다."""
    return [
        {
            "role": "checkpoint",
            "path": _INTERFUSER_CHECKPOINT_PATH.as_posix(),
            "expected_filename": "interfuser.pth.tar",
            "sha256": checkpoint_sha256,
            "provenance": {
                "classification": "third_party_compatibility_artifact",
                "repository": "https://github.com/MingfeiCheng/Drivora",
                "commit": "25d56d2300f422d502c6549695b9e49efd5e5e00",
                "google_drive_id": "1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6",
            },
        },
        {
            "role": "image_backbone",
            "path": _INTERFUSER_BACKBONE_PATH.as_posix(),
            "source_url": (
                "https://github.com/rwightman/pytorch-image-models/releases/"
                "download/v0.1-weights/resnet50d_ra2-464e36ba.pth"
            ),
            "sha256": backbone_sha256,
        },
    ]


def _asset_preflight(
    tmp_path: Path,
    runtime_assets: list[dict[str, Any]],
    *,
    model_id: str = "interfuser",
) -> EnvironmentPreflight:
    """host 검증을 통과한 preflight에 격리 runtime asset 선언을 넣는다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    preflight = preflight_environment(
        root, model_id, runner=runner, dependencies=dependencies
    )
    profile = dict(preflight.selected_profile.data)
    profile["runtime_assets"] = runtime_assets
    selected_profile = dataclasses.replace(preflight.selected_profile, data=profile)
    return dataclasses.replace(preflight, selected_profile=selected_profile)


def _write_asset(root: Path, relative: Path, payload: bytes) -> Path:
    """격리 project 아래에 작은 regular asset을 만든다."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "",
        "/absolute.bin",
        "../escape.bin",
        "models/../escape.bin",
        "models/./escape.bin",
        "models//escape.bin",
        "models/escape.bin/",
        "models\\escape.bin",
        "models/bad\nname.bin",
    ),
)
def test_fingerprint_rejects_noncanonical_repository_relative_asset_paths(
    tmp_path: Path,
    unsafe_path: str,
):
    """asset 선언은 root 탈출, alias, backslash와 control 문자를 허용하지 않는다."""
    preflight = _asset_preflight(
        tmp_path,
        [{"role": "checkpoint", "path": unsafe_path, "sha256": None}],
        model_id="tfv6",
    )

    with pytest.raises(EnvironmentSetupError, match="repository-relative") as caught:
        fingerprint_runtime_assets(preflight)

    assert caught.value.action == "assets"
    assert not (preflight.root / "models").exists()


@pytest.mark.parametrize("kind", ("parent_symlink", "final_symlink", "fifo"))
def test_fingerprint_rejects_symlinked_and_special_asset_files(
    tmp_path: Path,
    kind: str,
):
    """asset parent/final symlink과 blocking special file을 no-follow로 거부한다."""
    declaration = {
        "role": "checkpoint",
        "path": "models/tfv6/checkpoint.bin",
        "sha256": None,
    }
    preflight = _asset_preflight(tmp_path, [declaration], model_id="tfv6")
    asset_path = preflight.root / declaration["path"]
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target.bin"
    target.write_bytes(b"outside")
    if kind == "parent_symlink":
        (preflight.root / "models").symlink_to(outside, target_is_directory=True)
    else:
        asset_path.parent.mkdir(parents=True)
        if kind == "final_symlink":
            asset_path.symlink_to(target)
        else:
            os.mkfifo(asset_path)

    with pytest.raises(EnvironmentSetupError, match="symlink|regular file") as caught:
        fingerprint_runtime_assets(preflight)

    assert caught.value.action == "assets"


def test_prepare_missing_interfuser_checkpoint_reports_full_provenance_first(
    tmp_path: Path,
):
    """checkpoint 누락은 cache 검사나 directory 생성 전에 완전한 복구 정보를 준다."""
    root = _preflight_project(tmp_path)
    dependencies, runner, _ = _preflight_host(tmp_path)
    preflight = preflight_environment(
        root, "interfuser", runner=runner, dependencies=dependencies
    )
    expected_path = root / _INTERFUSER_CHECKPOINT_PATH

    with pytest.raises(EnvironmentSetupError) as caught:
        prepare_local_runtime_assets(
            preflight,
            interfuser_user_cache=Path("relative-cache-must-not-be-read"),
        )

    assert caught.value.action == "assets"
    message = str(caught.value)
    assert str(expected_path) in message
    assert _INTERFUSER_CHECKPOINT_SHA256 in message
    assert "third_party_compatibility_artifact" in message
    assert "https://github.com/MingfeiCheng/Drivora" in message
    assert "25d56d2300f422d502c6549695b9e49efd5e5e00" in message
    assert "1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6" in message
    assert not (root / "models").exists()


@pytest.mark.parametrize("kind", ("mismatch", "symlink"))
def test_prepare_never_overwrites_invalid_interfuser_checkpoint(
    tmp_path: Path,
    kind: str,
):
    """기존 checkpoint hash 불일치와 symlink는 원본을 바꾸지 않고 실패한다."""
    checkpoint_payload = b"expected checkpoint"
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), "b" * 64
    )
    preflight = _asset_preflight(tmp_path, assets)
    checkpoint = preflight.root / _INTERFUSER_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    outside = tmp_path / "outside-checkpoint.bin"
    outside.write_bytes(b"outside checkpoint")
    if kind == "mismatch":
        checkpoint.write_bytes(b"wrong checkpoint")
        before = checkpoint.read_bytes()
    else:
        checkpoint.symlink_to(outside)
        before = outside.read_bytes()

    with pytest.raises(EnvironmentSetupError, match="SHA-256|symlink") as caught:
        prepare_local_runtime_assets(
            preflight, interfuser_user_cache=tmp_path / "unused-cache.pth"
        )

    assert caught.value.action == "assets"
    if kind == "mismatch":
        assert checkpoint.read_bytes() == before
    else:
        assert checkpoint.is_symlink()
        assert outside.read_bytes() == before


@pytest.mark.parametrize("exact", (True, False))
def test_prepare_never_rewrites_existing_project_backbone(
    tmp_path: Path,
    exact: bool,
):
    """기존 project backbone은 exact면 재사용하고 mismatch면 덮어쓰지 않는다."""
    checkpoint_payload = b"checkpoint"
    backbone_payload = b"backbone"
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)
    backbone = _write_asset(
        preflight.root,
        _INTERFUSER_BACKBONE_PATH,
        backbone_payload if exact else b"existing mismatch",
    )
    before = backbone.lstat()
    if exact:
        outside = tmp_path / "outside-cache.pth"
        outside.write_bytes(b"must not be read")
        cache = tmp_path / "cache-link.pth"
        cache.symlink_to(outside)
    else:
        cache = _write_asset(
            tmp_path, Path("user-cache/backbone.pth"), backbone_payload
        )

    if exact:
        prepare_local_runtime_assets(preflight, interfuser_user_cache=cache)
    else:
        with pytest.raises(EnvironmentSetupError, match="SHA-256"):
            prepare_local_runtime_assets(preflight, interfuser_user_cache=cache)

    after = backbone.lstat()
    assert backbone.read_bytes() == (
        backbone_payload if exact else b"existing mismatch"
    )
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_prepare_leaves_project_cache_absent_when_user_cache_is_absent(
    tmp_path: Path,
):
    """user cache가 없으면 official source_url fetcher를 위해 project를 건드리지 않는다."""
    checkpoint_payload = b"checkpoint"
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), "c" * 64
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)

    prepare_local_runtime_assets(
        preflight, interfuser_user_cache=tmp_path / "missing-user-cache.pth"
    )

    assert not (preflight.root / _INTERFUSER_BACKBONE_PATH).exists()
    assert not (preflight.root / "models" / "shared").exists()


def test_prepare_copies_exact_user_cache_with_atomic_fsync_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """exact cache를 fsync한 뒤 no-clobber link, temp unlink와 parent fsync한다."""
    checkpoint_payload = b"checkpoint"
    backbone_payload = b"cached backbone" * 4096
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)
    user_cache = _write_asset(tmp_path, Path("user/cache/backbone.pth"), backbone_payload)
    events: list[tuple[Any, ...]] = []
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = os.unlink

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append(("fsync", "directory" if stat.S_ISDIR(mode) else "file"))
        real_fsync(descriptor)

    def recording_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        events.append(
            (
                "link",
                source,
                destination,
                src_dir_fd == dst_dir_fd,
                follow_symlinks,
            )
        )
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def recording_unlink(path: str, *, dir_fd: int | None = None) -> None:
        events.append(("unlink", path))
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(environment_setup.os, "fsync", recording_fsync)
    monkeypatch.setattr(environment_setup.os, "link", recording_link)
    monkeypatch.setattr(environment_setup.os, "unlink", recording_unlink)

    prepare_local_runtime_assets(preflight, interfuser_user_cache=user_cache)

    destination = preflight.root / _INTERFUSER_BACKBONE_PATH
    assert destination.read_bytes() == backbone_payload
    assert events[0] == ("fsync", "file")
    assert events[1][0] == "link"
    assert events[1][2:] == (destination.name, True, False)
    assert events[2] == ("unlink", events[1][1])
    assert events[3] == ("fsync", "directory")
    assert not any(path.name.startswith(".asset-") for path in destination.parent.iterdir())


@pytest.mark.parametrize("racer_kind", ("regular", "symlink"))
def test_prepare_atomic_install_never_clobbers_destination_created_at_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    racer_kind: str,
):
    """publication 직전 생긴 regular file이나 symlink를 atomic no-clobber로 보존한다."""
    checkpoint_payload = b"checkpoint"
    backbone_payload = b"cached backbone" * 4096
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)
    user_cache = _write_asset(tmp_path, Path("user/cache/backbone.pth"), backbone_payload)
    destination = preflight.root / _INTERFUSER_BACKBONE_PATH
    symlink_target = tmp_path / "race-target.pth"
    symlink_target.write_bytes(b"outside")
    raced: dict[str, int] = {}
    real_link = os.link
    real_replace = os.replace

    def create_racer(name: str, parent_descriptor: int) -> None:
        if racer_kind == "regular":
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                os.write(descriptor, b"racer content")
            finally:
                os.close(descriptor)
        else:
            os.symlink(str(symlink_target), name, dir_fd=parent_descriptor)
        raced["inode"] = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        ).st_ino

    def racing_link(
        source: str,
        destination_name: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert src_dir_fd is not None and dst_dir_fd is not None
        create_racer(destination_name, dst_dir_fd)
        real_link(
            source,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def racing_replace(
        source: str,
        destination_name: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        assert src_dir_fd is not None and dst_dir_fd is not None
        create_racer(destination_name, dst_dir_fd)
        real_replace(
            source,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(environment_setup.os, "link", racing_link)
    monkeypatch.setattr(environment_setup.os, "replace", racing_replace)

    with pytest.raises(EnvironmentSetupError, match="already exists|overwrite") as caught:
        prepare_local_runtime_assets(preflight, interfuser_user_cache=user_cache)

    assert caught.value.action == "assets"
    assert destination.lstat().st_ino == raced["inode"]
    if racer_kind == "regular":
        assert destination.read_bytes() == b"racer content"
    else:
        assert destination.is_symlink()
        assert Path(os.readlink(destination)) == symlink_target
    assert not any(path.name.startswith(".asset-") for path in destination.parent.iterdir())


@pytest.mark.parametrize("kind", ("mismatch", "final_symlink", "parent_symlink"))
def test_prepare_rejects_invalid_user_cache_without_installing_backbone(
    tmp_path: Path,
    kind: str,
):
    """cache hash 불일치와 parent/final symlink는 project asset으로 복사하지 않는다."""
    checkpoint_payload = b"checkpoint"
    backbone_payload = b"backbone"
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "backbone.pth"
    target.write_bytes(backbone_payload if kind != "mismatch" else b"wrong")
    if kind == "parent_symlink":
        cache_parent = tmp_path / "user-cache"
        cache_parent.symlink_to(outside, target_is_directory=True)
        user_cache = cache_parent / target.name
    elif kind == "final_symlink":
        user_cache = tmp_path / "cache-link.pth"
        user_cache.symlink_to(target)
    else:
        user_cache = target

    with pytest.raises(EnvironmentSetupError, match="SHA-256|symlink") as caught:
        prepare_local_runtime_assets(preflight, interfuser_user_cache=user_cache)

    assert caught.value.action == "assets"
    assert not (preflight.root / _INTERFUSER_BACKBONE_PATH).exists()


def test_prepare_copy_failure_cleans_temp_and_leaves_destination_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """stream copy 오류는 sibling temp를 제거하고 destination을 만들지 않는다."""
    checkpoint_payload = b"checkpoint"
    backbone_payload = b"backbone" * 4096
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload)
    user_cache = _write_asset(tmp_path, Path("user-cache/backbone.pth"), backbone_payload)

    def fail_write(descriptor: int, block: bytes) -> int:
        raise OSError("injected copy failure")

    monkeypatch.setattr(environment_setup.os, "write", fail_write)

    with pytest.raises(EnvironmentSetupError, match="copy") as caught:
        prepare_local_runtime_assets(preflight, interfuser_user_cache=user_cache)

    destination = preflight.root / _INTERFUSER_BACKBONE_PATH
    assert caught.value.action == "assets"
    assert not destination.exists()
    assert destination.parent.is_dir()
    assert list(destination.parent.iterdir()) == []


def test_fingerprint_hashes_every_asset_in_declaration_order_with_lstat_metadata(
    tmp_path: Path,
):
    """모든 asset을 선언 순서대로 deep hash하고 repo-relative metadata를 반환한다."""
    checkpoint_payload = b"checkpoint fingerprint"
    backbone_payload = b"backbone fingerprint"
    assets = _interfuser_runtime_assets(
        _bytes_sha256(checkpoint_payload), _bytes_sha256(backbone_payload)
    )
    preflight = _asset_preflight(tmp_path, assets)
    checkpoint = _write_asset(
        preflight.root, _INTERFUSER_CHECKPOINT_PATH, checkpoint_payload
    )
    backbone = _write_asset(
        preflight.root, _INTERFUSER_BACKBONE_PATH, backbone_payload
    )

    fingerprints = fingerprint_runtime_assets(preflight)

    assert fingerprints == tuple(
        AssetFingerprint(
            path=relative,
            sha256=_bytes_sha256(payload),
            size=len(payload),
            mtime_ns=absolute.lstat().st_mtime_ns,
        )
        for relative, absolute, payload in (
            (_INTERFUSER_CHECKPOINT_PATH, checkpoint, checkpoint_payload),
            (_INTERFUSER_BACKBONE_PATH, backbone, backbone_payload),
        )
    )


def test_fingerprint_rejects_nonnull_expected_sha256_mismatch(tmp_path: Path):
    """선언된 expected SHA-256과 본문 hash 불일치는 fingerprint로 기록하지 않는다."""
    declaration = {
        "role": "checkpoint",
        "path": "models/tfv6/checkpoint.bin",
        "sha256": "0" * 64,
    }
    preflight = _asset_preflight(tmp_path, [declaration], model_id="tfv6")
    _write_asset(preflight.root, Path(declaration["path"]), b"not zero")

    with pytest.raises(EnvironmentSetupError, match="expected.*0{64}") as caught:
        fingerprint_runtime_assets(preflight)

    assert caught.value.action == "assets"


def test_fingerprint_streams_assets_larger_than_overlay_limit(tmp_path: Path):
    """runtime asset deep hash에는 ROS overlay의 64MiB 상한을 재사용하지 않는다."""
    relative = Path("models/tfv6/large-checkpoint.bin")
    declaration = {"role": "checkpoint", "path": relative.as_posix(), "sha256": None}
    preflight = _asset_preflight(tmp_path, [declaration], model_id="tfv6")
    asset = preflight.root / relative
    asset.parent.mkdir(parents=True)
    with asset.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"asset-tail")

    fingerprints = fingerprint_runtime_assets(preflight)

    assert len(fingerprints) == 1
    assert fingerprints[0].path == relative
    assert fingerprints[0].size == 64 * 1024 * 1024 + len(b"asset-tail")
    assert fingerprints[0].sha256 == sha256_file(asset)


_REAL_INTERFUSER_CHECKPOINT = _REPOSITORY_ROOT / _INTERFUSER_CHECKPOINT_PATH
_REAL_INTERFUSER_USER_CACHE = (
    Path.home()
    / ".cache"
    / "torch"
    / "hub"
    / "checkpoints"
    / _INTERFUSER_BACKBONE_PATH.name
)


@pytest.mark.skipif(
    not all(
        path.is_file() and not path.is_symlink()
        for path in (_REAL_INTERFUSER_CHECKPOINT, _REAL_INTERFUSER_USER_CACHE)
    ),
    reason="local InterFuser checkpoint/user cache are unavailable",
)
def test_real_interfuser_checkpoint_and_user_cache_integration_is_source_read_only(
    tmp_path: Path,
):
    """실제 두 대용량 source는 읽기만 하고 cache copy는 temp project에 제한한다."""
    dependencies, runner, _ = _preflight_host(tmp_path)
    checkpoint_preflight = preflight_environment(
        _REPOSITORY_ROOT,
        "tfv6",
        runner=runner,
        dependencies=dependencies,
    )
    checkpoint_profile = dict(checkpoint_preflight.selected_profile.data)
    checkpoint_profile["runtime_assets"] = [
        {
            "role": "checkpoint",
            "path": _INTERFUSER_CHECKPOINT_PATH.as_posix(),
            "sha256": _INTERFUSER_CHECKPOINT_SHA256,
        }
    ]
    checkpoint_preflight = dataclasses.replace(
        checkpoint_preflight,
        selected_profile=dataclasses.replace(
            checkpoint_preflight.selected_profile, data=checkpoint_profile
        ),
    )
    before = (
        _REAL_INTERFUSER_CHECKPOINT.lstat(),
        _REAL_INTERFUSER_USER_CACHE.lstat(),
    )
    checkpoint_fingerprint = fingerprint_runtime_assets(checkpoint_preflight)

    fixture_checkpoint = b"fixture checkpoint"
    preflight = _asset_preflight(
        tmp_path / "cache-copy",
        _interfuser_runtime_assets(
            _bytes_sha256(fixture_checkpoint), _INTERFUSER_BACKBONE_SHA256
        ),
    )
    _write_asset(preflight.root, _INTERFUSER_CHECKPOINT_PATH, fixture_checkpoint)
    prepare_local_runtime_assets(
        preflight, interfuser_user_cache=_REAL_INTERFUSER_USER_CACHE
    )
    fingerprints = fingerprint_runtime_assets(preflight)
    after = (
        _REAL_INTERFUSER_CHECKPOINT.lstat(),
        _REAL_INTERFUSER_USER_CACHE.lstat(),
    )

    assert checkpoint_fingerprint[0].sha256 == _INTERFUSER_CHECKPOINT_SHA256
    assert checkpoint_fingerprint[0].size == 636107072
    assert tuple(fingerprint.path for fingerprint in fingerprints) == (
        _INTERFUSER_CHECKPOINT_PATH,
        _INTERFUSER_BACKBONE_PATH,
    )
    assert fingerprints[1].sha256 == _INTERFUSER_BACKBONE_SHA256
    assert fingerprints[1].size == 102567109
    assert tuple(
        (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        for metadata in after
    ) == tuple(
        (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        for metadata in before
    )


class _SetupRunner:
    """setup subprocess를 실행하지 않고 파일 효과, stdout과 phase를 기록한다."""

    def __init__(self, root: Path, model_id: str, python_version: str) -> None:
        self.root = root
        self.model_id = model_id
        self.python_version = python_version
        self.calls: list[tuple[tuple[str, ...], dict[str, Any], str]] = []
        self.events: list[str] = []
        self.freeze_output = "z-package==2\r\na-package==1\r\n\r\n"
        self.git_status_output = ""
        self.source_commit = ""
        self.fail_phase: str | None = None
        self.candidate_cuda_available = True
        self.post_sparse_git_status_output = ""
        self.write_sparse_extensions = True
        self.required_import_origins_override: list[str] | None = None
        self.phase_hooks: dict[
            str, Callable[[tuple[str, ...], dict[str, Any]], None]
        ] = {}

    @staticmethod
    def _phase(command: tuple[str, ...]) -> str:
        executable = Path(command[0]).name
        if executable == "uv" and command[1:] == ("--version",):
            return "host_uv"
        if executable == "nvcc":
            return "host_nvcc"
        if executable == "gcc":
            return "host_gcc"
        if executable == "g++":
            return "host_gxx"
        if executable == "catkin_make":
            return "catkin_build"
        if command[0] == "/usr/bin/python3":
            return "catkin_messages"
        if executable == "git" and command[-2:] == ("rev-parse", "HEAD"):
            return "source_head"
        if executable == "git" and "status" in command:
            return "source_status"
        if executable == "uv" and len(command) > 1 and command[1] == "venv":
            return "uv_venv"
        if executable == "uv" and command[1:3] == ("pip", "sync"):
            return "uv_sync"
        if executable == "uv" and command[1:3] == ("pip", "freeze"):
            return "freeze"
        if command[1:] == ("setup.py", "build_ext", "--inplace"):
            return "sparse_build"
        try:
            code = command[command.index("-c") + 1]
        except (ValueError, IndexError):
            code = ""
        if "platform.python_version" in code:
            return "python_version"
        if "deformable_aggregation_with_depth_ext" in code and "origins =" in code:
            return "sparse_imports"
        if "torch.cuda.is_available" in code:
            return "cuda_runtime"
        if "build_selected_model" in code:
            return "model_load"
        if "validator_msgs.msg" in code and "rospy" in code:
            return "target_ros_import"
        if "import_module" in code:
            return "required_imports"
        return "unexpected"

    def __call__(self, argv: list[str] | tuple[str, ...], **kwargs: Any):
        assert isinstance(argv, (list, tuple))
        command = tuple(str(value) for value in argv)
        phase = self._phase(command)
        self.calls.append((command, dict(kwargs), phase))
        self.events.append(phase)
        if self.fail_phase == phase:
            raise subprocess.CalledProcessError(7, list(command))

        if phase == "host_uv":
            stdout = "uv 0.11.14 (x86_64-unknown-linux-gnu)\n"
        elif phase == "host_nvcc":
            stdout = "Cuda compilation tools, release 11.8, V11.8.89\n"
        elif phase in {"host_gcc", "host_gxx"}:
            stdout = "9.4.0\n"
        elif phase == "catkin_build":
            catkin = self.root / "devel" / "lib" / "python3" / "dist-packages"
            (catkin / "validator_msgs" / "msg").mkdir(parents=True, exist_ok=True)
            (catkin / "validator_msgs" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (catkin / "validator_msgs" / "msg" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            stdout = ""
        elif phase == "source_head":
            stdout = self.source_commit + "\n"
        elif phase == "source_status":
            stdout = (
                self.post_sparse_git_status_output
                if "sparse_build" in self.events
                else self.git_status_output
            )
        elif phase == "uv_venv":
            candidate = Path(command[-1])
            python = candidate / "bin" / "python"
            _write_executable(python)
            stdout = ""
        elif phase == "python_version":
            stdout = self.python_version + "\n"
        elif phase == "required_imports":
            names = {
                "interfuser": (
                    "torch",
                    "torchvision",
                    "numpy",
                    "cv2",
                    "timm",
                    "imgaug",
                    "skimage",
                ),
                "tfv6": (
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "numpy",
                    "timm",
                    "transformers",
                    "torchmetrics",
                ),
                "diffusiondrive_v2": (
                    "torch",
                    "torchvision",
                    "diffusers",
                    "einops",
                    "nuplan",
                ),
                "sparsedrive_v2": (
                    "torch",
                    "torchvision",
                    "nuplan",
                    "deformable_aggregation_ext",
                    "deformable_aggregation_with_depth_ext",
                ),
            }[self.model_id]
            candidate = Path(command[0]).parent.parent
            origins: list[str] = []
            for name in names:
                if self.model_id == "interfuser" and name == "timm":
                    origin = (
                        self.root
                        / "models/sources/interfuser/interfuser/timm/__init__.py"
                    )
                elif name in {
                    "deformable_aggregation_ext",
                    "deformable_aggregation_with_depth_ext",
                }:
                    origin = (
                        self.root
                        / "models/sources/sparsedrive_v2/navsim/agents/"
                        "sparsedrive/ops"
                        / f"{name}.cpython-39-x86_64-linux-gnu.so"
                    )
                else:
                    origin = candidate / "lib/python/site-packages" / name / "__init__.py"
                origins.append(str(origin))
            stdout = json.dumps(
                self.required_import_origins_override
                if self.required_import_origins_override is not None
                else origins
            )
        elif phase == "cuda_runtime":
            stdout = "1\n" if self.candidate_cuda_available else "0\n"
        elif phase == "freeze":
            stdout = self.freeze_output
        elif phase == "sparse_build":
            ops = Path(kwargs["cwd"])
            build_output = ops / "build" / "temp.linux-x86_64-cpython-39"
            build_output.mkdir(parents=True)
            (build_output / "generated.o").write_bytes(b"object")
            if self.write_sparse_extensions:
                for module in (
                    "deformable_aggregation_ext",
                    "deformable_aggregation_with_depth_ext",
                ):
                    (ops / f"{module}.cpython-39-x86_64-linux-gnu.so").write_bytes(
                        ("extension:" + module).encode("ascii")
                    )
            stdout = ""
        elif phase == "sparse_imports":
            ops = Path(command[-1])
            stdout = json.dumps(
                [
                    str(ops / f"{module}.cpython-39-x86_64-linux-gnu.so")
                    for module in (
                        "deformable_aggregation_ext",
                        "deformable_aggregation_with_depth_ext",
                    )
                ]
            )
        elif phase == "unexpected":
            raise AssertionError(f"unexpected setup command: {command!r}")
        else:
            stdout = ""
        hook = self.phase_hooks.get(phase)
        if hook is not None:
            hook(command, dict(kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _setup_fixture(
    tmp_path: Path,
    *,
    model_id: str = "tfv6",
    gpu_available: bool = True,
    recommended_policy: bool = False,
    strip_asset_hashes: bool = False,
    ros_smoke_hook: bool = True,
    build_ids: tuple[str, ...] = ("build-001", "build-002", "build-003"),
) -> tuple[
    Path,
    EnvironmentSetupDependencies,
    _SetupRunner,
    list[tuple[str, Any]],
]:
    """네트워크/GPU와 독립적인 setup project, dependency와 runner를 만든다."""
    root = _preflight_project(tmp_path)
    preflight_dependencies, _, _ = _preflight_host(
        tmp_path, gpu_available=gpu_available
    )
    for package in EXPECTED_ROS_PACKAGES:
        (
            preflight_dependencies.ros_python_root / package / "__init__.py"
        ).write_text("", encoding="utf-8")
    git_path = _write_executable(tmp_path / "host" / "commands" / "git")
    original_which = preflight_dependencies.which
    preflight_dependencies = dataclasses.replace(
        preflight_dependencies,
        which=lambda command: (
            str(git_path) if command == "git" else original_which(command)
        ),
    )
    if recommended_policy:
        original_select = preflight_dependencies.select_definition
        preflight_dependencies = dataclasses.replace(
            preflight_dependencies,
            select_definition=lambda manifest, selected: dataclasses.replace(
                original_select(manifest, selected), cuda_policy="recommended"
            ),
        )
    if strip_asset_hashes:
        original_load = preflight_dependencies.load_profile

        def load_without_asset_hashes(path: Path, selected: str):
            profile = original_load(path, selected)
            data = json.loads(json.dumps(profile.data))
            for declaration in data["runtime_assets"]:
                declaration["sha256"] = None
            return dataclasses.replace(profile, data=data)

        preflight_dependencies = dataclasses.replace(
            preflight_dependencies, load_profile=load_without_asset_hashes
        )

    manifest = preflight_dependencies.load_manifest(
        root
        / "src"
        / "Validator_pkg"
        / "validator_inference"
        / "config"
        / "model_environments.json"
    )
    definition = preflight_dependencies.select_definition(manifest, model_id)
    runner = _SetupRunner(root, model_id, definition.python_version)
    runner.source_commit = definition.source_commit
    events: list[tuple[str, Any]] = []
    checkout_names = {
        "tfv6": "lead",
        "interfuser": "interfuser",
        "diffusiondrive_v2": "diffusiondrive_v2",
        "sparsedrive_v2": "sparsedrive_v2",
    }

    def asset_fetcher(
        profile: dict[str, Any], *, model_id: str, project_root: Path
    ) -> Path:
        events.append(("asset_fetch", model_id))
        for declaration in profile["runtime_assets"]:
            asset = project_root / declaration["path"]
            asset.parent.mkdir(parents=True, exist_ok=True)
            if not asset.exists():
                asset.write_bytes(("asset:" + declaration["role"]).encode("ascii"))
        checkout = (
            project_root / "models" / "sources" / checkout_names[model_id]
        )
        checkout.mkdir(parents=True, exist_ok=True)
        if model_id == "sparsedrive_v2":
            (
                checkout / "navsim" / "agents" / "sparsedrive" / "ops"
            ).mkdir(parents=True, exist_ok=True)
        if model_id == "diffusiondrive_v2":
            derivative = (
                checkout
                / "navsim"
                / "agents"
                / "diffusiondrivev2"
                / "diffusiondrivev2_model_sel_inference.py"
            )
            derivative.parent.mkdir(parents=True, exist_ok=True)
            payload = b"DERIVATIVE = True\n"
            if not derivative.exists() or derivative.read_bytes() != payload:
                replacement = derivative.with_name(derivative.name + ".generated")
                replacement.write_bytes(payload)
                os.replace(replacement, derivative)
        return project_root / "models" / model_id / "asset-manifest.json"

    def asset_verifier(
        profile: dict[str, Any], *, project_root: Path
    ) -> dict[str, Any]:
        events.append(("asset_verify", profile["id"]))
        checkout = (
            project_root
            / "models"
            / "sources"
            / checkout_names[profile["id"]]
        )
        source: dict[str, Any] = {
            "checkout": str(checkout),
            "observed_commit": profile["source"]["commit"],
        }
        if profile["id"] == "diffusiondrive_v2":
            relative = Path(
                "navsim/agents/diffusiondrivev2/"
                "diffusiondrivev2_model_sel_inference.py"
            )
            source["local_derivative"] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(checkout / relative),
            }
        return {"source": source, "assets": []}

    identifiers = iter(build_ids)

    def smoke(python: Path, child_environment: dict[str, str]) -> None:
        events.append(("ros_smoke", (python, dict(child_environment))))

    dependencies = EnvironmentSetupDependencies(
        preflight_dependencies=preflight_dependencies,
        asset_fetcher=asset_fetcher,
        asset_verifier=asset_verifier,
        build_id=lambda: next(identifiers),
        utc_now=lambda: "2026-07-22T01:02:03Z",
        ros_smoke=smoke if ros_smoke_hook else None,
    )
    return root, dependencies, runner, events


def _setup_call(runner: _SetupRunner, phase: str):
    """runner 기록에서 유일한 setup phase call을 반환한다."""
    matches = [call for call in runner.calls if call[2] == phase]
    assert len(matches) == 1, (phase, matches)
    return matches[0]


def _selected_runtime_asset_path(
    root: Path,
    dependencies: EnvironmentSetupDependencies,
    model_id: str,
    index: int = 0,
) -> Path:
    """setup fixture profile에서 한 runtime asset의 절대 경로를 반환한다."""
    profile_path = root / "src" / "Validator_pkg" / "config" / "profile.json"
    selected = dependencies.preflight_dependencies.load_profile(profile_path, model_id)
    return root / selected.data["runtime_assets"][index]["path"]


def _mutate_file_preserving_size_and_mtime(path: Path) -> None:
    """fast fingerprint를 우회하는 same-size/mtime 변조를 만든다."""
    before = path.lstat()
    payload = path.read_bytes()
    assert payload
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


_DIFFUSION_GENERATED_PATH = Path(
    "models/sources/diffusiondrive_v2/navsim/agents/diffusiondrivev2/"
    "diffusiondrivev2_model_sel_inference.py"
)
_SPARSE_EXTENSION_PATHS = tuple(
    Path("models/sources/sparsedrive_v2/navsim/agents/sparsedrive/ops")
    / f"{module}.cpython-39-x86_64-linux-gnu.so"
    for module in (
        "deformable_aggregation_ext",
        "deformable_aggregation_with_depth_ext",
    )
)


def test_setup_public_result_and_success_phase_contract_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """새 candidate를 exact phase로 검증하고 state 기록 뒤에만 활성화한다."""
    root, dependencies, runner, events = _setup_fixture(tmp_path)
    lifecycle_events: list[str] = []
    real_write = environment_setup.write_environment_state
    real_activate = environment_setup.activate_candidate

    def record_write(path: Path, state: EnvironmentState) -> None:
        lifecycle_events.append("state")
        real_write(path, state)

    def record_activate(paths: EnvironmentPaths, candidate: Path) -> None:
        lifecycle_events.append("activate")
        real_activate(paths, candidate)

    monkeypatch.setattr(environment_setup, "write_environment_state", record_write)
    monkeypatch.setattr(environment_setup, "activate_candidate", record_activate)

    result = setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert tuple(field.name for field in dataclasses.fields(EnvironmentOperationResult)) == (
        "environment_dir",
        "state",
        "warnings",
        "changed",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.changed = False  # type: ignore[misc]
    assert result.changed is True
    assert result.environment_dir.name == "build-001"
    assert result.state == read_environment_state(result.environment_dir)
    assert result.warnings == ()
    assert environment_paths(root, "tfv6").current.resolve() == result.environment_dir
    assert lifecycle_events == ["state", "activate"]
    assert [event[0] for event in events] == [
        "asset_fetch",
        "asset_verify",
        "ros_smoke",
        "asset_verify",
        "asset_verify",
    ]
    assert [phase for _, _, phase in runner.calls] == [
        "host_uv",
        "catkin_build",
        "catkin_messages",
        "source_head",
        "source_status",
        "uv_venv",
        "uv_sync",
        "python_version",
        "required_imports",
        "cuda_runtime",
        "model_load",
        "freeze",
        "source_head",
        "source_status",
        "source_head",
        "source_status",
    ]
    assert all(call[1]["check"] is True for call in runner.calls)
    assert all(call[1]["shell"] is False for call in runner.calls)

    catkin_command, catkin_kwargs, _ = _setup_call(runner, "catkin_build")
    assert Path(catkin_command[0]).is_absolute()
    assert catkin_command == (str(dependencies.preflight_dependencies.which("catkin_make")),)
    assert catkin_kwargs["cwd"] == root
    message_command, message_kwargs, _ = _setup_call(runner, "catkin_messages")
    assert message_command[:3] == ("/usr/bin/python3", "-B", "-c")
    assert str(root / "devel" / "lib" / "python3" / "dist-packages") in message_command[3]
    assert str(dependencies.preflight_dependencies.ros_python_root) in message_command[3]
    assert all(
        name in message_command[3]
        for name in ("CameraCalibration", "DrivingFeature", "InferenceMetric")
    )
    assert message_kwargs["cwd"] == root

    candidate = result.environment_dir
    assert _setup_call(runner, "uv_venv")[0] == build_uv_venv_argv(
        result.state and Path(dependencies.preflight_dependencies.which("uv")),
        dependencies.preflight_dependencies.select_definition(
            dependencies.preflight_dependencies.load_manifest(
                root
                / "src"
                / "Validator_pkg"
                / "validator_inference"
                / "config"
                / "model_environments.json"
            ),
            "tfv6",
        ),
        candidate,
    )
    definition = dependencies.preflight_dependencies.select_definition(
        dependencies.preflight_dependencies.load_manifest(
            root
            / "src"
            / "Validator_pkg"
            / "validator_inference"
            / "config"
            / "model_environments.json"
        ),
        "tfv6",
    )
    assert _setup_call(runner, "uv_sync")[0] == build_uv_sync_argv(
        Path(dependencies.preflight_dependencies.which("uv")), definition, candidate
    )
    normalized = b"a-package==1\nz-package==2\n"
    assert result.state.installed_packages_sha256 == hashlib.sha256(normalized).hexdigest()
    freeze_command, _, _ = _setup_call(runner, "freeze")
    assert freeze_command == (
        str(Path(dependencies.preflight_dependencies.which("uv"))),
        "pip",
        "freeze",
        "--python",
        str(result.environment_dir / "bin" / "python"),
        "--strict",
        "--no-python-downloads",
    )
    assert result.state.assets == fingerprint_runtime_assets(
        preflight_environment(
            root,
            "tfv6",
            runner=runner,
            dependencies=dependencies.preflight_dependencies,
        )
    )


def test_setup_host_blocker_runs_before_lock_or_any_mutator(tmp_path: Path):
    """host preflight 실패는 .venvs, catkin, asset callback을 전혀 만들지 않는다."""
    root, dependencies, runner, events = _setup_fixture(
        tmp_path, gpu_available=False
    )

    with pytest.raises(EnvironmentSetupError, match="CUDA GPU is required") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert caught.value.action == "host_preflight"
    assert [phase for _, _, phase in runner.calls] == ["host_uv"]
    assert events == []
    assert not (root / ".venvs").exists()
    assert not (root / "devel").exists()


def test_setup_failed_sync_retains_candidate_and_preserves_old_current(tmp_path: Path):
    """candidate phase 실패는 state를 쓰지 않고 기존 current를 그대로 보존한다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    paths = environment_paths(root, "tfv6")
    old = create_candidate(paths, "old-build")
    paths.current.symlink_to(Path("builds/old-build"))
    runner.fail_phase = "uv_sync"

    with pytest.raises(EnvironmentSetupError) as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    candidate = paths.builds / "build-001"
    assert caught.value.action == "environment_sync"
    assert paths.current.resolve() == old
    assert candidate.is_dir()
    assert not (candidate / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_reuses_deep_valid_current_and_recreate_forces_new_build(tmp_path: Path):
    """deep-valid current는 재사용하고 recreate는 항상 새 candidate를 활성화한다."""
    root, dependencies, first_runner, events = _setup_fixture(tmp_path)
    first = setup_environment(
        root, "tfv6", runner=first_runner, dependencies=dependencies
    )
    events.clear()
    second_runner = _SetupRunner(root, "tfv6", first.state.python_version)
    second_runner.source_commit = first.state.source_commit

    reused = setup_environment(
        root, "tfv6", runner=second_runner, dependencies=dependencies
    )

    assert reused == EnvironmentOperationResult(
        environment_dir=first.environment_dir,
        state=first.state,
        warnings=(),
        changed=False,
    )
    assert not any(
        phase in {"uv_venv", "uv_sync"}
        for _, _, phase in second_runner.calls
    )
    assert events[0] == ("asset_verify", "tfv6")

    recreate_runner = _SetupRunner(root, "tfv6", first.state.python_version)
    recreate_runner.source_commit = first.state.source_commit
    recreated = setup_environment(
        root,
        "tfv6",
        recreate=True,
        runner=recreate_runner,
        dependencies=dependencies,
    )
    assert recreated.changed is True
    assert recreated.environment_dir.name == "build-002"
    assert recreated.environment_dir != first.environment_dir
    assert environment_paths(root, "tfv6").current.resolve() == recreated.environment_dir
    assert first.environment_dir.is_dir()


def test_setup_reuse_deep_hash_rejects_same_metadata_asset_tamper(tmp_path: Path):
    """size와 mtime을 복원한 asset 변조도 deep reuse 검사에서 차단한다."""
    root, dependencies, first_runner, _ = _setup_fixture(tmp_path)
    first = setup_environment(
        root, "tfv6", runner=first_runner, dependencies=dependencies
    )
    asset = root / first.state.assets[0].path
    before = asset.lstat()
    payload = asset.read_bytes()
    asset.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    os.utime(asset, ns=(before.st_atime_ns, before.st_mtime_ns))
    runner = _SetupRunner(root, "tfv6", first.state.python_version)
    runner.source_commit = first.state.source_commit

    with pytest.raises(EnvironmentSetupError, match="asset"):
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert environment_paths(root, "tfv6").current.resolve() == first.environment_dir
    assert not (environment_paths(root, "tfv6").builds / "build-002").exists()


@pytest.mark.parametrize("preflight_variant", ("stale", "missing", "extra"))
def test_setup_reuse_rejects_nonexact_preflight_identifiers(
    tmp_path: Path, preflight_variant: str
):
    """재사용 state는 현재 setup이 완료한 exact preflight 집합을 증명해야 한다."""
    root, dependencies, first_runner, _ = _setup_fixture(tmp_path)
    first = setup_environment(
        root, "tfv6", runner=first_runner, dependencies=dependencies
    )
    expected = first.state.preflight_ids
    variants = {
        "stale": ("legacy_host",) + expected[1:],
        "missing": expected[:-1],
        "extra": expected + ("future_check",),
    }
    stale_state = dataclasses.replace(
        first.state, preflight_ids=variants[preflight_variant]
    )
    environment_setup.write_environment_state(
        first.environment_dir / ENVIRONMENT_STATE_FILENAME, stale_state
    )
    runner = _SetupRunner(root, "tfv6", first.state.python_version)
    runner.source_commit = first.state.source_commit

    with pytest.raises(EnvironmentSetupError, match="preflight") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    paths = environment_paths(root, "tfv6")
    assert caught.value.action == "reuse_verification"
    assert paths.current.resolve() == first.environment_dir
    assert not (paths.builds / "build-002").exists()


def test_setup_propagates_preflight_warning_without_duplicate_runtime_warning(
    tmp_path: Path,
):
    """host/runtime CUDA가 모두 없으면 recommended warning을 중복 기록하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, gpu_available=False, recommended_policy=True
    )
    runner.candidate_cuda_available = False

    result = setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert result.warnings == (
        "CUDA GPU is not available; continuing because "
        "tfv6 declares cuda_policy=recommended",
    )
    assert any(phase == "cuda_runtime" for _, _, phase in runner.calls)


def test_setup_warns_when_host_gpu_exists_but_recommended_candidate_cuda_is_absent(
    tmp_path: Path,
):
    """host probe 성공 뒤 candidate torch CUDA 부재도 recommended 결과에 기록한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, gpu_available=True, recommended_policy=True
    )
    runner.candidate_cuda_available = False

    result = setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert result.warnings == (
        "candidate torch CUDA is not available; continuing because "
        "tfv6 declares cuda_policy=recommended",
    )


def test_setup_sparse_runs_exact_declared_extension_action(tmp_path: Path):
    """SparseDrive만 exact build_ext 환경과 두 extension origin import를 실행한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, model_id="sparsedrive_v2", strip_asset_hashes=True
    )

    result = setup_environment(
        root, "sparsedrive_v2", runner=runner, dependencies=dependencies
    )

    build_command, build_kwargs, _ = _setup_call(runner, "sparse_build")
    assert build_command == (
        str(result.environment_dir / "bin" / "python"),
        "setup.py",
        "build_ext",
        "--inplace",
    )
    assert build_kwargs["cwd"] == (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
    )
    assert {
        key: build_kwargs["env"][key]
        for key in ("CUDA_HOME", "FORCE_CUDA", "TORCH_CUDA_ARCH_LIST", "CC", "CXX")
    } == {
        "CUDA_HOME": str(dependencies.preflight_dependencies.cuda_root),
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": str(dependencies.preflight_dependencies.gcc_path),
        "CXX": str(dependencies.preflight_dependencies.gxx_path),
    }
    import_command, _, _ = _setup_call(runner, "sparse_imports")
    assert "deformable_aggregation_ext" in import_command[2]
    assert "deformable_aggregation_with_depth_ext" in import_command[2]
    assert "sys.path.insert" in import_command[2]
    assert str(build_kwargs["cwd"]) in import_command
    required_imports = _setup_call(runner, "required_imports")
    assert required_imports[1]["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        build_kwargs["cwd"]
    )
    assert [phase for _, _, phase in runner.calls].index("sparse_build") < [
        phase for _, _, phase in runner.calls
    ].index("sparse_imports")


def test_setup_postinstall_cuda_failure_never_writes_state_or_switches_current(
    tmp_path: Path,
):
    """host GPU가 있어도 candidate torch CUDA가 없으면 activation 전에 차단한다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    paths = environment_paths(root, "tfv6")
    old = create_candidate(paths, "old-build")
    paths.current.symlink_to(Path("builds/old-build"))
    runner.fail_phase = "cuda_runtime"

    with pytest.raises(EnvironmentSetupError) as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    candidate = paths.builds / "build-001"
    assert caught.value.action == "cuda_runtime"
    assert paths.current.resolve() == old
    assert candidate.is_dir()
    assert not (candidate / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_default_ros_smoke_uses_full_compatibility_and_base_stdlib_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """hook이 없으면 resolved target base stdlib 하나로 양방향 ROS smoke를 실행한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, ros_smoke_hook=False
    )
    base_python = _write_executable(
        tmp_path / "managed-python" / "bin" / "python3.10"
    )
    base_stdlib = base_python.parent.parent / "lib" / "python3.10"
    base_stdlib.mkdir(parents=True)
    observed: list[Any] = []

    def link_candidate_python(command: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
        candidate_python = Path(command[-1]) / "bin" / "python"
        candidate_python.unlink()
        candidate_python.symlink_to(base_python)

    runner.phase_hooks["uv_venv"] = link_candidate_python
    monkeypatch.setattr(
        environment_setup,
        "_run_ros_compatibility_smoke",
        lambda policy: observed.append(policy),
        raising=False,
    )

    result = setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert len(observed) == 1
    policy = observed[0]
    assert policy.target_python == result.environment_dir / "bin" / "python"
    assert policy.target_environment == result.environment_dir
    assert policy.ros_overlay == result.environment_dir / ".ros-overlay"
    assert policy.catkin_python_path == (
        root / "devel" / "lib" / "python3" / "dist-packages"
    )
    assert policy.target_stdlib_paths == (base_stdlib.resolve(),)
    assert policy.project_source_paths == (
        root / "src" / "Validator_pkg" / "validator_inference" / "src",
    )
    assert not any(phase == "target_ros_import" for _, _, phase in runner.calls)


def test_setup_rejects_dirty_source_before_candidate_creation(tmp_path: Path):
    """tracked checkout 변경은 asset 검증 뒤 candidate 생성 전에 차단한다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    runner.git_status_output = " M tracked.py\n"

    with pytest.raises(EnvironmentSetupError, match="dirty") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert caught.value.action == "source_verification"
    assert not (environment_paths(root, "tfv6").builds / "build-001").exists()


def test_setup_allows_only_exact_verified_diffusion_derivative_dirty_entry(
    tmp_path: Path,
):
    """Diffusion의 deep-verified generated derivative 한 경로만 dirty 예외로 허용한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    runner.git_status_output = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )

    result = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    assert result.changed is True


def test_setup_diffusion_state_fingerprints_derivative_and_reuse_rejects_tamper(
    tmp_path: Path,
):
    """재생성된 derivative SHA를 state에 고정하고 self-attested 변조를 거부한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status

    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    derivative = root / _DIFFUSION_GENERATED_PATH
    assert first.state.assets[-1] == AssetFingerprint(
        path=_DIFFUSION_GENERATED_PATH,
        sha256=sha256_file(derivative),
        size=derivative.lstat().st_size,
        mtime_ns=derivative.lstat().st_mtime_ns,
    )
    _mutate_file_preserving_size_and_mtime(derivative)
    reuse_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    reuse_runner.source_commit = first.state.source_commit
    reuse_runner.git_status_output = derivative_status

    with pytest.raises(EnvironmentSetupError, match="generated|fingerprint|SHA") as caught:
        setup_environment(
            root,
            "diffusiondrive_v2",
            runner=reuse_runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "reuse_verification"
    assert environment_paths(root, "diffusiondrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_diffusion_recreate_failure_restores_active_derivative_fingerprint(
    tmp_path: Path,
):
    """changed derivative 뒤 model failure는 old inode/mtime/content를 복구한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    expected = first.state.assets[-1]
    derivative = root / _DIFFUSION_GENERATED_PATH
    expected_inode = derivative.stat().st_ino

    def publish_changed_derivative(
        profile: dict[str, Any], *, model_id: str, project_root: Path
    ) -> Path:
        assert model_id == "diffusiondrive_v2"
        replacement = derivative.with_name(derivative.name + ".changed")
        replacement.write_text("DERIVATIVE = 'changed'\n", encoding="utf-8")
        os.replace(replacement, derivative)
        return project_root / "models" / profile["id"] / "asset-manifest.json"

    changed_dependencies = dataclasses.replace(
        dependencies, asset_fetcher=publish_changed_derivative
    )
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status
    recreate_runner.fail_phase = "model_load"

    with pytest.raises(EnvironmentSetupError, match="model_load"):
        setup_environment(
            root,
            "diffusiondrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=changed_dependencies,
        )

    metadata = derivative.stat()
    observed = AssetFingerprint(
        path=_DIFFUSION_GENERATED_PATH,
        sha256=sha256_file(derivative),
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    inspection_preflight = preflight_environment(
        root,
        "diffusiondrive_v2",
        runner=_SetupRunner(
            root, "diffusiondrive_v2", first.state.python_version
        ),
        dependencies=dependencies.preflight_dependencies,
    )
    inspection = environment_setup.inspect_active_environment(
        inspection_preflight.paths,
        inspection_preflight.definition,
        sha256_file(inspection_preflight.manifest.manifest_path),
    )
    assert observed == expected
    assert derivative.stat().st_ino == expected_inode
    assert inspection_preflight.paths.current.resolve() == first.environment_dir
    assert inspection.valid is True


def test_setup_diffusion_equal_recreate_inplace_mutation_restores_active_derivative(
    tmp_path: Path,
):
    """equal recreate의 새 derivative가 in-place 변조돼도 old inode를 복구한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    derivative = root / _DIFFUSION_GENERATED_PATH
    expected = first.state.assets[-1]
    expected_inode = derivative.stat().st_ino
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status
    recreate_runner.phase_hooks["model_load"] = lambda _command, _kwargs: (
        derivative.write_bytes(derivative.read_bytes() + b"# concurrent mutation\n")
    )
    recreate_runner.fail_phase = "freeze"

    with pytest.raises(EnvironmentSetupError, match="freeze"):
        setup_environment(
            root,
            "diffusiondrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    metadata = derivative.stat()
    assert AssetFingerprint(
        path=_DIFFUSION_GENERATED_PATH,
        sha256=sha256_file(derivative),
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    ) == expected
    assert metadata.st_ino == expected_inode
    assert environment_paths(root, "diffusiondrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_diffusion_post_activation_verification_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """activation 뒤 derivative 변조도 old derivative/current로 rollback한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    derivative = root / _DIFFUSION_GENERATED_PATH
    expected = first.state.assets[-1]
    expected_inode = derivative.stat().st_ino

    def publish_changed_derivative(
        profile: dict[str, Any], *, model_id: str, project_root: Path
    ) -> Path:
        assert model_id == "diffusiondrive_v2"
        replacement = derivative.with_name(derivative.name + ".changed")
        replacement.write_text("DERIVATIVE = 'changed'\n", encoding="utf-8")
        os.replace(replacement, derivative)
        return project_root / "models" / profile["id"] / "asset-manifest.json"

    real_activate = environment_setup.activate_candidate

    def activate_then_mutate(paths: EnvironmentPaths, candidate: Path) -> None:
        real_activate(paths, candidate)
        derivative.write_bytes(derivative.read_bytes() + b"# post activation\n")

    monkeypatch.setattr(
        environment_setup, "activate_candidate", activate_then_mutate
    )
    changed_dependencies = dataclasses.replace(
        dependencies, asset_fetcher=publish_changed_derivative
    )
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status

    with pytest.raises(EnvironmentSetupError, match="changed before transaction commit"):
        setup_environment(
            root,
            "diffusiondrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=changed_dependencies,
        )

    metadata = derivative.stat()
    observed = AssetFingerprint(
        path=_DIFFUSION_GENERATED_PATH,
        sha256=sha256_file(derivative),
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )
    inspection_preflight = preflight_environment(
        root,
        "diffusiondrive_v2",
        runner=_SetupRunner(
            root, "diffusiondrive_v2", first.state.python_version
        ),
        dependencies=dependencies.preflight_dependencies,
    )
    inspection = environment_setup.inspect_active_environment(
        inspection_preflight.paths,
        inspection_preflight.definition,
        sha256_file(inspection_preflight.manifest.manifest_path),
    )
    assert observed == expected
    assert derivative.stat().st_ino == expected_inode
    assert inspection_preflight.paths.current.resolve() == first.environment_dir
    assert inspection.valid is True


def test_setup_diffusion_backup_cleanup_failure_keeps_successful_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """logical activation 뒤 backup cleanup 실패는 warning과 backup을 남긴다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    def fail_backup_cleanup(
        transaction: environment_setup._DiffusionDerivativeTransaction,
    ) -> None:
        raise EnvironmentSetupError(
            "diffusion_derivative", "forced backup cleanup failure"
        )

    monkeypatch.setattr(
        environment_setup._DiffusionDerivativeTransaction,
        "_remove_backup_link",
        fail_backup_cleanup,
    )
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status

    result = setup_environment(
        root,
        "diffusiondrive_v2",
        recreate=True,
        runner=recreate_runner,
        dependencies=dependencies,
    )

    backup_directories = tuple(
        environment_paths(root, "diffusiondrive_v2").model_root.glob(
            ".diffusion-derivative-backup.*"
        )
    )
    assert result.changed is True
    assert environment_paths(root, "diffusiondrive_v2").current.resolve() == (
        result.environment_dir
    )
    assert any(
        "generated artifact backup cleanup deferred" in warning
        and "forced backup cleanup failure" in warning
        for warning in result.warnings
    )
    assert len(backup_directories) == 1
    assert (
        backup_directories[0]
        / "diffusiondrivev2_model_sel_inference.py"
    ).is_file()


def test_setup_finalize_runtime_error_is_success_warning_after_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """activation 뒤 ordinary finalize 오류는 성공 결과의 warning으로 반환한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    def fail_finalize(
        _transaction: environment_setup._DiffusionDerivativeTransaction,
    ) -> None:
        raise RuntimeError("forced ordinary finalize failure")

    monkeypatch.setattr(
        environment_setup._DiffusionDerivativeTransaction,
        "finalize_commit",
        fail_finalize,
    )
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status

    result = setup_environment(
        root,
        "diffusiondrive_v2",
        recreate=True,
        runner=recreate_runner,
        dependencies=dependencies,
    )

    assert result.changed is True
    assert environment_paths(root, "diffusiondrive_v2").current.resolve() == (
        result.environment_dir
    )
    assert any(
        "generated artifact backup cleanup deferred" in warning
        and "forced ordinary finalize failure" in warning
        for warning in result.warnings
    )


@pytest.mark.parametrize(
    "control_error",
    (KeyboardInterrupt(), SystemExit(9)),
    ids=("keyboard-interrupt", "system-exit"),
)
def test_setup_finalize_control_exception_propagates_after_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_error: BaseException,
):
    """activation 뒤 process control 예외는 cleanup warning으로 삼키지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    derivative_status = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.git_status_output = derivative_status
    first = setup_environment(
        root,
        "diffusiondrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    def abort_finalize(
        _transaction: environment_setup._DiffusionDerivativeTransaction,
    ) -> None:
        raise control_error

    monkeypatch.setattr(
        environment_setup._DiffusionDerivativeTransaction,
        "finalize_commit",
        abort_finalize,
    )
    recreate_runner = _SetupRunner(
        root, "diffusiondrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = derivative_status

    with pytest.raises(type(control_error)):
        setup_environment(
            root,
            "diffusiondrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    assert environment_paths(root, "diffusiondrive_v2").current.resolve() == (
        environment_paths(root, "diffusiondrive_v2").builds / "build-002"
    )


def test_setup_diffusion_fresh_failure_removes_generated_derivative(tmp_path: Path):
    """active derivative가 없는 fresh setup failure는 생성 파일을 철회한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    runner.git_status_output = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    runner.fail_phase = "model_load"

    with pytest.raises(EnvironmentSetupError, match="model_load"):
        setup_environment(
            root,
            "diffusiondrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert not (root / _DIFFUSION_GENERATED_PATH).exists()
    assert not environment_paths(root, "diffusiondrive_v2").current.exists()


def test_setup_diffusion_fresh_cleanup_fsync_failure_finishes_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """fresh derivative 삭제 fsync 실패도 canonical과 backup을 정리한 뒤 보고한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    runner.git_status_output = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    armed = False
    failed = False
    real_fsync = os.fsync

    def arm_cleanup(_command: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal armed
        armed = True

    def fail_first_cleanup_fsync(descriptor: int) -> None:
        nonlocal failed
        if armed and not failed:
            failed = True
            raise OSError("forced generated cleanup fsync failure")
        real_fsync(descriptor)

    runner.phase_hooks["model_load"] = arm_cleanup
    runner.fail_phase = "freeze"
    monkeypatch.setattr(environment_setup.os, "fsync", fail_first_cleanup_fsync)

    with pytest.raises(EnvironmentSetupError, match="quarantine"):
        setup_environment(
            root,
            "diffusiondrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    paths = environment_paths(root, "diffusiondrive_v2")
    assert not (root / _DIFFUSION_GENERATED_PATH).exists()
    assert not paths.current.exists()
    assert not tuple(paths.model_root.glob(".diffusion-derivative-backup.*"))


def test_setup_sparse_state_fingerprints_exact_built_extensions(tmp_path: Path):
    """Sparse import가 선택한 두 in-place extension만 state asset으로 기록한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    runner.post_sparse_git_status_output = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )

    result = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )

    generated = result.state.assets[-2:]
    assert tuple(item.path for item in generated) == _SPARSE_EXTENSION_PATHS
    assert tuple(item.sha256 for item in generated) == tuple(
        sha256_file(root / path) for path in _SPARSE_EXTENSION_PATHS
    )
    assert not (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
        / "build"
    ).exists()
    sparse_import_call = _setup_call(runner, "sparse_imports")
    assert sparse_import_call[1]["capture_output"] is True
    assert sparse_import_call[1]["text"] is True


def test_setup_sparse_reuse_rejects_saved_extension_sha_mismatch(tmp_path: Path):
    """Sparse reuse는 같은 size/mtime의 compiled extension 변조도 SHA로 거부한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    _mutate_file_preserving_size_and_mtime(root / _SPARSE_EXTENSION_PATHS[0])
    reuse_runner = _SetupRunner(root, "sparsedrive_v2", first.state.python_version)
    reuse_runner.source_commit = first.state.source_commit
    reuse_runner.git_status_output = extension_status

    with pytest.raises(EnvironmentSetupError, match="fingerprint|SHA") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=reuse_runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "reuse_verification"
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_sparse_recreate_allows_exact_existing_extensions(tmp_path: Path):
    """성공 setup이 남긴 exact 두 extension은 recreate source 검증을 통과한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status

    recreated = setup_environment(
        root,
        "sparsedrive_v2",
        recreate=True,
        runner=recreate_runner,
        dependencies=dependencies,
    )

    assert recreated.changed is True
    assert recreated.environment_dir.name == "build-002"
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        recreated.environment_dir
    )


def test_setup_sparse_rejects_unattested_preexisting_exact_extensions(
    tmp_path: Path,
):
    """active state가 없는 exact-name .so는 candidate 생성 전에 거부한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    for index, path in enumerate(_SPARSE_EXTENSION_PATHS):
        absolute = root / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_bytes(f"untrusted:{index}".encode("ascii"))
    runner.git_status_output = extension_status
    runner.post_sparse_git_status_output = extension_status

    with pytest.raises(EnvironmentSetupError, match="active state|authenticated"):
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert not any(phase == "uv_venv" for _, _, phase in runner.calls)
    assert not environment_paths(root, "sparsedrive_v2").current.exists()


def test_setup_sparse_recreate_rejects_extension_not_matching_active_state(
    tmp_path: Path,
):
    """recreate도 active state 저장 SHA와 다른 preexisting .so를 신뢰하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    _mutate_file_preserving_size_and_mtime(root / _SPARSE_EXTENSION_PATHS[0])
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status

    with pytest.raises(EnvironmentSetupError, match="SHA|authenticated"):
        setup_environment(
            root,
            "sparsedrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    assert not any(phase == "uv_venv" for _, _, phase in recreate_runner.calls)
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_sparse_stale_state_requires_extension_rebuild(tmp_path: Path):
    """stale lock state의 authenticated .so를 no-op build로 재인증하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    stale_state = dataclasses.replace(first.state, lock_sha256="0" * 64)
    environment_setup.write_environment_state(
        first.environment_dir / ENVIRONMENT_STATE_FILENAME, stale_state
    )
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status
    recreate_runner.write_sparse_extensions = False

    with pytest.raises(EnvironmentSetupError, match="rebuild|unchanged") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "sparse_extensions"
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_sparse_stale_state_rejects_metadata_only_extension_replacement(
    tmp_path: Path,
):
    """기존 .so bytes를 그대로 inode만 바꾼 build는 fresh rebuild 증명이 아니다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    stale_state = dataclasses.replace(first.state, lock_sha256="0" * 64)
    environment_setup.write_environment_state(
        first.environment_dir / ENVIRONMENT_STATE_FILENAME, stale_state
    )
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status
    recreate_runner.write_sparse_extensions = False

    def replace_metadata_only(
        _command: tuple[str, ...], _kwargs: dict[str, Any]
    ) -> None:
        for path in _SPARSE_EXTENSION_PATHS:
            absolute = root / path
            if not absolute.exists():
                continue
            replacement = absolute.with_name(absolute.name + ".metadata-only")
            replacement.write_bytes(absolute.read_bytes())
            os.replace(replacement, absolute)

    recreate_runner.phase_hooks["sparse_build"] = replace_metadata_only

    with pytest.raises(EnvironmentSetupError, match="fresh|create|rebuild"):
        setup_environment(
            root,
            "sparsedrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )


def test_setup_sparse_stale_state_accepts_rebuilt_extensions(tmp_path: Path):
    """stale state라도 build가 두 .so identity를 교체하면 새 state로 복구한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    stale_state = dataclasses.replace(first.state, lock_sha256="0" * 64)
    environment_setup.write_environment_state(
        first.environment_dir / ENVIRONMENT_STATE_FILENAME, stale_state
    )
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status

    def replace_extensions(
        _command: tuple[str, ...], _kwargs: dict[str, Any]
    ) -> None:
        for path in _SPARSE_EXTENSION_PATHS:
            absolute = root / path
            replacement = absolute.with_name(absolute.name + ".rebuilt")
            replacement.write_bytes(absolute.read_bytes() + b":rebuilt")
            os.replace(replacement, absolute)

    recreate_runner.phase_hooks["sparse_build"] = replace_extensions

    recreated = setup_environment(
        root,
        "sparsedrive_v2",
        recreate=True,
        runner=recreate_runner,
        dependencies=dependencies,
    )

    assert recreated.changed is True
    assert recreated.state.lock_sha256 == first.state.lock_sha256
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        recreated.environment_dir
    )


def test_setup_sparse_recreate_failure_restores_active_extension_fingerprints(
    tmp_path: Path,
):
    """changed .so 뒤 model load 실패는 old pair를 exact metadata/SHA로 복구한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    expected = first.state.assets[-len(_SPARSE_EXTENSION_PATHS) :]
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status
    recreate_runner.fail_phase = "model_load"

    def change_extensions(
        _command: tuple[str, ...], _kwargs: dict[str, Any]
    ) -> None:
        for path in _SPARSE_EXTENSION_PATHS:
            absolute = root / path
            absolute.write_bytes(absolute.read_bytes() + b":new-build")

    recreate_runner.phase_hooks["sparse_build"] = change_extensions

    with pytest.raises(EnvironmentSetupError, match="model_load"):
        setup_environment(
            root,
            "sparsedrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    observed = tuple(
        AssetFingerprint(
            path=fingerprint.path,
            sha256=sha256_file(root / fingerprint.path),
            size=(root / fingerprint.path).stat().st_size,
            mtime_ns=(root / fingerprint.path).stat().st_mtime_ns,
        )
        for fingerprint in expected
    )
    paths = environment_paths(root, "sparsedrive_v2")
    inspection_preflight = preflight_environment(
        root,
        "sparsedrive_v2",
        runner=_SetupRunner(root, "sparsedrive_v2", first.state.python_version),
        dependencies=dependencies.preflight_dependencies,
    )
    inspection = environment_setup.inspect_active_environment(
        paths,
        inspection_preflight.definition,
        sha256_file(inspection_preflight.manifest.manifest_path),
    )
    assert observed == expected
    assert paths.current.resolve() == first.environment_dir
    assert inspection.valid is True


def test_setup_sparse_cleanup_unlink_failure_restores_active_extensions_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """new output quarantine cleanup 실패 뒤에도 old pair와 current를 복구한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    extension_status = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.post_sparse_git_status_output = extension_status
    first = setup_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies,
    )
    expected = first.state.assets[-len(_SPARSE_EXTENSION_PATHS) :]
    recreate_runner = _SetupRunner(
        root, "sparsedrive_v2", first.state.python_version
    )
    recreate_runner.source_commit = first.state.source_commit
    recreate_runner.git_status_output = extension_status
    recreate_runner.post_sparse_git_status_output = extension_status
    armed = False
    real_unlink = os.unlink

    def arm_cleanup(_command: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal armed
        armed = True

    def fail_first_quarantine_unlink(
        path: str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal armed
        if (
            armed
            and isinstance(path, str)
            and path.startswith(".validator-remove.")
            and kwargs.get("dir_fd") is not None
        ):
            armed = False
            raise OSError("forced quarantine unlink failure")
        real_unlink(path, *args, **kwargs)

    recreate_runner.phase_hooks["model_load"] = arm_cleanup
    recreate_runner.fail_phase = "freeze"
    monkeypatch.setattr(environment_setup.os, "unlink", fail_first_quarantine_unlink)

    with pytest.raises(EnvironmentSetupError, match="quarantine"):
        setup_environment(
            root,
            "sparsedrive_v2",
            recreate=True,
            runner=recreate_runner,
            dependencies=dependencies,
        )

    observed = tuple(
        AssetFingerprint(
            path=fingerprint.path,
            sha256=sha256_file(root / fingerprint.path),
            size=(root / fingerprint.path).stat().st_size,
            mtime_ns=(root / fingerprint.path).stat().st_mtime_ns,
        )
        for fingerprint in expected
    )
    assert observed == expected
    assert environment_paths(root, "sparsedrive_v2").current.resolve() == (
        first.environment_dir
    )


def test_setup_sparse_fresh_failure_removes_generated_extensions(tmp_path: Path):
    """active state가 없는 build의 later failure는 새 exact .so를 남기지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    runner.post_sparse_git_status_output = "".join(
        f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
        for path in _SPARSE_EXTENSION_PATHS
    )
    runner.fail_phase = "model_load"

    with pytest.raises(EnvironmentSetupError, match="model_load"):
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert all(not (root / path).exists() for path in _SPARSE_EXTENSION_PATHS)
    assert not environment_paths(root, "sparsedrive_v2").current.exists()


def test_setup_revalidates_runtime_asset_immediately_before_state_and_activation(
    tmp_path: Path,
):
    """긴 candidate build 중 runtime asset 변조는 state/current를 남기지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    paths = environment_paths(root, "tfv6")
    old = create_candidate(paths, "old-build")
    paths.current.symlink_to(Path("builds/old-build"))
    runtime_asset = _selected_runtime_asset_path(root, dependencies, "tfv6")
    runner.phase_hooks["model_load"] = lambda _command, _kwargs: (
        _mutate_file_preserving_size_and_mtime(runtime_asset)
    )

    with pytest.raises(EnvironmentSetupError, match="changed during setup") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert caught.value.action == "setup_revalidation"
    assert paths.current.resolve() == old
    assert not (paths.builds / "build-001" / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_revalidates_again_after_state_write_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """state write 중 asset이 바뀌면 completion state를 지우고 old current를 보존한다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    paths = environment_paths(root, "tfv6")
    old = create_candidate(paths, "old-build")
    paths.current.symlink_to(Path("builds/old-build"))
    runtime_asset = _selected_runtime_asset_path(root, dependencies, "tfv6")
    real_write = environment_setup.write_environment_state

    def write_then_mutate(path: Path, state: EnvironmentState) -> None:
        real_write(path, state)
        _mutate_file_preserving_size_and_mtime(runtime_asset)

    monkeypatch.setattr(environment_setup, "write_environment_state", write_then_mutate)

    with pytest.raises(EnvironmentSetupError, match="changed during setup") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert caught.value.action == "setup_revalidation"
    assert paths.current.resolve() == old
    assert not (paths.builds / "build-001" / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_revalidates_source_immediately_before_state_and_activation(
    tmp_path: Path,
):
    """model load 중 source checkout 변경도 검증 완료 state로 승격하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    paths = environment_paths(root, "tfv6")
    old = create_candidate(paths, "old-build")
    paths.current.symlink_to(Path("builds/old-build"))
    runner.phase_hooks["model_load"] = lambda _command, _kwargs: setattr(
        runner, "git_status_output", " M tracked.py\n"
    )

    with pytest.raises(EnvironmentSetupError, match="dirty") as caught:
        setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)

    assert caught.value.action == "source_verification"
    assert paths.current.resolve() == old
    assert not (paths.builds / "build-001" / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_revalidates_generated_derivative_before_activation(tmp_path: Path):
    """model load 중 derivative 변조는 verifier의 self hash와 무관하게 실패한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="diffusiondrive_v2",
        strip_asset_hashes=True,
    )
    runner.git_status_output = (
        "?? navsim/agents/diffusiondrivev2/"
        "diffusiondrivev2_model_sel_inference.py\n"
    )
    derivative = root / _DIFFUSION_GENERATED_PATH
    runner.phase_hooks["model_load"] = lambda _command, _kwargs: (
        _mutate_file_preserving_size_and_mtime(derivative)
    )

    with pytest.raises(EnvironmentSetupError, match="changed during setup") as caught:
        setup_environment(
            root,
            "diffusiondrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "setup_revalidation"
    paths = environment_paths(root, "diffusiondrive_v2")
    assert not paths.current.exists()
    assert not (paths.builds / "build-001" / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_sparse_rejects_build_dirty_output_outside_exact_extensions(
    tmp_path: Path,
):
    """Sparse build 후 exact extension 외 source 출력은 activation 전에 차단한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    runner.post_sparse_git_status_output = "".join(
        [
            *(
                f"?? {path.relative_to('models/sources/sparsedrive_v2').as_posix()}\n"
                for path in _SPARSE_EXTENSION_PATHS
            ),
            "?? navsim/agents/sparsedrive/ops/unexpected_build_output.py\n",
        ]
    )

    with pytest.raises(EnvironmentSetupError, match="dirty") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "source_verification"
    paths = environment_paths(root, "sparsedrive_v2")
    assert not paths.current.exists()
    assert not (paths.builds / "build-001" / ENVIRONMENT_STATE_FILENAME).exists()


def test_setup_sparse_preserves_and_rejects_preexisting_build_tree(tmp_path: Path):
    """setup은 사용자가 남긴 ops/build를 삭제하거나 재사용하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    build = (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
        / "build"
    )
    build.mkdir(parents=True)
    user_file = build / "user-diagnostic.txt"
    user_file.write_text("keep\n", encoding="utf-8")

    with pytest.raises(EnvironmentSetupError, match="pre-existing.*build") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "sparse_extensions"
    assert user_file.read_text(encoding="utf-8") == "keep\n"
    assert not any(phase == "sparse_build" for _, _, phase in runner.calls)


def test_setup_sparse_cleanup_preserves_replaced_build_directory(tmp_path: Path):
    """build 중 root inode가 교체되면 replacement를 삭제하지 않고 실패한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    build = (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
        / "build"
    )
    marker = build / "concurrent-owner.txt"

    def replace_build_tree(_command: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
        shutil.rmtree(build)
        build.mkdir()
        marker.write_text("preserve\n", encoding="utf-8")

    runner.phase_hooks["sparse_build"] = replace_build_tree

    with pytest.raises(EnvironmentSetupError, match="changed|replaced") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "sparse_extensions"
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert not environment_paths(root, "sparsedrive_v2").current.exists()


def test_setup_sparse_cleanup_does_not_follow_replacement_after_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """ownership check 뒤 교체된 nonempty build root는 dirfd cleanup이 삭제하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    build = (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
        / "build"
    )
    moved = build.with_name("owned-build-moved")
    marker = build / "concurrent-owner.txt"

    def replace_before_fd_cleanup(_descriptor: int) -> None:
        build.rename(moved)
        build.mkdir()
        marker.write_text("preserve\n", encoding="utf-8")

    monkeypatch.setattr(
        environment_setup,
        "_clear_sparse_build_tree_fd",
        replace_before_fd_cleanup,
        raising=False,
    )

    with pytest.raises(EnvironmentSetupError, match="changed|replaced") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "sparse_extensions"
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert moved.is_dir()
    assert not environment_paths(root, "sparsedrive_v2").current.exists()


def test_setup_sparse_cleanup_preserves_leaf_replaced_before_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """lstat 뒤 교체된 nested regular file은 cleanup이 삭제하지 않고 실패한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path,
        model_id="sparsedrive_v2",
        strip_asset_hashes=True,
    )
    build = (
        root
        / "models"
        / "sources"
        / "sparsedrive_v2"
        / "navsim"
        / "agents"
        / "sparsedrive"
        / "ops"
        / "build"
    )
    generated = build / "temp.linux-x86_64-cpython-39" / "generated.o"
    moved = generated.with_name("owned-generated.o")
    marker = b"concurrent-user-file"
    real_rename = os.rename
    replaced = False

    def replace_before_quarantine(
        source: str | bytes,
        destination: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replaced
        if source == "generated.o" and src_dir_fd is not None and not replaced:
            replaced = True
            real_rename(generated, moved)
            generated.write_bytes(marker)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(environment_setup.os, "rename", replace_before_quarantine)

    with pytest.raises(EnvironmentSetupError, match="changed|replaced") as caught:
        setup_environment(
            root,
            "sparsedrive_v2",
            runner=runner,
            dependencies=dependencies,
        )

    preserved = tuple(
        path
        for path in build.rglob("*")
        if path.is_file() and path.read_bytes() == marker
    )
    assert caught.value.action == "sparse_extensions"
    assert preserved


def test_setup_reuse_revalidates_assets_after_runtime_checks(tmp_path: Path):
    """reuse deep check 뒤 발생한 same-metadata 변조도 반환 직전에 탐지한다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    first = setup_environment(root, "tfv6", runner=runner, dependencies=dependencies)
    runtime_asset = root / first.state.assets[0].path
    reuse_runner = _SetupRunner(root, "tfv6", first.state.python_version)
    reuse_runner.source_commit = first.state.source_commit
    reuse_runner.phase_hooks["model_load"] = lambda _command, _kwargs: (
        _mutate_file_preserving_size_and_mtime(runtime_asset)
    )

    with pytest.raises(EnvironmentSetupError, match="changed during verification") as caught:
        setup_environment(
            root,
            "tfv6",
            runner=reuse_runner,
            dependencies=dependencies,
        )

    assert caught.value.action == "reuse_verification"
    assert environment_paths(root, "tfv6").current.resolve() == first.environment_dir


def _completed_setup_for_maintenance(
    tmp_path: Path,
) -> tuple[
    Path,
    EnvironmentSetupDependencies,
    _SetupRunner,
    EnvironmentOperationResult,
]:
    """verify/run 유지보수 테스트용 정상 active environment를 만든다."""
    root, dependencies, setup_runner, _ = _setup_fixture(tmp_path)
    completed = setup_environment(
        root, "tfv6", runner=setup_runner, dependencies=dependencies
    )
    runner = _SetupRunner(root, "tfv6", completed.state.python_version)
    runner.source_commit = completed.state.source_commit
    return root, dependencies, runner, completed


def test_verify_refreshes_only_asset_metadata_and_preserves_completion(
    tmp_path: Path,
):
    """본문 SHA가 같으면 size/mtime state만 원자 갱신하고 완료 시각을 보존한다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)
    asset = root / completed.state.assets[0].path
    before = asset.stat()
    changed_mtime = before.st_mtime_ns + 10_000_000
    os.utime(asset, ns=(before.st_atime_ns, changed_mtime))

    result = environment_setup.verify_environment(
        root, "tfv6", runner=runner, dependencies=dependencies
    )

    assert result.changed is True
    assert result.environment_dir == completed.environment_dir
    assert result.state.completed_at == completed.state.completed_at
    assert result.state.assets[0].sha256 == completed.state.assets[0].sha256
    assert result.state.assets[0].mtime_ns == changed_mtime
    assert read_environment_state(completed.environment_dir) == result.state
    assert environment_paths(root, "tfv6").current.resolve() == completed.environment_dir
    phases = [phase for _, _, phase in runner.calls]
    assert "catkin_build" not in phases
    assert "uv_venv" not in phases
    assert "uv_sync" not in phases
    assert "sparse_build" not in phases


def test_verify_rejects_same_metadata_content_change_without_rewriting_state(
    tmp_path: Path,
):
    """size/mtime를 보존한 본문 변조도 deep SHA로 거부하고 state를 보존한다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)
    state_path = completed.environment_dir / ENVIRONMENT_STATE_FILENAME
    before_state = state_path.read_bytes()
    _mutate_file_preserving_size_and_mtime(root / completed.state.assets[0].path)

    with pytest.raises(EnvironmentSetupError, match="fingerprint|SHA"):
        environment_setup.verify_environment(
            root, "tfv6", runner=runner, dependencies=dependencies
        )

    assert state_path.read_bytes() == before_state
    assert environment_paths(root, "tfv6").current.resolve() == completed.environment_dir


def test_verify_and_run_preflight_reject_truncated_asset_state(
    tmp_path: Path,
):
    """fast inspector가 허용하는 빈 asset 배열도 profile exact 계약에서는 실패한다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)
    truncated = dataclasses.replace(completed.state, assets=())
    environment_setup.write_environment_state(
        completed.environment_dir / ENVIRONMENT_STATE_FILENAME, truncated
    )

    for operation in (
        environment_setup.verify_environment,
        environment_setup.preflight_run_environment,
    ):
        with pytest.raises(EnvironmentSetupError, match="asset|runtime"):
            operation(root, "tfv6", runner=runner, dependencies=dependencies)


def test_verify_unchanged_environment_runs_full_checks_without_setup_mutators(
    tmp_path: Path,
):
    """변경 없는 verify도 source/overlay/runtime/model/freeze를 검증하고 설치하지 않는다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)

    result = environment_setup.verify_environment(
        root, "tfv6", runner=runner, dependencies=dependencies
    )

    assert result.changed is False
    assert result.state == completed.state
    phases = [phase for _, _, phase in runner.calls]
    for required in (
        "source_head",
        "source_status",
        "python_version",
        "required_imports",
        "cuda_runtime",
        "model_load",
        "freeze",
    ):
        assert required in phases
    for forbidden in ("catkin_build", "uv_venv", "uv_sync", "sparse_build"):
        assert forbidden not in phases


def test_prune_environment_builds_uses_only_validated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """public prune은 profile/host/ROS/uv 없이 active를 보존하고 inactive만 지운다."""
    root = tmp_path.resolve()
    paths = environment_paths(root, "interfuser")
    active = create_candidate(paths, "active")
    inactive = create_candidate(paths, "inactive")
    paths.current.symlink_to(Path("builds/active"))
    monkeypatch.setattr(
        environment_setup,
        "preflight_environment",
        lambda *_args, **_kwargs: pytest.fail("prune must not run host preflight"),
    )

    removed = environment_setup.prune_environment_builds(root, "interfuser")

    assert removed == (inactive,)
    assert active.is_dir()
    assert paths.current.resolve() == active


def test_run_preflight_is_read_only_and_executes_current_runtime_checks(
    tmp_path: Path,
):
    """run 점검은 실제 source/Python/import/CUDA를 확인하되 deep/setup 단계를 생략한다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run preflight invoked a deep or mutating dependency")

    run_dependencies = dataclasses.replace(
        dependencies,
        asset_fetcher=forbidden,
        asset_verifier=forbidden,
        build_id=forbidden,
        utc_now=forbidden,
        ros_smoke=forbidden,
    )
    original_fingerprint = environment_setup.fingerprint_runtime_assets
    state_path = completed.environment_dir / ENVIRONMENT_STATE_FILENAME
    state_before = state_path.read_bytes()
    environment_setup.fingerprint_runtime_assets = forbidden
    try:
        result = environment_setup.preflight_run_environment(
            root, "tfv6", runner=runner, dependencies=run_dependencies
        )
    finally:
        environment_setup.fingerprint_runtime_assets = original_fingerprint

    assert result == EnvironmentOperationResult(
        environment_dir=completed.environment_dir,
        state=completed.state,
        warnings=(),
        changed=False,
    )
    assert state_path.read_bytes() == state_before
    phases = [phase for _, _, phase in runner.calls]
    for required in (
        "host_uv",
        "catkin_messages",
        "source_head",
        "source_status",
        "python_version",
        "target_ros_import",
        "required_imports",
        "cuda_runtime",
    ):
        assert required in phases
    for forbidden_phase in (
        "catkin_build",
        "uv_venv",
        "uv_sync",
        "sparse_build",
        "model_load",
        "freeze",
    ):
        assert forbidden_phase not in phases
    for _command, kwargs, phase in runner.calls:
        if phase in {"source_head", "source_status"}:
            assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        if phase in {
            "catkin_messages",
            "python_version",
            "target_ros_import",
            "required_imports",
            "cuda_runtime",
        }:
            assert "-B" in _command


def test_run_preflight_missing_model_root_does_not_create_it(tmp_path: Path):
    """run 점검은 누락 환경을 실패시키되 .venvs/model 또는 lock을 생성하지 않는다."""
    root, dependencies, runner, _ = _setup_fixture(tmp_path)
    model_root = environment_paths(root, "tfv6").model_root

    with pytest.raises(EnvironmentSetupError, match="model root") as caught:
        environment_setup.preflight_run_environment(
            root, "tfv6", runner=runner, dependencies=dependencies
        )

    assert caught.value.action == "runtime_preflight"
    assert not model_root.exists()
    assert not model_root.is_symlink()


def test_run_preflight_uses_read_only_shared_lifecycle_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """run은 기존 lock을 shared/read-only로 열고 metadata를 바꾸지 않는다."""
    root, dependencies, runner, _completed = _completed_setup_for_maintenance(
        tmp_path
    )
    lock_path = environment_paths(root, "tfv6").model_root / ".lifecycle.lock"
    before = lock_path.stat()
    calls: list[int] = []
    real_flock = environment_setup.fcntl.flock

    def recording_flock(file_descriptor: int, operation: int) -> None:
        calls.append(operation)
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(environment_setup.fcntl, "flock", recording_flock)
    time.sleep(0.01)

    environment_setup.preflight_run_environment(
        root, "tfv6", runner=runner, dependencies=dependencies
    )

    after = lock_path.stat()
    assert calls == [environment_setup.fcntl.LOCK_SH, environment_setup.fcntl.LOCK_UN]
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    assert after.st_ctime_ns == before.st_ctime_ns


@pytest.mark.parametrize("lock_state", ("missing", "permissive", "symlink"))
def test_run_preflight_rejects_stale_lifecycle_lock_without_repair(
    tmp_path: Path, lock_state: str
):
    """run은 누락/비정상 lock을 만들거나 고치지 않고 stale로 실패시킨다."""
    root, dependencies, runner, _completed = _completed_setup_for_maintenance(
        tmp_path
    )
    lock_path = environment_paths(root, "tfv6").model_root / ".lifecycle.lock"
    outside = tmp_path / "outside-lock"
    if lock_state == "missing":
        lock_path.unlink()
    elif lock_state == "permissive":
        lock_path.chmod(0o644)
    else:
        lock_path.unlink()
        outside.write_text("keep\n", encoding="utf-8")
        lock_path.symlink_to(outside)

    with pytest.raises(EnvironmentSetupError, match="stale") as caught:
        environment_setup.preflight_run_environment(
            root, "tfv6", runner=runner, dependencies=dependencies
        )

    assert caught.value.action == "runtime_preflight"
    if lock_state == "missing":
        assert not lock_path.exists()
        assert not lock_path.is_symlink()
    elif lock_state == "permissive":
        assert lock_path.stat().st_mode & 0o777 == 0o644
    else:
        assert lock_path.is_symlink()
        assert outside.read_text(encoding="utf-8") == "keep\n"


def test_run_preflight_holds_model_lock_against_current_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """inspect부터 runtime probe까지 current 교체가 같은 lifecycle lock에서 대기한다."""
    root, dependencies, runner, completed = _completed_setup_for_maintenance(tmp_path)
    paths = environment_paths(root, "tfv6")
    replacement = create_candidate(paths, "replacement")
    probe_entered = threading.Event()
    release_probe = threading.Event()
    switch_finished = threading.Event()
    errors: list[BaseException] = []
    results: list[EnvironmentOperationResult] = []
    real_probe = environment_setup._run_candidate_import_cuda_checks

    def blocking_probe(*args: Any, **kwargs: Any) -> str | None:
        probe_entered.set()
        if not release_probe.wait(5.0):
            raise RuntimeError("timed out waiting to release runtime preflight")
        return real_probe(*args, **kwargs)

    def run_preflight() -> None:
        try:
            results.append(
                environment_setup.preflight_run_environment(
                    root, "tfv6", runner=runner, dependencies=dependencies
                )
            )
        except BaseException as exc:
            errors.append(exc)

    def switch_current() -> None:
        try:
            activate_candidate(paths, replacement)
        except BaseException as exc:
            errors.append(exc)
        finally:
            switch_finished.set()

    monkeypatch.setattr(
        environment_setup, "_run_candidate_import_cuda_checks", blocking_probe
    )
    preflight_thread = threading.Thread(target=run_preflight)
    preflight_thread.start()
    assert probe_entered.wait(5.0)
    switch_thread = threading.Thread(target=switch_current)
    switch_thread.start()
    assert not switch_finished.wait(0.2)
    assert paths.current.resolve() == completed.environment_dir
    release_probe.set()
    preflight_thread.join(5.0)
    switch_thread.join(5.0)

    assert not preflight_thread.is_alive()
    assert not switch_thread.is_alive()
    assert errors == []
    assert results[0].environment_dir == completed.environment_dir
    assert paths.current.resolve() == replacement


def test_interfuser_runtime_probe_prepends_and_attests_vendored_timm(
    tmp_path: Path,
):
    """InterFuser는 checkout/interfuser를 먼저 넣고 그 아래 timm origin만 허용한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, model_id="interfuser"
    )
    preflight = preflight_environment(
        root,
        "interfuser",
        runner=runner,
        dependencies=dependencies.preflight_dependencies,
    )
    checkout = root / "models" / "sources" / "interfuser"
    (checkout / "interfuser" / "timm").mkdir(parents=True)
    (checkout / "interfuser" / "timm" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    candidate = create_candidate(preflight.paths, "probe")
    _write_executable(candidate / "bin" / "python")
    runner.calls.clear()

    environment_setup._run_candidate_import_cuda_checks(
        preflight, candidate, {}, runner
    )

    command, _, _ = _setup_call(runner, "required_imports")
    assert command[:3] == (str(candidate / "bin" / "python"), "-B", "-c")
    assert "timm" in command[3]
    assert command[-1] == str(checkout / "interfuser")

    names = preflight.definition.required_imports
    runner.required_import_origins_override = [
        str(candidate / "lib" / "python" / "site-packages" / name / "__init__.py")
        for name in names
    ]
    with pytest.raises(EnvironmentSetupError, match="timm origin") as caught:
        environment_setup._run_candidate_import_cuda_checks(
            preflight, candidate, {}, runner
        )
    assert caught.value.action == "required_imports"

    outside = tmp_path / "outside-timm.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    escaped = checkout / "interfuser" / "timm" / "escaped.py"
    escaped.symlink_to(outside)
    runner.required_import_origins_override = [
        str(
            escaped
            if name == "timm"
            else candidate
            / "lib"
            / "python"
            / "site-packages"
            / name
            / "__init__.py"
        )
        for name in names
    ]
    with pytest.raises(EnvironmentSetupError, match="timm origin"):
        environment_setup._run_candidate_import_cuda_checks(
            preflight, candidate, {}, runner
        )


def test_sparsedrive_runtime_probe_rejects_foreign_extension_origin(tmp_path: Path):
    """SparseDrive required import가 attested source pair 밖 native module이면 실패한다."""
    root, dependencies, runner, _ = _setup_fixture(
        tmp_path, model_id="sparsedrive_v2", strip_asset_hashes=True
    )
    preflight = preflight_environment(
        root,
        "sparsedrive_v2",
        runner=runner,
        dependencies=dependencies.preflight_dependencies,
    )
    checkout = root / "models" / "sources" / "sparsedrive_v2"
    ops = checkout / "navsim" / "agents" / "sparsedrive" / "ops"
    ops.mkdir(parents=True)
    for module in (
        "deformable_aggregation_ext",
        "deformable_aggregation_with_depth_ext",
    ):
        (ops / f"{module}.cpython-39-x86_64-linux-gnu.so").write_bytes(b"native")
    candidate = create_candidate(preflight.paths, "probe")
    _write_executable(candidate / "bin" / "python")
    names = preflight.definition.required_imports
    runner.required_import_origins_override = [
        str(candidate / "lib" / "python" / "site-packages" / name / "__init__.py")
        for name in names
    ]

    with pytest.raises(EnvironmentSetupError, match="extension origin") as caught:
        environment_setup._run_candidate_import_cuda_checks(
            preflight, candidate, {}, runner
        )

    assert caught.value.action == "required_imports"
