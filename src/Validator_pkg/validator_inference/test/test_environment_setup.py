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
    EnvironmentPreflight,
    EnvironmentPreflightDependencies,
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
    verify_ros_overlay,
)
from validator_inference.model_environment import (
    ROS_OVERLAY_MANIFEST_FILENAME,
    AssetFingerprint,
    EnvironmentPaths,
    environment_paths,
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
