"""모델 환경 candidate와 순수 Python ROS overlay 보안 계약을 검사한다."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable

import pytest

from validator_inference import environment_setup
from validator_inference.environment_setup import (
    PURE_ROS_PACKAGES,
    EnvironmentSetupError,
    RosOverlayFile,
    RosOverlayManifest,
    activate_candidate,
    build_ros_overlay,
    create_candidate,
    prune_inactive_builds,
    verify_ros_overlay,
)
from validator_inference.model_environment import (
    ROS_OVERLAY_MANIFEST_FILENAME,
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
