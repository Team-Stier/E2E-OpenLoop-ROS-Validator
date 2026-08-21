"""모델별 interpreter launch-prefix의 격리 및 exec 계약을 검사한다."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Mapping, Optional, Sequence, Union

import pytest

from validator_inference.environment_setup import (
    PURE_ROS_PACKAGES,
    activate_candidate,
    build_ros_overlay,
    prune_inactive_builds,
)
from validator_inference.model_environment import (
    ENVIRONMENT_STATE_FILENAME,
    EXPECTED_PYTHON_PLATFORM,
    EXPECTED_UV_VERSION,
    ROS_OVERLAY_MANIFEST_FILENAME,
    EnvironmentState,
    environment_paths,
    load_environment_manifest,
    select_model_environment,
    sha256_file,
    write_environment_state,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PACKAGE_ROOT / "scripts" / "model_env_exec"
MODEL_ID = "interfuser"
CAPTURE_ENVIRONMENT_KEYS = (
    "CONDA_PREFIX",
    "OPENLOOP_MODEL_ENV_ROOT",
    "OPENLOOP_MODEL_ID",
    "OPENLOOP_PROJECT_ROOT",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "ROS_MASTER_URI",
    "TORCH_HOME",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True)
class _WrapperFixture:
    """한 subprocess wrapper 실행에 필요한 가짜 Catkin 및 active build다."""

    wrapper: Path
    package_root: Path
    project_root: Path
    model_env_root: Path
    catkin_prefix: Path
    catkin_python_path: Path
    relay: Path
    environment_dir: Path
    overlay: Path
    overlay_digest: str
    capture: Path
    copied_environment_setup: Path

    def run(
        self,
        remappings: Sequence[str] = (),
        *,
        overrides: Optional[Mapping[str, Optional[str]]] = None,
        relay: Optional[Union[str, Path]] = None,
        child_exit: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        """격리 경계에 오염된 parent 환경과 ROS remapping argv를 전달한다."""
        environment = dict(os.environ)
        foreign = self.project_root / "foreign environment"
        conda = self.project_root / "conda environment"
        environment.update(
            {
                "CAPTURE_PATH": str(self.capture),
                "CHILD_EXIT": str(child_exit),
                "CONDA_PREFIX": str(conda),
                "OPENLOOP_MODEL_ENV_ROOT": str(self.model_env_root),
                "OPENLOOP_MODEL_ID": MODEL_ID,
                "OPENLOOP_PROJECT_ROOT": str(self.project_root),
                "PATH": os.pathsep.join(
                    (str(foreign / "bin"), str(conda / "bin"), "/usr/bin", "/bin")
                ),
                "PYTHONHOME": "/usr",
                "PYTHONPATH": "/untrusted/python",
                "PYTHONUSERBASE": "/untrusted/userbase",
                "ROS_MASTER_URI": "http://127.0.0.1:11311",
                "VIRTUAL_ENV": str(foreign),
            }
        )
        if overrides:
            for name, value in overrides.items():
                if value is None:
                    environment.pop(name, None)
                else:
                    environment[name] = value
        self.capture.unlink(missing_ok=True)
        selected_relay = str(self.relay if relay is None else relay)
        return subprocess.run(
            [str(self.wrapper), selected_relay, *remappings],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def _copy_runtime_modules(catkin_python_path: Path) -> Path:
    """가짜 Catkin prefix가 실제 환경 검증 모듈만 제공하게 구성한다."""
    source_package = PACKAGE_ROOT / "src" / "validator_inference"
    target_package = catkin_python_path / "validator_inference"
    target_package.mkdir(parents=True)
    for name in ("__init__.py", "model_environment.py", "environment_setup.py"):
        shutil.copy2(source_package / name, target_package / name)

    messages = catkin_python_path / "validator_msgs" / "msg"
    messages.mkdir(parents=True)
    (messages.parent / "__init__.py").write_text("", encoding="utf-8")
    (messages / "__init__.py").write_text("", encoding="utf-8")
    return target_package / "environment_setup.py"


def _install_package_copy(prefix: Path) -> tuple[Path, Path]:
    """install-space share layout에 wrapper와 manifest 자료를 복사한다."""
    assert WRAPPER.is_file(), f"missing wrapper: {WRAPPER}"
    package_root = prefix / "share" / "validator_inference"
    scripts = package_root / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / WRAPPER.name
    shutil.copy2(WRAPPER, wrapper)
    shutil.copytree(PACKAGE_ROOT / "config", package_root / "config")
    shutil.copytree(PACKAGE_ROOT / "requirements", package_root / "requirements")
    return wrapper, package_root


def _write_capture_python(path: Path) -> None:
    """execve 인자와 정화된 child 환경을 JSON으로 기록하는 가짜 Python을 만든다."""
    keys = repr(CAPTURE_ENVIRONMENT_KEYS)
    path.parent.mkdir(parents=True)
    path.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        f"keys = {keys}\n"
        "payload = {\n"
        "    'argv': sys.argv,\n"
        "    'environment': {key: os.environ[key] for key in keys if key in os.environ},\n"
        "}\n"
        "with open(os.environ['CAPTURE_PATH'], 'w', encoding='utf-8') as target:\n"
        "    json.dump(payload, target, sort_keys=True)\n"
        "if os.environ.get('CHILD_STARTED_PATH'):\n"
        "    from pathlib import Path\n"
        "    import time\n"
        "    started = Path(os.environ['CHILD_STARTED_PATH'])\n"
        "    release = Path(os.environ['CHILD_RELEASE_PATH'])\n"
        "    started.write_text('started\\n', encoding='utf-8')\n"
        "    while not release.exists():\n"
        "        time.sleep(0.01)\n"
        "raise SystemExit(int(os.environ.get('CHILD_EXIT', '0')))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_ros_source(root: Path) -> Path:
    """exact allowlist의 최소 pure-Python ROS source tree를 만든다."""
    source = root / "ros source"
    for package in PURE_ROS_PACKAGES:
        package_root = source / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            f"PACKAGE = {package!r}\n", encoding="utf-8"
        )
    return source


def _wrapper_fixture(tmp_path: Path, layout: str = "devel") -> _WrapperFixture:
    """실제 manifest/state/overlay를 가진 source 또는 install 실행 fixture를 만든다."""
    assert WRAPPER.is_file(), f"missing wrapper: {WRAPPER}"
    project_root = (tmp_path / "project root with spaces").resolve()
    project_root.mkdir()
    catkin_prefix = (tmp_path / f"{layout} prefix with spaces").resolve()
    if layout == "install":
        wrapper, package_root = _install_package_copy(catkin_prefix)
    else:
        wrapper, package_root = WRAPPER, PACKAGE_ROOT

    catkin_python_path = catkin_prefix / "lib" / "python3" / "dist-packages"
    copied_environment_setup = _copy_runtime_modules(catkin_python_path)
    relay = catkin_prefix / "lib" / "validator_inference" / "inference_engine_node.py"
    relay.parent.mkdir(parents=True)
    relay.write_text(
        "#!/usr/bin/python3\n"
        "python_script = '/outside/source/inference_engine_node.py'\n",
        encoding="utf-8",
    )
    relay.chmod(0o755)

    manifest_path = package_root / "config" / "model_environments.json"
    manifest = load_environment_manifest(manifest_path)
    definition = select_model_environment(manifest, MODEL_ID)
    model_env_root = project_root / ".venvs"
    environment_dir = model_env_root / MODEL_ID / "builds" / "build-001"
    environment_dir.mkdir(parents=True)
    python = environment_dir / "bin" / "python"
    _write_capture_python(python)
    overlay = environment_dir / ".ros-overlay"
    build_ros_overlay(_fake_ros_source(tmp_path), overlay)
    overlay_manifest = environment_dir / ROS_OVERLAY_MANIFEST_FILENAME
    overlay_digest = sha256_file(overlay_manifest)
    state = EnvironmentState(
        schema=1,
        model_id=MODEL_ID,
        python_version=definition.python_version,
        python_platform=EXPECTED_PYTHON_PLATFORM,
        uv_version=EXPECTED_UV_VERSION,
        manifest_sha256=sha256_file(manifest_path),
        lock_sha256=definition.requirements_lock_sha256,
        ros_overlay_sha256=overlay_digest,
        source_commit=definition.source_commit,
        installed_packages_sha256="a" * 64,
        python_executable=Path("bin") / "python",
        assets=(),
        completed_at="2026-07-21T12:34:56Z",
        preflight_ids=(),
    )
    write_environment_state(environment_dir / ENVIRONMENT_STATE_FILENAME, state)
    current = model_env_root / MODEL_ID / "current"
    current.symlink_to(Path("builds") / environment_dir.name)
    lifecycle_lock = current.parent / ".lifecycle.lock"
    lifecycle_lock.write_text("", encoding="utf-8")
    lifecycle_lock.chmod(0o600)
    return _WrapperFixture(
        wrapper=wrapper,
        package_root=package_root,
        project_root=project_root,
        model_env_root=model_env_root,
        catkin_prefix=catkin_prefix,
        catkin_python_path=catkin_python_path,
        relay=relay,
        environment_dir=environment_dir,
        overlay=overlay,
        overlay_digest=overlay_digest,
        capture=tmp_path / f"{layout}-capture.json",
        copied_environment_setup=copied_environment_setup,
    )


def _assert_one_line_error(result: subprocess.CompletedProcess[str]) -> None:
    """wrapper 실패가 traceback 없는 단일 stderr 진단인지 확인한다."""
    assert result.returncode != 0
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    assert "model_env_exec:" in result.stderr
    assert "Traceback" not in result.stderr


def test_wrapper_has_exact_system_python_shebang_and_is_executable():
    """wrapper가 Catkin rewriting 없이 system Python으로 직접 시작된다."""
    assert WRAPPER.is_file()
    assert WRAPPER.read_bytes().splitlines()[0] == b"#!/usr/bin/python3"
    assert os.access(str(WRAPPER), os.X_OK)


@pytest.mark.parametrize("layout", ("devel", "install"))
def test_wrapper_exec_preserves_remappings_exit_status_and_sanitizes_environment(
    tmp_path: Path, layout: str
):
    """신뢰된 controller에서 argv를 보존하고 target child 환경만 정화한다."""
    fixture = _wrapper_fixture(tmp_path, layout)
    remappings = (
        "__name:=inference engine",
        "__ns:=/validator namespace",
        "camera/image:=/recorded camera/image raw",
        "_private:=value=with=equals",
    )

    result = fixture.run(remappings, child_exit=23)

    assert result.returncode == 23, result.stderr
    assert result.stderr == ""
    payload = json.loads(fixture.capture.read_text(encoding="utf-8"))
    python = fixture.environment_dir / "bin" / "python"
    assert payload["argv"] == [str(python), str(fixture.relay), *remappings]
    environment = payload["environment"]
    assert environment["PYTHONPATH"] == os.pathsep.join(
        (str(fixture.overlay), str(fixture.catkin_python_path))
    )
    assert environment["PATH"] == os.pathsep.join(
        (str(fixture.environment_dir / "bin"), "/usr/bin")
    )
    assert environment["VIRTUAL_ENV"] == str(fixture.environment_dir)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["TORCH_HOME"] == str(
        fixture.project_root / "models" / "shared" / "torch-cache"
    )
    assert environment["ROS_MASTER_URI"] == "http://127.0.0.1:11311"
    assert environment["OPENLOOP_MODEL_ENV_ROOT"] == str(fixture.model_env_root)
    assert environment["OPENLOOP_MODEL_ID"] == MODEL_ID
    assert environment["OPENLOOP_PROJECT_ROOT"] == str(fixture.project_root)
    assert "PYTHONHOME" not in environment
    assert "PYTHONUSERBASE" not in environment
    assert "CONDA_PREFIX" not in environment


def test_wrapper_rejects_path_and_model_injection_before_exec(tmp_path: Path):
    """relative/noncanonical 경로와 separator를 포함한 model ID를 거부한다."""
    fixture = _wrapper_fixture(tmp_path)
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    wrong_node = fixture.relay.with_name("recorder_node.py")
    shutil.copy2(fixture.relay, wrong_node)
    noncanonical_relay = (
        str(fixture.catkin_prefix)
        + "/lib/validator_inference/../validator_inference/inference_engine_node.py"
    )
    cases = (
        ({"OPENLOOP_MODEL_ENV_ROOT": "relative/.venvs"}, None),
        ({"OPENLOOP_MODEL_ENV_ROOT": str(outside)}, None),
        ({"OPENLOOP_PROJECT_ROOT": str(fixture.project_root) + "/."}, None),
        ({"OPENLOOP_MODEL_ID": "../interfuser"}, None),
        ({"OPENLOOP_MODEL_ID": "interfuser/other"}, None),
        ({"OPENLOOP_MODEL_ID": "unknown"}, None),
        ({}, "relative/inference_engine_node.py"),
        ({}, noncanonical_relay),
        ({}, wrong_node),
    )

    for overrides, relay in cases:
        result = fixture.run(overrides=overrides, relay=relay)
        _assert_one_line_error(result)
        assert not fixture.capture.exists()


def test_wrapper_requires_regular_catkin_package_markers(tmp_path: Path):
    """파생 Catkin Python path의 inference/message package 표식을 검증한다."""
    fixture = _wrapper_fixture(tmp_path)
    marker = fixture.catkin_python_path / "validator_msgs" / "msg" / "__init__.py"
    marker.unlink()

    result = fixture.run()

    _assert_one_line_error(result)
    assert "validator_msgs" in result.stderr


@pytest.mark.parametrize(
    "target", ("relay-package", "inference-package", "message-package", "config")
)
def test_wrapper_rejects_symlinked_intermediate_runtime_directories(
    tmp_path: Path, target: str
):
    """relay/import/config marker의 중간 directory symlink를 따라가지 않는다."""
    fixture = _wrapper_fixture(tmp_path, "install")
    if target == "relay-package":
        directory = fixture.relay.parent
    elif target == "inference-package":
        directory = fixture.catkin_python_path / "validator_inference"
    elif target == "message-package":
        directory = fixture.catkin_python_path / "validator_msgs" / "msg"
    else:
        directory = fixture.package_root / "config"
    relocated = directory.with_name(directory.name + "-real")
    directory.rename(relocated)
    directory.symlink_to(relocated.name, target_is_directory=True)

    result = fixture.run()

    _assert_one_line_error(result)
    assert not fixture.capture.exists()


def test_wrapper_reports_stale_state_and_overlay_without_traceback(tmp_path: Path):
    """빠른 state 검사와 deep overlay 검사의 실패를 모두 launch 오류로 정규화한다."""
    fixture = _wrapper_fixture(tmp_path)
    state_path = fixture.environment_dir / ENVIRONMENT_STATE_FILENAME
    original = state_path.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["manifest_sha256"] = "0" * 64
    state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stale = fixture.run()

    _assert_one_line_error(stale)
    assert "stale" in stale.stderr

    state_path.write_text(original, encoding="utf-8")
    (fixture.overlay / "rospy" / "__init__.py").write_text(
        "PACKAGE = 'changed'\n", encoding="utf-8"
    )
    modified = fixture.run()

    _assert_one_line_error(modified)
    assert "hash" in modified.stderr


def test_wrapper_uses_inspected_build_snapshot_when_current_switches(tmp_path: Path):
    """inspect 이후 current가 바뀌어도 검증한 build interpreter만 exec한다."""
    fixture = _wrapper_fixture(tmp_path)
    current = fixture.model_env_root / MODEL_ID / "current"
    replacement = current.parent / "builds" / "build-002"
    replacement.mkdir()
    fixture.copied_environment_setup.write_text(
        "from contextlib import contextmanager\n"
        "import os\n"
        "from pathlib import Path\n"
        "class EnvironmentSetupError(RuntimeError):\n"
        "    pass\n"
        "@contextmanager\n"
        "def model_runtime_lease(paths, action, inheritable=False):\n"
        "    yield\n"
        "def verify_ros_overlay(destination, manifest_path):\n"
        "    expected = Path(os.environ['EXPECTED_OVERLAY'])\n"
        "    if Path(destination) != expected:\n"
        "        raise EnvironmentSetupError('wrong inspected overlay')\n"
        "    current = Path(os.environ['CURRENT_LINK'])\n"
        "    temporary = current.with_name('.current-test-switch')\n"
        "    temporary.symlink_to(Path('builds') / 'build-002')\n"
        "    os.replace(str(temporary), str(current))\n"
        "    return os.environ['OVERLAY_DIGEST']\n",
        encoding="utf-8",
    )

    result = fixture.run(
        overrides={
            "CURRENT_LINK": str(current),
            "EXPECTED_OVERLAY": str(fixture.overlay),
            "OVERLAY_DIGEST": fixture.overlay_digest,
        }
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(fixture.capture.read_text(encoding="utf-8"))
    assert payload["environment"]["VIRTUAL_ENV"] == str(fixture.environment_dir)
    assert current.resolve(strict=True) == replacement


def test_wrapper_revalidates_after_previous_build_is_replaced_and_pruned(
    tmp_path: Path,
):
    """CLI preflight 뒤 교체가 끝났다면 wrapper는 새 current를 다시 검사한다."""
    fixture = _wrapper_fixture(tmp_path)
    paths = environment_paths(fixture.project_root, MODEL_ID)
    replacement = shutil.copytree(
        fixture.environment_dir, paths.builds / "build-002"
    )

    activate_candidate(paths, replacement)
    assert prune_inactive_builds(paths) == (fixture.environment_dir,)

    result = fixture.run()

    assert result.returncode == 0, result.stderr
    payload = json.loads(fixture.capture.read_text(encoding="utf-8"))
    assert payload["argv"][0] == str(replacement / "bin" / "python")
    assert payload["environment"]["VIRTUAL_ENV"] == str(replacement)


def test_wrapper_missing_lifecycle_lock_is_stale_and_not_created(tmp_path: Path):
    """wrapper도 missing lock을 고치지 않고 setup이 필요한 stale 상태로 거부한다."""
    fixture = _wrapper_fixture(tmp_path)
    lock_path = fixture.model_env_root / MODEL_ID / ".lifecycle.lock"
    lock_path.unlink()

    result = fixture.run()

    _assert_one_line_error(result)
    assert "stale" in result.stderr
    assert not lock_path.exists()
    assert not lock_path.is_symlink()


def test_wrapper_holds_shared_lease_through_exec_against_switch_and_prune(
    tmp_path: Path,
):
    """target process 수명 동안 active 교체와 old build prune가 대기한다."""
    fixture = _wrapper_fixture(tmp_path)
    paths = environment_paths(fixture.project_root, MODEL_ID)
    replacement = paths.builds / "build-002"
    replacement.mkdir()
    started = tmp_path / "target-started"
    release = tmp_path / "release-target"
    lifecycle_finished = threading.Event()
    results: list[subprocess.CompletedProcess[str]] = []
    removed: list[tuple[Path, ...]] = []
    errors: list[BaseException] = []

    def run_wrapper() -> None:
        try:
            results.append(
                fixture.run(
                    overrides={
                        "CHILD_STARTED_PATH": str(started),
                        "CHILD_RELEASE_PATH": str(release),
                    }
                )
            )
        except BaseException as exc:
            errors.append(exc)

    def switch_and_prune() -> None:
        try:
            activate_candidate(paths, replacement)
            removed.append(prune_inactive_builds(paths))
        except BaseException as exc:
            errors.append(exc)
        finally:
            lifecycle_finished.set()

    wrapper_thread = threading.Thread(target=run_wrapper)
    lifecycle_thread = threading.Thread(target=switch_and_prune)
    wrapper_thread.start()
    try:
        for _attempt in range(500):
            if started.exists():
                break
            time.sleep(0.01)
        assert started.exists()
        lifecycle_thread.start()
        assert not lifecycle_finished.wait(0.2)
        assert paths.current.resolve(strict=True) == fixture.environment_dir
        assert fixture.environment_dir.is_dir()
    finally:
        release.write_text("release\n", encoding="utf-8")
        wrapper_thread.join(5.0)
        if lifecycle_thread.ident is not None:
            lifecycle_thread.join(5.0)

    assert not wrapper_thread.is_alive()
    assert not lifecycle_thread.is_alive()
    assert errors == []
    assert results[0].returncode == 0, results[0].stderr
    assert paths.current.resolve(strict=True) == replacement
    assert removed == [(fixture.environment_dir,)]
    assert not fixture.environment_dir.exists()
