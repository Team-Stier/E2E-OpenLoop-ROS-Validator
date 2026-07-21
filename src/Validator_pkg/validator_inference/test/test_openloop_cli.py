"""Repository-local openloop command contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from validator_inference.environment_setup import EnvironmentSetupError
from validator_inference.model_environment import SUPPORTED_MODEL_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = PROJECT_ROOT / "src/Validator_pkg/validator_inference/scripts/openloop_cli.py"
ROOT_SHIM = PROJECT_ROOT / "openloop"


@pytest.fixture
def cli() -> ModuleType:
    """Load the source CLI without requiring a Catkin install."""
    assert SCRIPT.is_file(), f"missing CLI script: {SCRIPT}"
    module_name = "_openloop_cli_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _write_profile(project_root: Path, bag_path: Any) -> Path:
    """Write only the profile structure owned by bag dispatch."""
    profile = project_root / "src/Validator_pkg/config/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps({"observed_bag_contract": {"bag_path": bag_path}}),
        encoding="utf-8",
    )
    return profile


def test_normalize_argv_accepts_only_exact_model_shorthand(cli: ModuleType):
    for model_id in SUPPORTED_MODEL_IDS:
        assert cli._normalize_argv([model_id, "--rviz"]) == [
            "run",
            model_id,
            "--rviz",
        ]

    assert cli._normalize_argv(["inter", "--rviz"]) == ["inter", "--rviz"]
    assert cli._normalize_argv(["setup", "interfuser"]) == [
        "setup",
        "interfuser",
    ]
    assert cli._normalize_argv([]) == []


@pytest.mark.parametrize(
    ("argv", "command", "model_id", "enabled"),
    [
        (["setup", "interfuser"], "setup", "interfuser", False),
        (["setup", "interfuser", "--recreate"], "setup", "interfuser", True),
        (["run", "tfv6"], "run", "tfv6", False),
        (["run", "tfv6", "--rviz"], "run", "tfv6", True),
        (["verify", "diffusiondrive_v2"], "verify", "diffusiondrive_v2", None),
        (["prune", "sparsedrive_v2"], "prune", "sparsedrive_v2", None),
        (["bag"], "bag", None, None),
    ],
)
def test_parser_exposes_all_commands(
    cli: ModuleType,
    argv: list[str],
    command: str,
    model_id: str | None,
    enabled: bool | None,
):
    args = cli._parser().parse_args(cli._normalize_argv(argv))

    assert args.command == command
    if model_id is not None:
        assert args.model_id == model_id
    if command == "setup":
        assert args.recreate is enabled
    if command == "run":
        assert args.rviz is enabled


@pytest.mark.parametrize(
    "argv",
    [
        ["inter", "--rviz"],
        ["run", "inter"],
        ["unknown"],
        ["setup", "interfuser", "--rec"],
        ["run", "interfuser", "--r"],
        ["bag", "--pa", "capture.bag"],
    ],
)
def test_parser_rejects_unknown_models_commands_and_abbreviations(
    cli: ModuleType, argv: list[str]
):
    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args(cli._normalize_argv(argv))

    assert error.value.code == 2


@pytest.mark.parametrize("rviz", [False, True])
def test_launch_argv_is_exact_array_contract(
    cli: ModuleType, tmp_path: Path, rviz: bool
):
    root = tmp_path.resolve()

    result = cli._launch_argv(root, "interfuser", rviz)

    assert isinstance(result, list)
    assert result == [
        "roslaunch",
        "Validator_pkg",
        "openloop_validator.launch",
        f"project_root:={root}",
        "model_id:=interfuser",
        f"model_env_root:={root / '.venvs'}",
        f"show_rviz:={'true' if rviz else 'false'}",
    ]


def test_bag_argv_is_exact_array_contract(cli: ModuleType, tmp_path: Path):
    bag = (tmp_path / "capture.bag").resolve()

    result = cli._bag_argv(bag)

    assert isinstance(result, list)
    assert result == ["rosbag", "play", "--clock", str(bag)]


def test_setup_delegates_to_environment_api(cli: ModuleType, monkeypatch, tmp_path):
    calls: list[tuple[Path, str, bool]] = []

    def setup(project_root: Path, model_id: str, recreate: bool = False):
        calls.append((project_root, model_id, recreate))
        return SimpleNamespace(environment_dir=tmp_path / "candidate", warnings=())

    monkeypatch.setattr(cli.environment_setup, "setup_environment", setup)

    assert cli.main(
        ["setup", "interfuser", "--recreate"], project_root=tmp_path
    ) == 0
    assert calls == [(tmp_path.resolve(), "interfuser", True)]


@pytest.mark.parametrize(
    ("command", "function_name"),
    [("verify", "verify_environment"), ("prune", "prune_environment_builds")],
)
def test_maintenance_commands_delegate_to_environment_api(
    cli: ModuleType,
    monkeypatch,
    tmp_path: Path,
    command: str,
    function_name: str,
):
    calls: list[tuple[Path, str]] = []

    def operation(project_root: Path, model_id: str):
        calls.append((project_root, model_id))
        return SimpleNamespace(environment_dir=tmp_path / "current", warnings=())

    monkeypatch.setattr(cli.environment_setup, function_name, operation, raising=False)

    assert cli.main([command, "tfv6"], project_root=tmp_path) == 0
    assert calls == [(tmp_path.resolve(), "tfv6")]


def test_setup_prints_phase_error_and_returns_nonzero(
    cli: ModuleType, monkeypatch, tmp_path: Path, capsys
):
    def fail(*_args, **_kwargs):
        raise EnvironmentSetupError("assets", "checkpoint is missing")

    monkeypatch.setattr(cli.environment_setup, "setup_environment", fail)

    assert cli.main(["setup", "interfuser"], project_root=tmp_path) == 1
    assert capsys.readouterr().err == "openloop: assets: checkpoint is missing\n"


def test_run_delegates_to_read_only_runtime_preflight_before_exec(
    cli: ModuleType, monkeypatch, tmp_path: Path
):
    calls: list[Any] = []

    def preflight(project_root: Path, model_id: str):
        calls.append(("preflight", project_root, model_id))
        return SimpleNamespace(warnings=("runtime warning",), changed=False)

    monkeypatch.setattr(
        cli.environment_setup,
        "preflight_run_environment",
        preflight,
        raising=False,
    )
    monkeypatch.setattr(
        cli.environment_setup,
        "setup_environment",
        lambda *_args, **_kwargs: pytest.fail("run must not invoke setup"),
    )

    class ExecCalled(Exception):
        pass

    def execvpe(program: str, argv: list[str], environment: dict[str, str]):
        calls.append(("exec", program))
        assert program == "roslaunch"
        assert isinstance(argv, list)
        assert argv == cli._launch_argv(tmp_path.resolve(), "interfuser", True)
        assert environment == dict(os.environ)
        raise ExecCalled

    monkeypatch.setattr(cli.os, "execvpe", execvpe)

    with pytest.raises(ExecCalled):
        cli.main(["interfuser", "--rviz"], project_root=tmp_path)

    assert calls == [
        ("preflight", tmp_path.resolve(), "interfuser"),
        ("exec", "roslaunch"),
    ]


def test_runtime_preflight_failure_prints_only_recovery_and_never_starts_ros(
    cli: ModuleType, monkeypatch, tmp_path: Path, capsys
):
    def fail(*_args, **_kwargs):
        raise EnvironmentSetupError("runtime_preflight", "required import is missing")

    monkeypatch.setattr(
        cli.environment_setup,
        "preflight_run_environment",
        fail,
        raising=False,
    )
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda *_args: pytest.fail("stale run must not start ROS"),
    )
    monkeypatch.setattr(
        cli.environment_setup,
        "setup_environment",
        lambda *_args, **_kwargs: pytest.fail("stale run must not invoke setup"),
    )

    assert cli.main(["run", "interfuser"], project_root=tmp_path) == 1
    assert capsys.readouterr().err == (
        "interfuser environment is missing or stale.\n"
        "Run: ./openloop setup interfuser\n"
    )


def test_bag_uses_structured_profile_default_and_execs_absolute_path(
    cli: ModuleType, monkeypatch, tmp_path: Path
):
    bag = tmp_path / "rosbag/capture.bag"
    bag.parent.mkdir()
    bag.touch()
    _write_profile(tmp_path, "rosbag/capture.bag")
    observed: list[Any] = []

    class ExecCalled(Exception):
        pass

    def execvpe(program: str, argv: list[str], environment: dict[str, str]):
        observed.append((program, argv, environment))
        raise ExecCalled

    monkeypatch.setattr(cli.os, "execvpe", execvpe)

    with pytest.raises(ExecCalled):
        cli.main(["bag"], project_root=tmp_path)

    assert observed == [
        ("rosbag", ["rosbag", "play", "--clock", str(bag.resolve())], dict(os.environ))
    ]


def test_bag_override_resolves_relative_to_caller_directory(
    cli: ModuleType, monkeypatch, tmp_path: Path
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    caller = tmp_path / "caller"
    caller.mkdir()
    bag = caller / "override.bag"
    bag.touch()
    monkeypatch.chdir(caller)

    class ExecCalled(Exception):
        pass

    def execvpe(_program: str, argv: list[str], _environment: dict[str, str]):
        assert argv == ["rosbag", "play", "--clock", str(bag.resolve())]
        raise ExecCalled

    monkeypatch.setattr(cli.os, "execvpe", execvpe)

    with pytest.raises(ExecCalled):
        cli.main(["bag", "--path", "override.bag"], project_root=project_root)


def test_bag_refuses_missing_file_before_exec(
    cli: ModuleType, monkeypatch, tmp_path: Path, capsys
):
    missing = tmp_path / "missing.bag"
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda *_args: pytest.fail("missing bag must not start rosbag"),
    )

    assert cli.main(["bag", "--path", str(missing)], project_root=tmp_path) == 1
    assert capsys.readouterr().err == f"openloop: bag file is missing: {missing}\n"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"observed_bag_contract": []},
        {"observed_bag_contract": {}},
        {"observed_bag_contract": {"bag_path": ""}},
        {"observed_bag_contract": {"bag_path": 123}},
    ],
)
def test_bag_rejects_invalid_profile_structure(
    cli: ModuleType, monkeypatch, tmp_path: Path, capsys, payload: Any
):
    profile = tmp_path / "src/Validator_pkg/config/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda *_args: pytest.fail("invalid profile must not start rosbag"),
    )

    assert cli.main(["bag"], project_root=tmp_path) == 1
    assert capsys.readouterr().err.startswith("openloop: ")


def test_root_shim_is_executable_and_preserves_bootstrap_contract():
    assert ROOT_SHIM.is_file(), f"missing root shim: {ROOT_SHIM}"
    assert os.access(ROOT_SHIM, os.X_OK)
    content = ROOT_SHIM.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'source /opt/ros/noetic/setup.bash' in content
    assert 'if [[ -f "$PROJECT_ROOT/devel/setup.bash" ]]; then' in content
    assert 'source "$PROJECT_ROOT/devel/setup.bash"' in content
    assert 'export OPENLOOP_PROJECT_ROOT="$PROJECT_ROOT"' in content
    assert (
        'export PYTHONPATH="$PROJECT_ROOT/src/Validator_pkg/validator_inference/src'
        '${PYTHONPATH:+:$PYTHONPATH}"'
    ) in content
    assert (
        'exec /usr/bin/python3 '
        '"$PROJECT_ROOT/src/Validator_pkg/validator_inference/scripts/openloop_cli.py" '
        '"$@"'
    ) in content
    subprocess.run(["bash", "-n", str(ROOT_SHIM)], check=True)


def test_default_project_root_prefers_valid_explicit_environment(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """install-space CLI가 명시된 실제 source project를 선택하는지 검사한다."""
    root = tmp_path / "project"
    (root / "src/Validator_pkg/config").mkdir(parents=True)
    (root / "src/Validator_pkg/config/profile.json").write_text("{}\n")
    manifest = root / "src/Validator_pkg/validator_inference/config"
    manifest.mkdir(parents=True)
    (manifest / "model_environments.json").write_text("{}\n")
    monkeypatch.setenv("OPENLOOP_PROJECT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "install/lib/openloop_cli.py"))

    assert cli._default_project_root() == root.resolve()


def test_default_project_root_finds_valid_caller_checkout_from_install_space(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """환경 변수가 없으면 install-space CLI가 caller checkout을 찾는지 검사한다."""
    root = tmp_path / "project"
    (root / "src/Validator_pkg/config").mkdir(parents=True)
    (root / "src/Validator_pkg/config/profile.json").write_text("{}\n")
    manifest = root / "src/Validator_pkg/validator_inference/config"
    manifest.mkdir(parents=True)
    (manifest / "model_environments.json").write_text("{}\n")
    monkeypatch.delenv("OPENLOOP_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "install/lib/openloop_cli.py"))

    assert cli._default_project_root() == root.resolve()


def test_default_project_root_rejects_invalid_explicit_environment(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """명시 project root가 다른 directory를 가리키면 암묵 fallback하지 않는다."""
    monkeypatch.setenv("OPENLOOP_PROJECT_ROOT", str(tmp_path / "not-a-project"))

    with pytest.raises(cli.OpenLoopCliError, match="OPENLOOP_PROJECT_ROOT"):
        cli._default_project_root()


def test_main_reports_invalid_explicit_project_root_without_traceback(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """install-space root 설정 오류가 표준 한 줄 CLI 진단으로 종료되는지 검사한다."""
    monkeypatch.setenv("OPENLOOP_PROJECT_ROOT", str(tmp_path / "not-a-project"))

    assert cli.main(["bag"]) == 1
    assert capsys.readouterr().err == (
        "openloop: OPENLOOP_PROJECT_ROOT must be a canonical source project root\n"
    )


def test_root_shim_clean_checkout_preserves_cli_argument_boundaries(tmp_path: Path):
    shim = tmp_path / "openloop"
    shim.write_bytes(ROOT_SHIM.read_bytes())
    shim.chmod(0o755)
    script = (
        tmp_path
        / "src/Validator_pkg/validator_inference/scripts/openloop_cli.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ROS_") and key != "CATKIN_SETUP_UTIL_ARGS"
    }
    result = subprocess.run(
        [str(shim), "--help", "literal value"],
        check=False,
        capture_output=True,
        text=True,
        env=clean_environment,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["--help", "literal value"]


def test_local_model_environments_are_ignored():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.venvs/" in lines
