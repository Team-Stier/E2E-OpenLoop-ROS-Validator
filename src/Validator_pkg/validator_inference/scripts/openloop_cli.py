#!/usr/bin/env python3
"""
모델 환경 관리와 open-loop ROS 실행을 한 명령으로 제공한다.
입력: setup/run/verify/prune/bag CLI 인자와 저장소 profile/environment state.
출력(pub): 없음. 선택 명령에 따라 환경을 관리하거나 ROS process로 교체한다.
구독(sub): 없음.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from validator_inference import environment_setup
from validator_inference.model_environment import (
    SUPPORTED_MODEL_IDS,
    EnvironmentConfigError,
    EnvironmentStateError,
)


_PACKAGE_RELATIVE_ROOT = Path("src/Validator_pkg/validator_inference")
_MANIFEST_RELATIVE_PATH = _PACKAGE_RELATIVE_ROOT / "config/model_environments.json"
_PROFILE_RELATIVE_PATH = Path("src/Validator_pkg/config/profile.json")
_SENSOR_CONFIG_RELATIVE_PATH = Path(
    "src/Validator_pkg/validator_pipeline/config/0725demo_ros.json"
)


class OpenLoopCliError(RuntimeError):
    """명령 실행 전에 사용자가 수정할 수 있는 입력 오류를 나타낸다."""


def _default_project_root() -> Path:
    """명시 환경, caller cwd 또는 source script에서 실제 저장소 루트를 찾는다."""
    def is_project_root(candidate: Path) -> bool:
        """CLI가 소유한 두 추적 설정 파일로 source project root를 식별한다."""
        return (
            candidate.is_dir()
            and (candidate / _MANIFEST_RELATIVE_PATH).is_file()
            and (candidate / _PROFILE_RELATIVE_PATH).is_file()
        )

    configured = os.environ.get("OPENLOOP_PROJECT_ROOT")
    if configured is not None:
        candidate = Path(configured)
        if (
            not configured
            or not candidate.is_absolute()
            or candidate.resolve() != candidate
            or not is_project_root(candidate)
        ):
            raise OpenLoopCliError(
                "OPENLOOP_PROJECT_ROOT must be a canonical source project root"
            )
        return candidate

    caller = Path.cwd().resolve()
    if is_project_root(caller):
        return caller
    script = Path(__file__).resolve()
    for candidate in script.parents:
        if is_project_root(candidate):
            return candidate
    raise OpenLoopCliError(
        "cannot locate source project root; set OPENLOOP_PROJECT_ROOT"
    )


def _parser() -> argparse.ArgumentParser:
    """축약 옵션을 허용하지 않는 명령 parser를 구성한다."""
    parser = argparse.ArgumentParser(
        prog="./openloop",
        description="Manage model environments and run open-loop validation.",
        epilog=(
            "Normal workflow:\n"
            "  ./openloop setup interfuser\n"
            "  ./openloop interfuser --rviz\n"
            "  ./openloop bag\n\n"
            "Run shorthand:\n"
            "  ./openloop MODEL [--rviz]\n"
            "  ./openloop run MODEL [--rviz]  (explicit equivalent)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser(
        "setup",
        help="create or verify a model environment",
        description="Create or verify a model environment.",
        allow_abbrev=False,
    )
    setup.add_argument(
        "model_id", choices=SUPPORTED_MODEL_IDS, help="model environment to prepare"
    )
    setup.add_argument(
        "--recreate",
        action="store_true",
        help="force creation of a new environment build",
    )

    run = commands.add_parser(
        "run",
        help="run validation using an explicit model argument",
        description="Run open-loop validation with a selected model.",
        allow_abbrev=False,
    )
    run.add_argument(
        "model_id", choices=SUPPORTED_MODEL_IDS, help="model environment to use"
    )
    run.add_argument(
        "--rviz", action="store_true", help="start RViz with the validator"
    )

    verify = commands.add_parser(
        "verify",
        help="deep-check an active model environment",
        description="Deep-check an active model environment.",
        allow_abbrev=False,
    )
    verify.add_argument(
        "model_id", choices=SUPPORTED_MODEL_IDS, help="model environment to verify"
    )

    prune = commands.add_parser(
        "prune",
        help="remove inactive model environment builds",
        description="Remove inactive model environment builds.",
        allow_abbrev=False,
    )
    prune.add_argument(
        "model_id", choices=SUPPORTED_MODEL_IDS, help="model environment to prune"
    )

    bag = commands.add_parser(
        "bag",
        help="play the configured rosbag with /clock",
        description="Play the configured rosbag with simulated time.",
        allow_abbrev=False,
    )
    bag.add_argument("--path", help="override the configured bag path")
    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """첫 token이 정확한 모델 ID일 때만 run shorthand로 변환한다."""
    normalized = list(argv)
    if normalized and normalized[0] in SUPPORTED_MODEL_IDS:
        return ["run", *normalized]
    return normalized


def _launch_argv(project_root: Path, model_id: str, rviz: bool) -> list[str]:
    """통합 validator launch의 shell-free argv를 반환한다."""
    root = Path(project_root).resolve()
    return [
        "roslaunch",
        "Validator_pkg",
        "openloop_validator.launch",
        f"project_root:={root}",
        f"profile_file:={root / _PROFILE_RELATIVE_PATH}",
        f"sensor_config:={root / _SENSOR_CONFIG_RELATIVE_PATH}",
        f"model_id:={model_id}",
        f"model_env_root:={root / '.venvs'}",
        f"morai_ddv2_domain:={'true' if model_id == 'diffusiondrive_v2' else 'false'}",
        f"show_rviz:={'true' if rviz else 'false'}",
    ]


def _bag_argv(bag_path: Path) -> list[str]:
    """시뮬레이션 clock을 제공하는 rosbag argv를 반환한다."""
    return ["rosbag", "play", "--clock", str(Path(bag_path).resolve())]


def _configured_bag_path(project_root: Path) -> Path:
    """profile의 observed bag 경로를 구조적으로 읽어 절대 경로로 만든다."""
    profile_path = project_root / _PROFILE_RELATIVE_PATH
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenLoopCliError(f"cannot read profile: {profile_path}") from exc
    if not isinstance(payload, dict):
        raise OpenLoopCliError("profile must be a JSON object")
    observed = payload.get("observed_bag_contract")
    if not isinstance(observed, dict):
        raise OpenLoopCliError("profile observed_bag_contract must be an object")
    configured = observed.get("bag_path")
    if not isinstance(configured, str) or not configured:
        raise OpenLoopCliError(
            "profile observed_bag_contract.bag_path must be a nonempty string"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_bag_path(project_root: Path, override: Optional[str]) -> Path:
    """명시 경로는 호출 cwd에서, 기본 경로는 저장소 profile에서 해석한다."""
    if override is None:
        return _configured_bag_path(project_root)
    if not override:
        raise OpenLoopCliError("bag path must be nonempty")
    return Path(override).expanduser().resolve()


def _emit_warnings(result: Any) -> None:
    """환경 작업 결과의 경고를 안정된 순서로 stderr에 표시한다."""
    warnings = getattr(result, "warnings", ())
    if not isinstance(warnings, (tuple, list)):
        return
    for warning in warnings:
        if isinstance(warning, str) and warning:
            print(f"warning: {warning}", file=sys.stderr)


def _run_model(project_root: Path, model_id: str, rviz: bool) -> int:
    """실제 read-only runtime 점검 후 통합 roslaunch로 process를 교체한다."""
    try:
        result = environment_setup.preflight_run_environment(project_root, model_id)
    except (
        environment_setup.EnvironmentSetupError,
        EnvironmentConfigError,
        EnvironmentStateError,
        OSError,
    ):
        print(f"{model_id} environment is missing or stale.", file=sys.stderr)
        print(f"Run: ./openloop setup {model_id}", file=sys.stderr)
        return 1
    _emit_warnings(result)

    argv = _launch_argv(project_root, model_id, rviz)
    os.execvpe(argv[0], argv, dict(os.environ))
    return 0


def _play_bag(project_root: Path, override: Optional[str]) -> int:
    """bag 존재 여부를 확인한 뒤 rosbag process로 교체한다."""
    bag_path = _resolve_bag_path(project_root, override)
    if not bag_path.is_file():
        raise OpenLoopCliError(f"bag file is missing: {bag_path}")
    argv = _bag_argv(bag_path)
    os.execvpe(argv[0], argv, dict(os.environ))
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Path] = None,
) -> int:
    """CLI를 dispatch하고 장기 실행 명령은 현재 process를 교체한다."""
    arguments = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(_normalize_argv(arguments))
    try:
        root = (
            _default_project_root()
            if project_root is None
            else Path(project_root).resolve()
        )
        if args.command == "setup":
            result = environment_setup.setup_environment(
                root, args.model_id, recreate=args.recreate
            )
            _emit_warnings(result)
            return 0
        if args.command == "verify":
            result = environment_setup.verify_environment(root, args.model_id)
            _emit_warnings(result)
            return 0
        if args.command == "prune":
            environment_setup.prune_environment_builds(root, args.model_id)
            return 0
        if args.command == "run":
            return _run_model(root, args.model_id, args.rviz)
        if args.command == "bag":
            return _play_bag(root, args.path)
        raise OpenLoopCliError(f"unsupported command: {args.command}")
    except (
        OpenLoopCliError,
        environment_setup.EnvironmentSetupError,
        EnvironmentConfigError,
        EnvironmentStateError,
        OSError,
    ) as exc:
        print(f"openloop: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
