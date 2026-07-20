#!/usr/bin/env python3
"""
선택된 로컬 model profile의 source/checkpoint asset을 내려받고 검증하는 CLI.
입력: --model-id, 로컬 profile.json, models/ 경로.
출력(pub): 없음. 검증된 asset-manifest.json만 models/ 아래에 생성.
구독(sub): 없음.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from validator_inference.assets import (
    AssetError,
    default_project_root,
    fetch_runtime_assets,
)
from validator_inference.profile_store import ProfileError, load_selected_profile


MODEL_IDS = ("tfv6", "interfuser", "diffusiondrive_v2", "sparsedrive_v2")


def _parser() -> argparse.ArgumentParser:
    """asset bootstrap CLI의 명시적 model·profile·refresh 인자를 정의한다.

    입력: 없음.
    반환: 검증 전 CLI argument를 parse할 ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Fetch pinned local source/checkpoint assets and write a verified manifest."
    )
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="explicitly fetch and repair a pre-existing checkout at another revision",
    )
    return parser


def main(argv=None) -> int:
    """선택 model의 source/checkpoint를 확보하고 검증 manifest 경로를 출력한다.

    입력: 선택적으로 test가 주입하는 CLI argument sequence.
    반환: 성공 시 0, profile·asset 오류 시 1.
    부작용: models 아래 source, asset, manifest를 생성·갱신할 수 있다.
    """
    args = _parser().parse_args(argv)
    project_root = (
        args.project_root.resolve() if args.project_root else default_project_root()
    )
    profile_file = (
        args.profile_file.resolve()
        if args.profile_file
        else project_root / "src/Validator_pkg/config/profile.json"
    )
    try:
        selected = load_selected_profile(profile_file, args.model_id)
        manifest = fetch_runtime_assets(
            selected.data,
            model_id=args.model_id,
            project_root=project_root,
            refresh_source=args.refresh_source,
        )
    except (AssetError, ProfileError, OSError, ValueError) as error:
        print(f"fetch_model_assets: {error}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
