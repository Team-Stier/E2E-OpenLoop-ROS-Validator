#!/usr/bin/env python3
"""Fetch and verify source/checkpoint assets for one local model profile."""

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
