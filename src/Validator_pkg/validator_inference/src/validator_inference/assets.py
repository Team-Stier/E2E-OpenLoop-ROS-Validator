"""Deterministic local bootstrap and verification for model runtime assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote
from urllib.request import urlretrieve


SOURCE_DIRECTORY_NAMES = {
    "tfv6": "lead",
    "interfuser": "interfuser",
    "diffusiondrive_v2": "diffusiondrive_v2",
    "sparsedrive_v2": "sparsedrive_v2",
}

DIFFUSION_DERIVATIVE_RELATIVE_PATH = Path(
    "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
)


class AssetError(RuntimeError):
    """Raised for missing, corrupt, or revision-incompatible runtime assets."""


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(project_root).resolve() / candidate).resolve()


def source_checkout_path(profile: Dict[str, Any], project_root: str | Path) -> Path:
    model_id = profile.get("id")
    try:
        directory = SOURCE_DIRECTORY_NAMES[model_id]
    except KeyError as error:
        raise AssetError(f"unsupported model_id for source checkout: {model_id}") from error
    return Path(project_root).resolve() / "models" / "sources" / directory


def _source_metadata(profile: Dict[str, Any], project_root: str | Path) -> Dict[str, Any]:
    model_id = profile.get("id", "<unknown>")
    source = profile.get("source")
    if not isinstance(source, dict):
        raise AssetError(f"model {model_id} has no source declaration")
    expected = source.get("commit")
    repository = source.get("repository")
    if not expected or not repository:
        raise AssetError(f"model {model_id} source requires repository and commit")

    checkout = source_checkout_path(profile, project_root)
    if not checkout.is_dir():
        raise AssetError(
            f"model {model_id} missing source checkout: {checkout}; expected commit {expected}"
        )
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssetError(
            f"model {model_id} source checkout is not a readable git repository: {checkout}; "
            f"expected commit {expected}"
        ) from error
    if observed != expected:
        raise AssetError(
            f"model {model_id} source revision mismatch at {checkout}: "
            f"expected commit {expected}, observed {observed}"
        )

    metadata: Dict[str, Any] = {
        "repository": repository,
        "commit": expected,
        "observed_commit": observed,
        "checkout": str(checkout),
    }
    if source.get("checkpoint_repository"):
        metadata["checkpoint_repository"] = source["checkpoint_repository"]
    if source.get("checkpoint_revision") is not None:
        metadata["checkpoint_revision"] = source["checkpoint_revision"]

    derivative = checkout / DIFFUSION_DERIVATIVE_RELATIVE_PATH
    if model_id == "diffusiondrive_v2" and derivative.is_file():
        metadata["local_derivative"] = {
            "path": str(DIFFUSION_DERIVATIVE_RELATIVE_PATH),
            "sha256": sha256_file(derivative),
        }
    return metadata


def verify_runtime_assets(
    profile: Dict[str, Any],
    project_root: str | Path | None = None,
    *,
    verify_source: bool = True,
) -> Dict[str, Any]:
    """Verify declared assets without downloading and return stable manifest metadata."""
    root = Path(project_root).resolve() if project_root else default_project_root()
    model_id = profile.get("id", "<unknown>")
    assets = []
    for declaration in profile.get("runtime_assets", []):
        path = _resolve(declaration["path"], root)
        expected = declaration.get("sha256")
        expected_label = expected if expected is not None else "<unconstrained>"
        if not path.is_file():
            raise AssetError(
                f"model {model_id} missing runtime asset {path}; "
                f"expected sha256 {expected_label}"
            )
        observed = sha256_file(path)
        if expected is not None and observed != expected:
            raise AssetError(
                f"model {model_id} runtime asset hash mismatch at {path}: "
                f"expected sha256 {expected}, observed sha256 {observed}"
            )
        item = {
            "role": declaration.get("role", ""),
            "path": declaration["path"],
            "resolved_path": str(path),
            "declared_sha256": expected,
            "observed_sha256": observed,
        }
        for key in (
            "source_url",
            "source_path",
            "source_repository_path",
            "shape",
        ):
            if key in declaration:
                item[key] = declaration[key]
        assets.append(item)

    metadata: Dict[str, Any] = {"model_id": model_id, "assets": assets}
    if verify_source:
        metadata["source"] = _source_metadata(profile, root)
    return metadata


def _manifest_path(profile: Dict[str, Any], project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "models" / str(profile["id"]) / "asset-manifest.json"


def write_asset_manifest(
    profile: Dict[str, Any], metadata: Dict[str, Any], project_root: str | Path | None = None
) -> Path:
    """Write canonical JSON only from already verified metadata."""
    root = Path(project_root).resolve() if project_root else default_project_root()
    payload = {
        "model_id": profile["id"],
        "source": metadata.get("source", {}),
        "assets": metadata["assets"],
    }
    path = _manifest_path(profile, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(str(temporary), str(path))
    return path


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssetError(f"command failed: {' '.join(command)}") from error


def _ensure_source_checkout(
    profile: Dict[str, Any], project_root: Path, refresh_source: bool
) -> Path:
    source = profile["source"]
    checkout = source_checkout_path(profile, project_root)
    expected = source["commit"]
    if checkout.exists():
        try:
            observed = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise AssetError(f"existing source path is not a git checkout: {checkout}") from error
        if observed == expected:
            return checkout
        if not refresh_source:
            raise AssetError(
                f"model {profile['id']} source revision mismatch at {checkout}: "
                f"expected commit {expected}, observed {observed}; use --refresh-source"
            )
        _run(["git", "-C", str(checkout), "fetch", "origin", expected])
        _run(["git", "-C", str(checkout), "checkout", "--detach", expected])
        return checkout

    checkout.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-checkout", source["repository"], str(checkout)])
    try:
        _run(["git", "-C", str(checkout), "checkout", "--detach", expected])
    except AssetError:
        _run(["git", "-C", str(checkout), "fetch", "origin", expected])
        _run(["git", "-C", str(checkout), "checkout", "--detach", expected])
    return checkout


def _checkpoint_url(profile: Dict[str, Any], asset: Dict[str, Any]) -> Optional[str]:
    if asset.get("source_url"):
        return str(asset["source_url"])
    source_path = asset.get("source_path")
    checkpoint_repository = profile.get("source", {}).get("checkpoint_repository")
    if source_path and checkpoint_repository:
        revision = profile["source"].get("checkpoint_revision", "main")
        return (
            f"{checkpoint_repository.rstrip('/')}/resolve/{quote(str(revision), safe='')}/"
            f"{quote(str(source_path), safe='/')}"
        )
    return None


def _download_atomic(
    url: str, destination: Path, downloader: Callable[[str, str | Path], Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        downloader(url, temporary)
        if not temporary.is_file():
            raise AssetError(f"download did not create temporary asset for {destination}")
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_exact_line_groups(text: str) -> str:
    forbidden_imports = (
        "from omegaconf import OmegaConf\n",
        "from hydra.utils import instantiate\n",
        "from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator\n",
        "from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer\n",
        "from navsim.evaluate.pdm_score import pdm_score, pdm_score_para\n",
        "from navsim.common.dataclasses import Trajectory\n",
        "import lzma\n",
        "import pickle\n",
        "import concurrent.futures as cf, cloudpickle, os\n",
        "import multiprocessing as mp\n",
        "from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (\n    WeightedMetricIndex as WIdx,\n)\n",
        "from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex\n",
        "import joblib\n",
    )
    for item in forbidden_imports:
        text = text.replace(item, "")
    return text


def create_diffusiondrive_derivative(checkout: str | Path) -> Dict[str, str]:
    """Create the pinned PDM-free SEL derivative beside its upstream module."""
    checkout = Path(checkout)
    source = checkout / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py"
    destination = checkout / DIFFUSION_DERIVATIVE_RELATIVE_PATH
    if not source.is_file():
        raise AssetError(f"DiffusionDrive V2 missing pinned source module: {source}")
    text = source.read_text(encoding="utf-8")
    text = _remove_exact_line_groups(text)

    helper_start = text.find("def _pairwise_subscores")
    class_start = text.find("class V2TransfuserModel")
    if helper_start < 0 or class_start < 0 or helper_start >= class_start:
        raise AssetError(f"DiffusionDrive V2 derivative markers missing in {source}")
    text = text[:helper_start] + text[class_start:]

    pdm_init_start = text.find("        # pdm score\n")
    pdm_init_end = text.find("        self.loss_bce =", pdm_init_start)
    if pdm_init_start < 0 or pdm_init_end < 0:
        raise AssetError(f"DiffusionDrive V2 PDM constructor markers missing in {source}")
    text = text[:pdm_init_start] + text[pdm_init_end:]

    outer_signature = (
        "    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, "
        "eta=0.0, metric_cache=None, cal_pdm=True,token=None) -> Dict[str, torch.Tensor]:"
    )
    text = text.replace(
        outer_signature,
        "    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, eta=0.0) -> Dict[str, torch.Tensor]:",
        1,
    )
    text = text.replace("targets=targets,global_img=None,eta=eta,metric_cache=metric_cache, cal_pdm=cal_pdm,token=token)", "targets=targets,global_img=None,eta=eta)", 1)

    head_forward_start = text.find(
        "    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets=None,global_img=None,eta=0.0,metric_cache=None, cal_pdm=True, old_model=False,token=None)"
    )
    next_method = text.find("    def get_pdm_score_para", head_forward_start)
    if head_forward_start < 0 or next_method < 0:
        raise AssetError(f"DiffusionDrive V2 selector forward markers missing in {source}")
    head_forward = '''    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, camera_feature, targets=None, global_img=None, eta=0.0) -> Dict[str, torch.Tensor]:
        """PDM-free open-loop selector forward."""
        return self.forward_test_rl_without_pdm(
            ego_query, agents_query, bev_feature, bev_spatial_shape,
            status_encoding, status_feature, camera_feature, targets, global_img, eta
        )

'''
    text = text[:head_forward_start] + head_forward + text[next_method:]

    test_signature = "    def forward_test_rl(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets,global_img,metric_cache,eta=1.0,token=None)"
    text = text.replace(
        test_signature,
        "    def forward_test_rl_without_pdm(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets,global_img,eta=1.0)",
        1,
    )
    selector_marker = (
        "        # for official eval\n"
        "        # return {\"trajectory\": traj_to_score[:,-1]}\n"
    )
    selector_start = text.find(selector_marker)
    selector_end = text.find("    def compute_diversity", selector_start)
    if selector_start < 0 or selector_end < 0:
        raise AssetError(f"DiffusionDrive V2 final selector markers missing in {source}")
    text = (
        text[:selector_start]
        + "        return {\"trajectory\": traj_to_score[:, -1]}\n\n"
        + text[selector_end:]
    )
    header = (
        '"""Generated PDM-free open-loop derivative.\n\n'
        "Upstream commit is recorded by asset-manifest.json. Do not edit in place.\n"
        '"""\n'
    )
    destination.write_text(header + text, encoding="utf-8")
    return {
        "path": str(DIFFUSION_DERIVATIVE_RELATIVE_PATH),
        "sha256": sha256_file(destination),
    }


def fetch_runtime_assets(
    profile: Dict[str, Any],
    *,
    model_id: str,
    project_root: str | Path | None = None,
    refresh_source: bool = False,
    downloader: Callable[[str, str | Path], Any] = urlretrieve,
) -> Path:
    """Fetch one selected model's declared source/assets and write a verified manifest."""
    root = Path(project_root).resolve() if project_root else default_project_root()
    if profile.get("id") != model_id:
        raise AssetError(
            f"selected profile model_id {profile.get('id')} does not match requested {model_id}"
        )
    manifest = _manifest_path(profile, root)
    if manifest.exists():
        manifest.unlink()

    checkout = _ensure_source_checkout(profile, root, refresh_source)
    if model_id == "diffusiondrive_v2":
        create_diffusiondrive_derivative(checkout)

    for asset in profile.get("runtime_assets", []):
        destination = _resolve(asset["path"], root)
        if destination.is_file():
            continue
        repository_path = asset.get("source_repository_path")
        if repository_path:
            source = (checkout / repository_path).resolve()
            if not source.is_file() or checkout.resolve() not in source.parents:
                raise AssetError(
                    f"model {model_id} missing source_repository_path {repository_path} at {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.part")
            shutil.copyfile(source, temporary)
            os.replace(str(temporary), str(destination))
            continue
        url = _checkpoint_url(profile, asset)
        if not url:
            raise AssetError(
                f"model {model_id} asset {asset['path']} has no source_url, "
                "checkpoint source_path, or source_repository_path"
            )
        _download_atomic(url, destination, downloader)

    metadata = verify_runtime_assets(profile, project_root=root)
    return write_asset_manifest(profile, metadata, project_root=root)
