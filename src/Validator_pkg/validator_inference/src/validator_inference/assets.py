"""
모델 source/checkpoint asset의 로컬 bootstrap, hash 검증, manifest 생성을 담당한다.
입력: 선택된 profile, models/ 아래의 source 및 asset 파일.
출력(pub): 없음. 검증된 asset-manifest.json을 로컬에 생성.
구독(sub): 없음.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
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
DIFFUSION_SOURCE_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
DIFFUSION_SELECTOR_STAGES = (
    "diffusion_denoising",
    "coarse_scorer",
    "top_32",
    "fine_scorer",
    "final_trajectory",
)


class AssetError(RuntimeError):
    """runtime asset이 없거나 손상됐거나 revision과 맞지 않을 때 발생한다."""


def default_project_root() -> Path:
    """설치 위치와 무관하게 현재 workspace의 root를 계산한다.

    입력: 없음.
    반환: ``src/Validator_pkg``를 포함하는 project root 경로.
    """
    return Path(__file__).resolve().parents[5]


def sha256_file(path: str | Path) -> str:
    """큰 binary asset도 고정 크기 chunk로 읽어 SHA-256을 계산한다.

    입력: 읽을 수 있는 로컬 파일 경로.
    반환: 파일 전체 내용의 16진수 SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, project_root: str | Path) -> Path:
    """profile 상대 경로를 project root 기준의 절대 경로로 정규화한다.

    입력: profile path와 project root.
    반환: symbolic link가 해석된 절대 ``Path``.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(project_root).resolve() / candidate).resolve()


def source_checkout_path(profile: Dict[str, Any], project_root: str | Path) -> Path:
    """model id에 대응하는 고정 source checkout directory를 반환한다.

    입력: model profile과 workspace root.
    반환: ``models/sources/<model>`` 아래의 checkout path.
    """
    model_id = profile.get("id")
    try:
        directory = SOURCE_DIRECTORY_NAMES[model_id]
    except KeyError as error:
        raise AssetError(f"unsupported model_id for source checkout: {model_id}") from error
    return Path(project_root).resolve() / "models" / "sources" / directory


def _source_metadata(profile: Dict[str, Any], project_root: str | Path) -> Dict[str, Any]:
    """pinned Git checkout이 profile commit과 일치하는지 확인해 metadata를 만든다.

    입력: 선택 profile과 project root.
    반환: repository, 선언·관측 commit, checkout, 선택적 derivative hash metadata.
    """
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
    """다운로드 없이 선언된 asset과 선택적 source commit을 검증한다.

    입력: 선택 profile, project root, source 검증 여부.
    반환: 관측 SHA-256과 source revision을 담은 manifest metadata.
    """
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
    """선택 모델의 runtime asset manifest 저장 경로를 계산한다.

    입력: model profile과 project root.
    반환: ``models/<model>/asset-manifest.json`` 절대 경로.
    """
    return Path(project_root).resolve() / "models" / str(profile["id"]) / "asset-manifest.json"


def write_asset_manifest(
    profile: Dict[str, Any], metadata: Dict[str, Any], project_root: str | Path | None = None
) -> Path:
    """이미 검증된 metadata만으로 atomic canonical JSON manifest를 기록한다.

    입력: model profile, 검증 metadata, 선택적 project root.
    반환: 새로 기록한 manifest 경로.
    부작용: parent directory를 만들고 기존 manifest를 atomic replace한다.
    """
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
    """외부 bootstrap 명령을 실행하고 실패를 domain 오류로 바꾼다.

    입력: 인자 분리된 실행 명령.
    부작용: 명령 실패 시 ``AssetError``를 발생시킨다.
    """
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssetError(f"command failed: {' '.join(command)}") from error


def _ensure_source_checkout(
    profile: Dict[str, Any], project_root: Path, refresh_source: bool
) -> Path:
    """source checkout을 clone하거나 profile이 고정한 commit으로 맞춘다.

    입력: profile, project root, 기존 checkout refresh 허용 여부.
    반환: 검증 가능한 detached source checkout path.
    부작용: 필요할 때 Git clone/fetch/checkout을 실행한다.
    """
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
    """asset 선언에서 다운로드할 checkpoint URL을 결정한다.

    입력: model profile과 runtime asset 선언.
    반환: 직접 URL 또는 Hugging Face resolve URL, 불명확하면 ``None``.
    """
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
    """asset을 임시 파일로 내려받아 검증 전 불완전 파일 노출을 막는다.

    입력: URL, 최종 destination, 주입 가능한 downloader.
    부작용: 성공 시 destination을 atomic replace하고 실패 임시 파일을 제거한다.
    """
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


def _publish_generated_bytes(destination: Path, payload: bytes) -> None:
    """동일 generated bytes는 보존하고 변경된 bytes만 atomic replace한다."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        path_metadata = destination.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise AssetError(f"cannot inspect generated asset: {destination}") from error
    else:
        if not stat.S_ISREG(path_metadata.st_mode):
            raise AssetError(
                f"generated asset destination is not a regular file: {destination}"
            )
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                destination,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != path_metadata.st_dev
                or opened.st_ino != path_metadata.st_ino
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise AssetError(
                    f"generated asset changed while opening: {destination}"
                )
            blocks = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                blocks.append(block)
            after = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise AssetError(
                    f"generated asset changed while reading: {destination}"
                )
            existing = b"".join(blocks)
        except AssetError:
            raise
        except (MemoryError, OSError) as error:
            raise AssetError(f"cannot read generated asset: {destination}") from error
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
    if existing == payload:
        return

    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".generated",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        temporary.chmod(0o644)
        os.replace(temporary, destination)
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except (OSError, ValueError) as error:
        raise AssetError(f"cannot publish generated asset: {destination}") from error
    finally:
        if not handle.closed:
            handle.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _source_segment(lines: list[str], node: ast.AST) -> str:
    """AST node가 차지하는 원본 source line 범위를 그대로 추출한다.

    입력: 줄 단위 source와 시작·끝 line을 가진 AST node.
    반환: node 범위의 원본 Python text.
    """
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _replace_original_line_ranges(
    text: str, replacements: list[tuple[int, int, str]]
) -> str:
    """원본 source의 여러 line 범위를 뒤쪽부터 안전하게 교체한다.

    입력: 전체 source text와 ``(start, end, replacement)`` 목록.
    반환: line number 충돌 없이 모든 replacement를 반영한 source text.
    """
    lines = text.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1 : end] = [replacement]
    return "".join(lines)


def _trajectory_head_derivative(
    source_text: str, source_path: Path, tree: ast.Module
) -> tuple[ast.ClassDef, str]:
    """원본 TrajectoryHead에서 PDM 평가를 제거한 learned-selector source를 만든다.

    입력: 고정 commit의 원본 source text/path와 parse tree.
    반환: 원본 ``TrajectoryHead`` node와 PDM-free replacement class text.
    """
    lines = source_text.splitlines(keepends=True)
    trajectory = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrajectoryHead"
        ),
        None,
    )
    if trajectory is None:
        raise AssetError(f"DiffusionDrive V2 TrajectoryHead missing in {source_path}")
    methods = {
        node.name: node
        for node in trajectory.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "__init__",
        "norm_odo",
        "denorm_odo",
        "_get_scorer_inputs",
        "_select_topk",
        "add_mul_noise",
        "forward_test_rl",
        "bezier_xyyaw",
    }
    if required - set(methods):
        raise AssetError(
            f"DiffusionDrive V2 derivative methods missing in {source_path}: "
            f"{sorted(required - set(methods))}"
        )

    constructor = _source_segment(lines, methods["__init__"])
    pdm_start = constructor.find("        # pdm score\n")
    pdm_end = constructor.find("        self.loss_bce =", pdm_start)
    vocab_start = constructor.find("        self.vocab_pdm_gt_path =")
    if pdm_start < 0 or pdm_end < 0 or vocab_start < 0:
        raise AssetError(
            f"DiffusionDrive V2 pinned constructor markers changed in {source_path}"
        )
    constructor = constructor[:pdm_start] + constructor[pdm_end:vocab_start]

    source_forward = _source_segment(lines, methods["forward_test_rl"])
    signature_end = source_forward.find("\n")
    source_forward = (
        "    def forward_test_rl_without_pdm(self, ego_query,agents_query,bev_feature,"
        "bev_spatial_shape,status_encoding,status_feature,camera_feature, targets,"
        "global_img,eta=1.0) -> Dict[str, torch.Tensor]:"
        + source_forward[signature_end:]
    )
    fine_start = source_forward.find("        loss_fine, final_fine_reward")
    fine_end_marker = "        )\n"
    fine_end = source_forward.find(fine_end_marker, fine_start)
    if fine_start < 0 or fine_end < 0:
        raise AssetError(
            f"DiffusionDrive V2 pinned fine-selector markers changed in {source_path}"
        )
    fine_end += len(fine_end_marker)
    fine_selection = '''        fine_best_idx_list = []
        for feat in fine_traj_feature_list:
            EP_score = self.fine_EP_head(feat).squeeze(-1)
            NC_score = self.fine_NC_head(feat).squeeze(-1)
            DAC_score = self.fine_DAC_head(feat).squeeze(-1)
            TTC_score = self.fine_TTC_head(feat).squeeze(-1)
            C_score = self.fine_C_head(feat).squeeze(-1)
            final_fine_score = (
                self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
                (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) +
                 2 * self.sigmoid(C_score)) / 12
            )
            fine_best_idx_list.append(torch.argmax(final_fine_score, dim=-1))
'''
    source_forward = (
        source_forward[:fine_start] + fine_selection + source_forward[fine_end:]
    )
    concat = "        traj_to_score = torch.cat(traj_to_score, dim=1)"
    concat_start = source_forward.find(concat)
    if concat_start < 0:
        raise AssetError(
            f"DiffusionDrive V2 pinned final-selector marker changed in {source_path}"
        )
    concat_end = source_forward.find("\n", concat_start) + 1
    source_forward = (
        source_forward[:concat_end]
        + '        return {"trajectory": traj_to_score[:, -1]}\n'
    )

    class_preamble = "".join(
        lines[trajectory.lineno - 1 : methods["__init__"].lineno - 1]
    )
    forward = '''    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, camera_feature, targets=None, global_img=None, eta=0.0) -> Dict[str, torch.Tensor]:
        return self.forward_test_rl_without_pdm(
            ego_query, agents_query, bev_feature, bev_spatial_shape,
            status_encoding, status_feature, camera_feature, targets, global_img, eta
        )

'''
    retained = [
        constructor,
        _source_segment(lines, methods["norm_odo"]),
        _source_segment(lines, methods["denorm_odo"]),
        _source_segment(lines, methods["_get_scorer_inputs"]),
        _source_segment(lines, methods["_select_topk"]),
        _source_segment(lines, methods["add_mul_noise"]),
        source_forward,
        _source_segment(lines, methods["bezier_xyyaw"]),
    ]
    return trajectory, class_preamble + constructor + "\n" + forward + "\n".join(retained[1:])


def selector_flow_stages(source_text: str) -> tuple[str, ...]:
    """생성된 method의 learned-selector 단계가 원본 순서를 보존했는지 점검한다.

    입력: 생성된 DiffusionDrive PDM-free derivative source text.
    반환: 검증을 통과한 고정 stage 이름 tuple.
    """
    tree = ast.parse(source_text)
    trajectory = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrajectoryHead"
        ),
        None,
    )
    if trajectory is None:
        raise AssetError("DiffusionDrive V2 derivative has no TrajectoryHead")
    method = next(
        (
            node
            for node in trajectory.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_test_rl_without_pdm"
        ),
        None,
    )
    if method is None:
        raise AssetError(
            "DiffusionDrive V2 derivative has no forward_test_rl_without_pdm"
        )

    def call_line(attribute: str) -> int:
        """selector method 안에서 지정한 method 호출의 최초 source line을 찾는다.

        입력: ``diff_decoder`` 같은 AST attribute 이름.
        반환: 가장 이른 호출 line number.
        """
        lines = [
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute
        ]
        if not lines:
            raise AssetError(
                f"DiffusionDrive V2 selector method is missing {attribute} stage"
            )
        return min(lines)

    diffusion_line = call_line("diff_decoder")
    coarse_line = call_line("scorer_decoder")
    fine_line = call_line("fine_scorer_decoder")
    top_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "topk" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 32
    ]
    if not top_lines:
        raise AssetError("DiffusionDrive V2 selector method must retain topk = 32")
    top_line = min(top_lines)
    select_topk_line = call_line("_select_topk")
    if select_topk_line < top_line:
        raise AssetError("DiffusionDrive V2 _select_topk occurs before topk = 32")

    return_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Return)
        and ast.get_source_segment(source_text, node.value)
        == '{"trajectory": traj_to_score[:, -1]}'
    ]
    append_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "traj_to_score"
    ]
    cat_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "cat"
    ]
    if not return_lines or not append_lines or not cat_lines:
        raise AssetError(
            "DiffusionDrive V2 selector method lacks fine append/cat/final trajectory return"
        )
    return_line = min(return_lines)
    if not (
        diffusion_line
        < coarse_line
        < top_line
        <= select_topk_line
        < fine_line
        < max(append_lines)
        < max(cat_lines)
        < return_line
    ):
        raise AssetError(
            "DiffusionDrive V2 learned selector stages are not in source order"
        )
    return DIFFUSION_SELECTOR_STAGES


def create_diffusiondrive_derivative(checkout: str | Path) -> Dict[str, str]:
    """고정 DiffusionDrive source에서 PDM 평가 없는 selector derivative를 생성한다.

    입력: profile이 고정한 upstream Git checkout.
    반환: 생성 파일 상대 경로, SHA-256, 검증 commit metadata.
    부작용: checkout 내부 derivative Python module을 생성·검증한다.
    """
    checkout = Path(checkout)
    try:
        observed_commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssetError(
            f"DiffusionDrive V2 derivative requires a readable pinned checkout: {checkout}"
        ) from error
    if observed_commit != DIFFUSION_SOURCE_COMMIT:
        raise AssetError(
            f"DiffusionDrive V2 derivative requires commit {DIFFUSION_SOURCE_COMMIT}, "
            f"observed {observed_commit} at {checkout}"
        )

    source = checkout / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py"
    destination = checkout / DIFFUSION_DERIVATIVE_RELATIVE_PATH
    if not source.is_file():
        raise AssetError(f"DiffusionDrive V2 missing pinned source module: {source}")
    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))
    lines = source_text.splitlines(keepends=True)
    trajectory, trajectory_text = _trajectory_head_derivative(
        source_text, source, tree
    )

    banned_modules = {
        "omegaconf",
        "hydra.utils",
        "navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator",
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
        "navsim.evaluate.pdm_score",
        "navsim.common.dataclasses",
        "lzma",
        "pickle",
        "concurrent.futures",
        "multiprocessing",
        "navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums",
        "joblib",
    }
    removals: list[tuple[int, int, str]] = [
        (trajectory.lineno, trajectory.end_lineno, trajectory_text)
    ]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "") in banned_modules:
            removals.append((node.lineno, node.end_lineno, ""))
        elif isinstance(node, ast.Import) and any(
            alias.name in banned_modules for alias in node.names
        ):
            removals.append((node.lineno, node.end_lineno, ""))
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_pairwise_subscores",
            "_pairwise_scores",
            "_pdm_worker",
            "_init_pool",
        }:
            removals.append((node.lineno, node.end_lineno, ""))
    text = _replace_original_line_ranges(source_text, removals)

    outer_signature = (
        "    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, "
        "eta=0.0, metric_cache=None, cal_pdm=True,token=None) -> Dict[str, torch.Tensor]:"
    )
    outer_call = (
        "targets=targets,global_img=None,eta=eta,metric_cache=metric_cache, "
        "cal_pdm=cal_pdm,token=token)"
    )
    if text.count(outer_signature) != 1 or text.count(outer_call) != 1:
        raise AssetError(
            f"DiffusionDrive V2 pinned outer-forward markers changed in {source}"
        )
    text = text.replace(
        outer_signature,
        "    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, eta=0.0) -> Dict[str, torch.Tensor]:",
        1,
    ).replace(
        outer_call,
        "targets=targets,global_img=None,eta=eta)",
        1,
    )
    header = (
        '"""생성된 evaluation-free open-loop derivative.\n\n'
        f"Exact upstream commit: {DIFFUSION_SOURCE_COMMIT}. Do not edit in place.\n"
        '"""\n'
    )
    text = header + text
    banned_tokens = (
        "pdm_planner",
        "navsim.evaluate.pdm_score",
        "ProcessPoolExecutor",
        "_pdm_pool",
        "metric_cache",
        "get_pdm_score_para",
        "reward_dict",
    )
    present = [token for token in banned_tokens if token in text]
    if present:
        raise AssetError(
            f"DiffusionDrive V2 derivative retained evaluation token(s): {present}"
        )
    try:
        compile(text, str(destination), "exec")
    except SyntaxError as error:
        raise AssetError(
            f"DiffusionDrive V2 generated derivative does not compile: {error}"
        ) from error
    selector_flow_stages(text)
    _publish_generated_bytes(destination, text.encode("utf-8"))
    return {
        "path": str(DIFFUSION_DERIVATIVE_RELATIVE_PATH),
        "sha256": sha256_file(destination),
        "source_commit": observed_commit,
    }


def fetch_runtime_assets(
    profile: Dict[str, Any],
    *,
    model_id: str,
    project_root: str | Path | None = None,
    refresh_source: bool = False,
    downloader: Callable[[str, str | Path], Any] = urlretrieve,
) -> Path:
    """선택 모델의 선언 source·asset을 bootstrap하고 검증 manifest를 기록한다.

    입력: selected profile, model id, project root, source refresh, downloader.
    반환: hash와 source revision을 검증한 asset manifest 경로.
    부작용: source checkout·checkpoint·derivative·manifest 파일을 생성할 수 있다.
    """
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
