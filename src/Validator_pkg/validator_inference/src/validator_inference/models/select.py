"""
로컬 profile의 model_id로 정확히 하나의 concrete runtime을 선택한다.
입력: SelectedProfile과 project root.
출력(pub): 없음. 선택된 concrete model 객체.
구독(sub): 없음.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validator_inference.assets import default_project_root
from validator_inference.models.diffusiondrive_v2 import (
    DiffusionDriveV2InferenceModel,
)
from validator_inference.models.interfuser import InterFuserModel
from validator_inference.models.sparsedrive_v2 import SparseDriveV2Model
from validator_inference.models.tfv6 import TFv6Model


def build_selected_model(
    selected_profile: Any, project_root: str | Path | None = None
) -> object:
    """선택 profile의 model id에 대응하는 정확히 하나의 concrete wrapper를 만든다.

    입력: ``SelectedProfile``과 선택적 project root.
    반환: 아직 lazy-load 상태인 TFv6·InterFuser·DiffusionDrive·SparseDrive runtime.
    """
    root = Path(project_root).resolve() if project_root else default_project_root()
    concrete_types = {
        "tfv6": TFv6Model,
        "interfuser": InterFuserModel,
        "diffusiondrive_v2": DiffusionDriveV2InferenceModel,
        "sparsedrive_v2": SparseDriveV2Model,
    }
    try:
        concrete_type = concrete_types[selected_profile.model_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported model_id: {selected_profile.model_id}"
        ) from error
    return concrete_type(selected_profile, root)
