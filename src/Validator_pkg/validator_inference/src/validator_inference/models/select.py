"""Select exactly one concrete model runtime from the local profile."""

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
