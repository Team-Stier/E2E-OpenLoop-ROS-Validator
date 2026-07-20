"""profile에 대응하는 concrete source-native model runtime 모음.

입력: InferenceFrame과 로컬 checkpoint/source asset.
출력(pub): 없음. ModelPrediction을 반환.
구독(sub): 없음.
"""

from .diffusiondrive_v2 import DiffusionDriveV2InferenceModel
from .interfuser import InterFuserModel
from .sparsedrive_v2 import SparseDriveV2Model
from .tfv6 import TFv6Model

__all__ = [
    "TFv6Model",
    "InterFuserModel",
    "DiffusionDriveV2InferenceModel",
    "SparseDriveV2Model",
]
