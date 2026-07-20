"""
한 번에 하나의 model forward만 실행하고 completion을 한 칸 mailbox에 보관한다.
입력: InferenceFrame과 concrete model.
출력(pub): 없음. Completion을 engine이 꺼내도록 제공.
구독(sub): 없음.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock, Thread
from time import perf_counter_ns
from typing import Any, Optional

from validator_inference.models.contracts import ModelPrediction


@dataclass(frozen=True)
class SubmitResult:
    """worker가 한 frame 제출을 수락했는지와 거절 사유를 표현한다.

    입력: 수락 여부와 결과 이유 문자열.
    반환: engine이 busy metric을 결정하는 불변 제출 결과.
    """

    accepted: bool
    reason: str


class WorkerExecutionError(RuntimeError):
    """worker 내부에서 source-native model forward가 실패했음을 나타낸다."""

    def __init__(self, model_id: str, message: str) -> None:
        """model id를 보존한 worker forward 오류를 생성한다.

        입력: 실패한 model id와 원인 설명.
        반환: engine까지 전달되는 ``WorkerExecutionError`` instance.
        """
        self.model_id = model_id
        super().__init__("model {} forward failed: {}".format(model_id, message))


@dataclass(frozen=True)
class Completion:
    """worker thread가 engine mailbox에 넣는 forward 완료 결과다.

    입력: 원본 frame, model id, prediction/오류, 시간, 선택적 RNG seed.
    반환: 다음 timer tick이 소비할 불변 completion.
    """

    frame: Any
    model_id: str
    result: Optional[ModelPrediction]
    elapsed_ms: float
    rng_seed: int
    error: Optional[WorkerExecutionError]


def seed_from(run_id: str, model_id: str, reference_stamp_ns: int) -> int:
    """profile 계약의 안정적인 unsigned 64-bit diffusion seed를 만든다.

    입력: 평가 run id, model id, frame 기준 nanosecond timestamp.
    반환: 같은 입력에서 항상 같은 64-bit seed.
    """
    raw = "{}:{}:{}".format(model_id, int(reference_stamp_ns), run_id).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _stamp_ns(stamp: Any) -> int:
    """frame 기준 ROS stamp를 정수 nanosecond로 변환한다.

    입력: ``to_nsec`` 또는 초·나노초 field를 가진 stamp.
    반환: 정수 nanosecond timestamp.
    """
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    raise TypeError("reference stamp must provide nanoseconds")


def _model_id(model: Any) -> str:
    """concrete runtime에서 선택 profile의 model id를 안전하게 얻는다.

    입력: ``selected_profile.model_id`` 또는 ``model_id``를 노출한 model.
    반환: 빈 값이 아닌 model id 문자열.
    """
    selected_profile = getattr(model, "selected_profile", None)
    value = getattr(selected_profile, "model_id", None)
    if value is None:
        value = getattr(model, "model_id", None)
    if not value:
        raise ValueError("inference model must expose selected_profile.model_id")
    return str(value)


class InferenceWorker:
    """model forward 하나와 completion 하나만 보관하는 비동기 worker다.

    입력: concrete model runtime과 평가 run id.
    반환: engine이 소비할 completion 또는 즉시 busy 거절 결과.
    """

    def __init__(self, model: Any, run_id: str) -> None:
        """model runtime, run id, forward/mailbox lock을 초기화한다.

        입력: ``forward``를 제공하는 선택 model과 run id.
        부작용: daemon thread와 completion을 보호할 lock을 생성한다.
        """
        self._model = model
        self._model_id = _model_id(model)
        self._run_id = str(run_id)
        self._forward_lock = Lock()
        self._completion_lock = Lock()
        self._completion = None  # type: Optional[Completion]

    def try_submit(self, frame: Any) -> SubmitResult:
        """유휴 worker에 frame을 제출하거나 queue 없이 즉시 거절한다.

        입력: ``InferenceFrame`` 호환 causal snapshot.
        반환: accepted 또는 ``busy_tick_skipped`` 제출 결과.
        """
        if not self._forward_lock.acquire(blocking=False):
            return SubmitResult(False, "busy_tick_skipped")
        # 한 칸 completion mailbox가 유실되지 않게 지킨다. engine이 직전에
        # completion을 꺼낸 뒤 이전 forward가 끝날 수 있다. 이 interleaving에서는
        # forward lock이 비어도, 그 completion을 발행하기 전 새 forward를 실행하면 안 된다.
        with self._completion_lock:
            if self._completion is not None:
                self._forward_lock.release()
                return SubmitResult(False, "busy_tick_skipped")
        try:
            thread = Thread(target=self._run_forward, args=(frame,), daemon=True)
            thread.start()
        except Exception:
            self._forward_lock.release()
            raise
        return SubmitResult(True, "accepted")

    def take_completed(self) -> Optional[Completion]:
        """mailbox의 completion을 한 번만 꺼내고 비운다.

        입력: 없음.
        반환: 완료 결과 또는 아직 완료가 없을 때 ``None``.
        """
        with self._completion_lock:
            completion = self._completion
            self._completion = None
            return completion

    def _run_forward(self, frame: Any) -> None:
        """daemon thread에서 forward 시간을 재고 결과·오류를 mailbox에 저장한다.

        입력: worker가 수락한 causal frame.
        부작용: DiffusionDrive seed를 만들고 completion을 기록한 뒤 lock을 해제한다.
        """
        rng_seed = 0
        result = None  # type: Optional[ModelPrediction]
        worker_error = None  # type: Optional[WorkerExecutionError]
        elapsed_ms = 0.0
        try:
            if self._model_id == "diffusiondrive_v2":
                rng_seed = seed_from(
                    self._run_id,
                    self._model_id,
                    _stamp_ns(frame.reference_stamp),
                )

            start_ns = perf_counter_ns()
            try:
                if self._model_id == "diffusiondrive_v2":
                    result = self._model.forward(frame, rng_seed=rng_seed)
                else:
                    result = self._model.forward(frame)
            finally:
                elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0

            if not isinstance(result, ModelPrediction):
                raise TypeError("forward must return ModelPrediction")
        except Exception as error:
            worker_error = WorkerExecutionError(self._model_id, str(error))
            worker_error.__cause__ = error
            result = None

        completion = Completion(
            frame=frame,
            model_id=self._model_id,
            result=result,
            elapsed_ms=elapsed_ms,
            rng_seed=rng_seed,
            error=worker_error,
        )
        try:
            with self._completion_lock:
                self._completion = completion
        finally:
            self._forward_lock.release()
