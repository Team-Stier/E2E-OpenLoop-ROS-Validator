"""One-running-forward worker with a one-slot completion mailbox."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock, Thread
from time import perf_counter_ns
from typing import Any, Optional

from validator_inference.models.contracts import ModelPrediction


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    reason: str


class WorkerExecutionError(RuntimeError):
    """A source-native model forward failed inside the worker."""

    def __init__(self, model_id: str, message: str) -> None:
        self.model_id = model_id
        super().__init__("model {} forward failed: {}".format(model_id, message))


@dataclass(frozen=True)
class Completion:
    frame: Any
    model_id: str
    result: Optional[ModelPrediction]
    elapsed_ms: float
    rng_seed: int
    error: Optional[WorkerExecutionError]


def seed_from(run_id: str, model_id: str, reference_stamp_ns: int) -> int:
    """Return the profile-defined stable, unsigned 64-bit diffusion seed."""
    raw = "{}:{}:{}".format(model_id, int(reference_stamp_ns), run_id).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _stamp_ns(stamp: Any) -> int:
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    raise TypeError("reference stamp must provide nanoseconds")


def _model_id(model: Any) -> str:
    selected_profile = getattr(model, "selected_profile", None)
    value = getattr(selected_profile, "model_id", None)
    if value is None:
        value = getattr(model, "model_id", None)
    if not value:
        raise ValueError("inference model must expose selected_profile.model_id")
    return str(value)


class InferenceWorker:
    """Execute at most one model forward and never queue rejected work."""

    def __init__(self, model: Any, run_id: str) -> None:
        self._model = model
        self._model_id = _model_id(model)
        self._run_id = str(run_id)
        self._forward_lock = Lock()
        self._completion_lock = Lock()
        self._completion = None  # type: Optional[Completion]

    def try_submit(self, frame: Any) -> SubmitResult:
        if not self._forward_lock.acquire(blocking=False):
            return SubmitResult(False, "busy_tick_skipped")
        # Keep the one-slot completion mailbox lossless.  The engine can take
        # a completion just before this call while the previous forward then
        # finishes; in that interleaving the forward lock is free, but its
        # completion still has to be published before another forward may run.
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
        with self._completion_lock:
            completion = self._completion
            self._completion = None
            return completion

    def _run_forward(self, frame: Any) -> None:
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
