"""Dual-GPU task pool: one GPU per worker process, round-robin task dispatch.

The daily-equities panel is tiny, so DDP's gradient sync would cost more than it
saves. The win from two GPUs here is *throughput across many models / HPO trials*,
not faster single fits. This pool runs one worker process per GPU, pins each to a
single device via ``CUDA_VISIBLE_DEVICES`` (process isolation avoids CUDA context
cross-talk and keeps one crashing trial from poisoning the others), and hands each
task the masked device string ``"cuda:0"``.

Each task is a :class:`GpuTask` wrapping a *module-level* (picklable) callable whose
first positional argument is the device string. When fewer than two devices are
available the pool falls back to running tasks sequentially in-process on ``"cpu"``
(or the single GPU), which is also what unit tests exercise.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

# Set in each worker process by ``_init_worker`` after the GPU is pinned.
_WORKER_DEVICE = "cpu"


@dataclass
class GpuTask:
    """A unit of GPU work. ``fn`` must be importable by reference (module-level).

    ``fn`` is invoked as ``fn(device, *args, **kwargs)`` where ``device`` is the
    pinned device string (e.g. ``"cuda:0"`` inside a worker, ``"cpu"`` in fallback).
    """

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = ""


def available_gpus() -> list[int]:
    """Visible CUDA device indices (already filtered by ``CUDA_VISIBLE_DEVICES``)."""
    try:
        import torch  # noqa: PLC0415 — optional at import time; only needed when GPUs are used
    except ImportError:  # pragma: no cover - torch is a hard dep here
        return []
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _init_worker(gpu_queue: Any) -> None:
    """Worker initializer: pull one GPU id from the queue and pin to it."""
    global _WORKER_DEVICE  # noqa: PLW0603 — worker-local process global by design
    gid = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    # With the env mask set before torch initializes CUDA in this fresh process,
    # the pinned GPU is the only visible one, so it is always "cuda:0".
    _WORKER_DEVICE = "cuda:0"


def _run_task(task: GpuTask) -> Any:
    return task.fn(_WORKER_DEVICE, *task.args, **task.kwargs)


def run_on_gpus(
    tasks: Sequence[GpuTask],
    gpu_ids: Sequence[int] | None = None,
) -> list[Any]:
    """Run ``tasks`` across the given GPUs, returning results in task order.

    With 0 or 1 device, runs sequentially in-process (device ``"cpu"`` or the single
    ``"cuda:N"``). With ≥2 devices, spawns one worker per GPU and dispatches tasks
    round-robin; the pool blocks until every task completes.
    """
    if not tasks:
        return []
    ids = list(gpu_ids) if gpu_ids is not None else available_gpus()

    if len(ids) <= 1:
        device = "cpu" if not ids else f"cuda:{ids[0]}"
        logger.info("running %d task(s) sequentially on %s", len(tasks), device)
        return [task.fn(device, *task.args, **task.kwargs) for task in tasks]

    ctx = get_context("spawn")
    gpu_queue: Any = ctx.Queue()
    for gid in ids:
        gpu_queue.put(gid)

    logger.info("running %d task(s) across GPUs %s", len(tasks), ids)
    with ProcessPoolExecutor(
        max_workers=len(ids),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue,),
    ) as pool:
        return list(pool.map(_run_task, tasks))


__all__ = ["GpuTask", "available_gpus", "run_on_gpus"]
