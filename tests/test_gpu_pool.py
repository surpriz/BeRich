"""Tests for the GPU task pool (sequential fallback + device contract)."""

from __future__ import annotations

from berich.training.gpu_pool import GpuTask, available_gpus, run_on_gpus


def _device_and_value(device: str, value: int) -> tuple[str, int]:
    return device, value * 2


def test_available_gpus_returns_list():
    gpus = available_gpus()
    assert isinstance(gpus, list)
    assert all(isinstance(g, int) for g in gpus)


def test_empty_tasks():
    assert run_on_gpus([]) == []


def test_sequential_cpu_fallback_preserves_order():
    tasks = [GpuTask(fn=_device_and_value, args=(i,)) for i in range(5)]
    results = run_on_gpus(tasks, gpu_ids=[])
    assert [v for _, v in results] == [0, 2, 4, 6, 8]
    assert all(dev == "cpu" for dev, _ in results)


def test_single_gpu_uses_that_device():
    tasks = [GpuTask(fn=_device_and_value, args=(3,))]
    results = run_on_gpus(tasks, gpu_ids=[1])
    assert results[0][0] == "cuda:1"
