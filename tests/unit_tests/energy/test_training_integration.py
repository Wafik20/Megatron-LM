# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

import megatron.energy.training_integration as training_integration
from megatron.energy import EnergyTracker, PowerModel, TrainingEnergyTracker


class FakeTimer:
    def __init__(self, active_time=0.0):
        self._active_time = active_time

    def active_time(self):
        return self._active_time


class FakeTimers:
    def __init__(self):
        self._timers = {}

    def set_active_time(self, name, value):
        self._timers[name] = FakeTimer(value)


def test_training_energy_tracker_records_iteration_and_writes_csv(tmp_path):
    csv_path = tmp_path / "energy_log.csv"
    tracker = TrainingEnergyTracker(
        EnergyTracker(PowerModel()),
        csv_path=csv_path,
        simulated_gpu_count=16,
    )
    timers = FakeTimers()

    tracker.prime(timers)
    tracker.start_iteration()

    timers.set_active_time("interval-time", 1.2)
    timers.set_active_time("forward-compute", 0.3)
    timers.set_active_time("backward-compute", 0.6)
    timers.set_active_time("optimizer", 0.1)
    timers.set_active_time("all-grads-sync", 0.1)
    timers.set_active_time("params-all-gather", 0.1)

    summary = tracker.record_iteration(timers, iteration=1)

    assert summary["phase_times"] == {
        "forward": pytest.approx(0.3),
        "backward": pytest.approx(0.6),
        "optimizer": pytest.approx(0.1),
        "comm": pytest.approx(0.2),
        "idle": pytest.approx(0.0),
    }
    assert summary["phase_energy"] == {
        "forward": pytest.approx(84.0),
        "backward": pytest.approx(192.0),
        "optimizer": pytest.approx(25.0),
        "comm": pytest.approx(36.0),
        "idle": pytest.approx(0.0),
    }
    assert summary["total_energy_j"] == pytest.approx(337.0)
    assert summary["avg_power_w"] == pytest.approx(337.0 / 1.2)
    assert summary["actual_per_gpu_power_w"] == [pytest.approx(337.0 / 1.2)]
    assert summary["per_gpu_power_w"] == [pytest.approx(337.0 / 1.2)] * 16
    assert summary["simulated_gpu_count"] == 16
    assert summary["simulated_total_energy_j"] == pytest.approx(337.0 * 16)
    assert summary["simulated_total_power_w"] == pytest.approx((337.0 / 1.2) * 16)
    assert csv_path.exists()
    assert (
        csv_path.read_text().splitlines()[0]
        == "iteration,forward_time,backward_time,optimizer_time,comm_time,idle_time,"
        "forward_energy,backward_energy,optimizer_energy,comm_energy,idle_energy,"
        "total_energy,simulated_gpu_count,simulated_total_energy,"
        "simulated_total_power,master_gpu_power,"
        "power_gpu1,power_gpu2,power_gpu3,power_gpu4,power_gpu5,power_gpu6,power_gpu7,"
        "power_gpu8,power_gpu9,power_gpu10,power_gpu11,power_gpu12,power_gpu13,"
        "power_gpu14,power_gpu15,power_gpu16"
    )


def test_training_energy_tracker_gathers_per_gpu_power(monkeypatch):
    monkeypatch.setattr(training_integration.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_integration.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training_integration.dist, "get_world_size", lambda: 4)

    def fake_all_gather_object(output, value):
        del value
        output[:] = [100.0, 110.0, 120.0, 130.0]

    monkeypatch.setattr(training_integration.dist, "all_gather_object", fake_all_gather_object)

    gathered = TrainingEnergyTracker._gather_per_gpu_power(100.0)

    assert gathered == [100.0, 110.0, 120.0, 130.0]


def test_training_energy_tracker_builds_simulated_gpu_power_list():
    simulated = TrainingEnergyTracker._build_simulated_per_gpu_power(
        [140.0, 100.0, 110.0, 120.0],
        16,
    )

    assert len(simulated) == 16
    assert simulated[0] == pytest.approx(140.0)
    assert sum(simulated) == pytest.approx((140.0 + 100.0 + 110.0 + 120.0) * 4)


def test_training_energy_tracker_csv_header_reflects_multiple_gpu_power_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(training_integration.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_integration.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training_integration.dist, "get_world_size", lambda: 4)

    def fake_all_gather_object(output, value):
        output[:] = [value, value + 1.0, value + 2.0, value + 3.0]

    monkeypatch.setattr(training_integration.dist, "all_gather_object", fake_all_gather_object)

    csv_path = tmp_path / "energy_log.csv"
    tracker = TrainingEnergyTracker(EnergyTracker(PowerModel()), csv_path=csv_path)
    timers = FakeTimers()

    tracker.prime(timers)
    tracker.start_iteration()
    timers.set_active_time("interval-time", 1.2)
    timers.set_active_time("forward-compute", 0.3)
    timers.set_active_time("backward-compute", 0.6)
    timers.set_active_time("optimizer", 0.1)
    timers.set_active_time("all-grads-sync", 0.1)
    timers.set_active_time("params-all-gather", 0.1)

    tracker.record_iteration(timers, iteration=1)

    assert (
        csv_path.read_text().splitlines()[0]
        == "iteration,forward_time,backward_time,optimizer_time,comm_time,idle_time,"
        "forward_energy,backward_energy,optimizer_energy,comm_energy,idle_energy,"
        "total_energy,simulated_gpu_count,simulated_total_energy,simulated_total_power,"
        "master_gpu_power,power_gpu1,power_gpu2,power_gpu3,power_gpu4"
    )


def test_training_energy_tracker_assigns_remaining_time_to_idle():
    tracker = TrainingEnergyTracker(EnergyTracker(PowerModel()))
    timers = FakeTimers()

    tracker.prime(timers)
    tracker.start_iteration()

    timers.set_active_time("interval-time", 1.5)
    timers.set_active_time("forward-compute", 0.3)
    timers.set_active_time("backward-compute", 0.6)
    timers.set_active_time("optimizer", 0.1)
    timers.set_active_time("all-grads-sync", 0.2)

    summary = tracker.record_iteration(timers, iteration=1)

    assert summary["phase_times"]["comm"] == pytest.approx(0.2)
    assert summary["phase_times"]["idle"] == pytest.approx(0.3)


def test_training_energy_tracker_format_log():
    summary = {
        "phase_times": {
            "forward": 0.3,
            "backward": 0.6,
            "optimizer": 0.1,
            "comm": 0.2,
            "idle": 0.0,
        },
        "phase_energy": {
            "forward": 84.0,
            "backward": 192.0,
            "optimizer": 25.0,
            "comm": 36.0,
            "idle": 0.0,
        },
        "total_energy_j": 337.0,
        "total_time_s": 1.2,
        "avg_power_w": 280.83333333333337,
        "simulated_gpu_count": 16,
        "simulated_total_energy_j": 5392.0,
        "simulated_total_power_w": 4493.333333333334,
    }

    log_line = TrainingEnergyTracker.format_log(1, summary)

    assert "[Energy] iter 1" in log_line
    assert "total_energy=337.0 J" in log_line
    assert "avg_power=280.8 W" in log_line
    assert "fwd=84.0 J" in log_line
    assert "simulated_gpus=16" in log_line
    assert "sim_total_energy=5392.0 J" in log_line
