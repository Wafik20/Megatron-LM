# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math

import pytest

from megatron.energy import EnergyTracker, PowerModel


def test_power_model_defaults():
    power_model = PowerModel()

    assert power_model.get_power("forward") == 280.0
    assert power_model.get_power("backward") == 320.0
    assert power_model.get_power("optimizer") == 250.0
    assert power_model.get_power("comm") == 180.0
    assert power_model.get_power("idle") == 70.0


def test_power_model_unknown_phase_raises():
    power_model = PowerModel()

    with pytest.raises(ValueError, match="Unknown phase: warmup"):
        power_model.get_power("warmup")


def test_energy_tracker_summary_and_reset():
    tracker = EnergyTracker(PowerModel())

    tracker.record_phase_time("forward", 0.2)
    tracker.record_phase_time("forward", 0.1)
    tracker.record_phase_time("backward", 0.6)
    tracker.record_phase_time("optimizer", 0.1)
    tracker.record_phase_time("comm", 0.2)

    breakdown = tracker.compute_energy_breakdown()
    summary = tracker.get_summary()

    assert tracker.compute_total_time() == pytest.approx(1.2)
    assert breakdown == {
        "forward": pytest.approx(84.0),
        "backward": pytest.approx(192.0),
        "optimizer": pytest.approx(25.0),
        "comm": pytest.approx(36.0),
        "idle": pytest.approx(0.0),
    }
    assert tracker.compute_total_energy() == 337.0
    assert tracker.compute_average_power() == pytest.approx(337.0 / 1.2)
    assert set(summary) == {
        "phase_times",
        "phase_energy",
        "total_energy_j",
        "total_time_s",
        "avg_power_w",
    }
    assert summary["phase_times"] == {
        "forward": pytest.approx(0.3),
        "backward": pytest.approx(0.6),
        "optimizer": pytest.approx(0.1),
        "comm": pytest.approx(0.2),
        "idle": pytest.approx(0.0),
    }
    assert summary["phase_energy"] == breakdown
    assert summary["total_energy_j"] == 337.0
    assert summary["total_time_s"] == pytest.approx(1.2)
    assert summary["avg_power_w"] == pytest.approx(337.0 / 1.2)

    tracker.reset()
    assert tracker.phase_times == {
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "comm": 0.0,
        "idle": 0.0,
    }
    assert tracker.compute_total_energy() == 0.0
    assert tracker.compute_total_time() == 0.0
    assert tracker.compute_average_power() == 0.0


@pytest.mark.parametrize("seconds", [-0.1, math.nan, math.inf, -math.inf])
def test_energy_tracker_invalid_duration_raises(seconds):
    tracker = EnergyTracker(PowerModel())

    with pytest.raises(ValueError, match="Invalid phase duration"):
        tracker.record_phase_time("forward", seconds)


def test_energy_tracker_invalid_phase_raises():
    tracker = EnergyTracker(PowerModel())

    with pytest.raises(ValueError, match="Unknown phase: warmup"):
        tracker.record_phase_time("warmup", 0.1)
