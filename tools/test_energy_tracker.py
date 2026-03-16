# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone smoke test for the Megatron energy tracker."""

import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.energy import EnergyTracker, PowerModel


def main() -> None:
    tracker = EnergyTracker(PowerModel())

    tracker.record_phase_time("forward", 0.3)
    tracker.record_phase_time("backward", 0.6)
    tracker.record_phase_time("optimizer", 0.1)
    tracker.record_phase_time("comm", 0.2)
    tracker.record_phase_time("idle", 0.0)

    summary = tracker.get_summary()
    expected_energy = {
        "forward": 84.0,
        "backward": 192.0,
        "optimizer": 25.0,
        "comm": 36.0,
        "idle": 0.0,
    }

    assert summary["phase_times"] == {
        "forward": 0.3,
        "backward": 0.6,
        "optimizer": 0.1,
        "comm": 0.2,
        "idle": 0.0,
    }
    assert summary["phase_energy"] == expected_energy
    assert summary["total_energy_j"] == 337.0
    assert summary["total_time_s"] == 1.2
    assert summary["avg_power_w"] == 337.0 / 1.2

    print("Energy Summary:")
    pprint(summary)


if __name__ == "__main__":
    main()
