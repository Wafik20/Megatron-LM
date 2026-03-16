# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Utilities for converting phase timing data into energy estimates."""

import math

from .power_model import _PHASES


class EnergyTracker:
    """Accumulate phase timings and derive energy summaries."""

    def __init__(self, power_model) -> None:
        self.power_model = power_model
        self.reset()

    def reset(self) -> None:
        self.phase_times = {phase: 0.0 for phase in _PHASES}

    def record_phase_time(self, phase: str, seconds: float) -> None:
        if phase not in self.phase_times:
            raise ValueError(f"Unknown phase: {phase}")

        try:
            seconds = float(seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seconds value: {seconds}") from exc

        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"Invalid phase duration: {seconds}")

        self.phase_times[phase] += seconds

    def compute_energy_breakdown(self) -> dict[str, float]:
        energy = {}
        for phase, time_s in self.phase_times.items():
            energy[phase] = self.power_model.get_power(phase) * time_s
        return energy

    def compute_total_energy(self) -> float:
        return sum(self.compute_energy_breakdown().values())

    def compute_total_time(self) -> float:
        return sum(self.phase_times.values())

    def compute_average_power(self) -> float:
        total_time = self.compute_total_time()
        if total_time == 0.0:
            return 0.0
        return self.compute_total_energy() / total_time

    def get_summary(self) -> dict[str, dict[str, float] | float]:
        phase_energy = self.compute_energy_breakdown()
        total_energy = sum(phase_energy.values())
        total_time = self.compute_total_time()
        avg_power = 0.0 if total_time == 0.0 else total_energy / total_time

        return {
            "phase_times": dict(self.phase_times),
            "phase_energy": phase_energy,
            "total_energy_j": total_energy,
            "total_time_s": total_time,
            "avg_power_w": avg_power,
        }
