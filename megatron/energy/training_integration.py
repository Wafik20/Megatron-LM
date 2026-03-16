# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for integrating timer-based energy estimation into training."""

import csv
from pathlib import Path

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from .energy_tracker import EnergyTracker

COMM_TIMERS = (
    "all-grads-sync",
    "params-all-gather",
    "forward-recv",
    "forward-send",
    "backward-recv",
    "backward-send",
    "forward-send-forward-recv",
    "forward-send-backward-recv",
    "backward-send-forward-recv",
    "backward-send-backward-recv",
    "forward-backward-send-forward-backward-recv",
)

TRACKED_TIMERS = ("interval-time", "forward-compute", "backward-compute", "optimizer", *COMM_TIMERS)


class TrainingEnergyTracker:
    """Translate Megatron timer deltas into per-iteration energy summaries."""

    def __init__(
        self,
        energy_tracker: EnergyTracker,
        csv_path: str | Path | None = None,
        simulated_gpu_count: int | None = None,
    ) -> None:
        self.energy_tracker = energy_tracker
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.simulated_gpu_count = simulated_gpu_count
        self._previous_timer_totals = {name: 0.0 for name in TRACKED_TIMERS}

    def prime(self, timers) -> None:
        """Capture the current timer totals as the baseline for future iteration deltas."""
        self._previous_timer_totals = {
            name: self._get_timer_active_time(timers, name) for name in TRACKED_TIMERS
        }

    def start_iteration(self) -> None:
        """Reset the per-iteration energy accumulator."""
        self.energy_tracker.reset()

    def record_iteration(self, timers, iteration: int | None = None) -> dict[str, dict[str, float] | float]:
        """Compute a per-iteration summary from Megatron timer deltas."""
        timer_deltas = {}
        for name in TRACKED_TIMERS:
            current_total = self._get_timer_active_time(timers, name)
            previous_total = self._previous_timer_totals.get(name, 0.0)
            timer_deltas[name] = max(0.0, current_total - previous_total)
            self._previous_timer_totals[name] = current_total

        iteration_time = timer_deltas["interval-time"]
        forward_time = timer_deltas["forward-compute"]
        backward_time = timer_deltas["backward-compute"]
        optimizer_time = timer_deltas["optimizer"]
        explicit_comm_time = sum(timer_deltas[name] for name in COMM_TIMERS)

        remaining_time = max(0.0, iteration_time - (forward_time + backward_time + optimizer_time))
        comm_time = min(remaining_time, explicit_comm_time)
        idle_time = max(0.0, remaining_time - comm_time)

        self.energy_tracker.record_phase_time("forward", forward_time)
        self.energy_tracker.record_phase_time("backward", backward_time)
        self.energy_tracker.record_phase_time("optimizer", optimizer_time)
        self.energy_tracker.record_phase_time("comm", comm_time)
        self.energy_tracker.record_phase_time("idle", idle_time)

        summary = self.energy_tracker.get_summary()
        actual_per_gpu_power_w = self._gather_per_gpu_power(summary["avg_power_w"])
        summary["actual_per_gpu_power_w"] = actual_per_gpu_power_w
        summary["per_gpu_power_w"] = actual_per_gpu_power_w
        if self.simulated_gpu_count is not None:
            simulated_per_gpu_power_w = self._build_simulated_per_gpu_power(
                actual_per_gpu_power_w,
                self.simulated_gpu_count,
            )
            summary["simulated_gpu_count"] = self.simulated_gpu_count
            summary["per_gpu_power_w"] = simulated_per_gpu_power_w
            summary["simulated_total_power_w"] = sum(simulated_per_gpu_power_w)
            summary["simulated_total_energy_j"] = (
                summary["total_time_s"] * summary["simulated_total_power_w"]
            )

        if iteration is not None and self.csv_path is not None:
            self._append_csv_row(iteration, summary)

        return summary

    @staticmethod
    def format_log(iteration: int, summary: dict[str, dict[str, float] | float]) -> str:
        """Format a compact energy log line for stdout."""
        phase_energy = summary["phase_energy"]
        log_line = (
            f"[Energy] iter {iteration} | "
            f"total_energy={summary['total_energy_j']:.1f} J | "
            f"avg_power={summary['avg_power_w']:.1f} W | "
            f"fwd={phase_energy['forward']:.1f} J | "
            f"bwd={phase_energy['backward']:.1f} J | "
            f"comm={phase_energy['comm']:.1f} J"
        )
        if "simulated_gpu_count" in summary:
            log_line += (
                f" | simulated_gpus={summary['simulated_gpu_count']} | "
                f"sim_total_energy={summary['simulated_total_energy_j']:.1f} J | "
                f"sim_total_power={summary['simulated_total_power_w']:.1f} W"
            )
        return log_line

    @staticmethod
    def _get_timer_active_time(timers, name: str) -> float:
        timer = getattr(timers, "_timers", {}).get(name)
        if timer is None:
            return 0.0
        return timer.active_time()

    @staticmethod
    def _gather_per_gpu_power(avg_power_w: float) -> list[float]:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return [avg_power_w]

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, avg_power_w)
        return [float(value) for value in gathered]

    @staticmethod
    def _build_simulated_per_gpu_power(
        actual_per_gpu_power_w: list[float], simulated_gpu_count: int
    ) -> list[float]:
        if simulated_gpu_count <= len(actual_per_gpu_power_w):
            return actual_per_gpu_power_w[:simulated_gpu_count]

        target_total_power = sum(actual_per_gpu_power_w) * (
            simulated_gpu_count / len(actual_per_gpu_power_w)
        )
        master_power = actual_per_gpu_power_w[0]
        worker_profiles = actual_per_gpu_power_w[1:] or [master_power]

        simulated_worker_powers = [
            worker_profiles[index % len(worker_profiles)] for index in range(simulated_gpu_count - 1)
        ]
        worker_total_power = sum(simulated_worker_powers)
        if worker_total_power > 0.0:
            worker_scale = max(0.0, (target_total_power - master_power) / worker_total_power)
            simulated_worker_powers = [power * worker_scale for power in simulated_worker_powers]

        return [master_power, *simulated_worker_powers]

    def _append_csv_row(self, iteration: int, summary: dict[str, dict[str, float] | float]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        phase_times = summary["phase_times"]
        phase_energy = summary["phase_energy"]
        row = {
            "iteration": iteration,
            "forward_time": phase_times["forward"],
            "backward_time": phase_times["backward"],
            "optimizer_time": phase_times["optimizer"],
            "comm_time": phase_times["comm"],
            "idle_time": phase_times["idle"],
            "forward_energy": phase_energy["forward"],
            "backward_energy": phase_energy["backward"],
            "optimizer_energy": phase_energy["optimizer"],
            "comm_energy": phase_energy["comm"],
            "idle_energy": phase_energy["idle"],
            "total_energy": summary["total_energy_j"],
            "simulated_gpu_count": summary.get("simulated_gpu_count"),
            "simulated_total_energy": summary.get("simulated_total_energy_j"),
            "simulated_total_power": summary.get("simulated_total_power_w"),
            "master_gpu_power": summary["per_gpu_power_w"][0],
        }
        for index, power in enumerate(summary["per_gpu_power_w"], start=1):
            row[f"power_gpu{index}"] = power
        should_write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row))
            if should_write_header:
                writer.writeheader()
            writer.writerow(row)
