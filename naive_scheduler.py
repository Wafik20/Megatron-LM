#!/usr/bin/env python3
"""
naive_scheduler.py
------------------
Offline bucket-based carbon-aware elastic parallelism scheduler.

This intentionally ignores slack and deadlines. It assumes the full carbon
forecast is known a priori, splits that forecast into low / medium / high
carbon buckets, and runs the matching low / medium / high parallel strategy
for the bucket the run is currently in.

Bucket index is derived from training progress (step / total_steps), not
wall-clock time, so the full CSV is replayed exactly once across the run
regardless of actual hardware throughput. A forward-looking dwell window
(``min_dwell_steps``) smooths the CI signal so brief dips/spikes don't
trigger resharding more expensive than the savings.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Strategy:
    tp: int
    pp: int
    power_w: float
    step_time_s: float

    @property
    def energy_per_step_j(self) -> float:
        return self.power_w * self.step_time_s

    @property
    def key(self) -> Tuple[int, int]:
        return (self.tp, self.pp)


class NaiveScheduler:
    """Naive forecast-bucket scheduler.

    ``strategies`` must contain exactly three entries ordered as:
      1. low-carbon strategy
      2. medium-carbon strategy
      3. high-carbon strategy

    Bucket assignment is done by averaging CI over a forward window of
    ``min_dwell_steps`` steps starting from the current step, then
    classifying the average against the tertile thresholds derived from
    the full forecast. ``min_dwell_steps == 0`` reduces to a single-row
    lookup (the previous behavior).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __init__(
        self,
        strategies: List[Strategy],
        ci_forecast_hourly_gco2_per_kwh: List[float],
        total_steps: int,
        initial_strategy_idx: int = 0,
        ema_alpha: float = 0.3,
        switch_time_s: float = 0.0,
        switch_power_w: float = 0.0,
        min_dwell_steps: int = 0,
    ):
        if len(strategies) != 3:
            raise ValueError(
                "strategies must contain exactly three entries: low, medium, high"
            )
        if not ci_forecast_hourly_gco2_per_kwh:
            raise ValueError("ci_forecast_hourly_gco2_per_kwh must be non-empty")
        if not 0 <= initial_strategy_idx < len(strategies):
            raise ValueError("initial_strategy_idx is out of range")
        if total_steps is None or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")

        self._strategy_by_bucket: Dict[str, Strategy] = {
            self.LOW: strategies[0],
            self.MEDIUM: strategies[1],
            self.HIGH: strategies[2],
        }
        self._ops: Dict[Tuple[int, int], Strategy] = {
            s.key: s for s in strategies
        }
        self._current: Strategy = strategies[initial_strategy_idx]

        self._ci_hourly_gco2_per_kwh = list(ci_forecast_hourly_gco2_per_kwh)
        self.total_steps = int(total_steps)
        self.min_dwell_steps = max(0, int(min_dwell_steps))
        self.ema_alpha = ema_alpha
        self.switch_time_s = switch_time_s
        self.switch_energy_j = switch_power_w * switch_time_s

        self._low_max, self._medium_max = self._bucket_thresholds(
            self._ci_hourly_gco2_per_kwh
        )

    # ────────────────────────────────────────────────────────────
    # Online updates
    # ────────────────────────────────────────────────────────────

    def observe(
        self,
        key: Tuple[int, int],
        power_w: float,
        step_time_s: float,
    ) -> None:
        """Update cached operating points for reporting/replay.

        Measurements do not affect bucket selection; the next strategy is
        determined only by the current carbon bucket. If two buckets share
        the same ``(tp, pp)`` key they are kept in sync here.
        """
        if key not in self._ops:
            return
        old = self._ops[key]
        a = self.ema_alpha
        updated = replace(
            old,
            power_w=(1 - a) * old.power_w + a * power_w,
            step_time_s=(1 - a) * old.step_time_s + a * step_time_s,
        )
        self._ops[key] = updated
        for bucket, strategy in self._strategy_by_bucket.items():
            if strategy.key == key:
                self._strategy_by_bucket[bucket] = updated
        if self._current.key == key:
            self._current = updated

    def update_forecast(
        self,
        ci_hourly_gco2_per_kwh: List[float],
    ) -> None:
        """Replace the full forecast and recompute low/medium/high thresholds."""
        if not ci_hourly_gco2_per_kwh:
            raise ValueError("ci_hourly_gco2_per_kwh must be non-empty")
        self._ci_hourly_gco2_per_kwh = list(ci_hourly_gco2_per_kwh)
        self._low_max, self._medium_max = self._bucket_thresholds(
            self._ci_hourly_gco2_per_kwh
        )

    # ────────────────────────────────────────────────────────────
    # Read-only views
    # ────────────────────────────────────────────────────────────

    @property
    def current(self) -> Strategy:
        return self._current

    @property
    def strategies(self) -> List[Strategy]:
        return [
            self._strategy_by_bucket[self.LOW],
            self._strategy_by_bucket[self.MEDIUM],
            self._strategy_by_bucket[self.HIGH],
        ]

    # ────────────────────────────────────────────────────────────
    # Bucketing
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _bucket_thresholds(values: List[float]) -> Tuple[float, float]:
        sorted_values = sorted(values)
        n = len(sorted_values)
        low_idx = max(0, min(n - 1, n // 3 - 1))
        medium_idx = max(0, min(n - 1, (2 * n) // 3 - 1))
        return sorted_values[low_idx], sorted_values[medium_idx]

    def _bucket_idx_for_step(self, step: int) -> int:
        """Map a training step to a CSV row index by progress fraction."""
        n = len(self._ci_hourly_gco2_per_kwh)
        progress = max(0.0, min(1.0, step / self.total_steps))
        idx = int(progress * n)
        return max(0, min(idx, n - 1))

    def _label_for_ci(self, ci: float) -> str:
        if ci <= self._low_max:
            return self.LOW
        if ci <= self._medium_max:
            return self.MEDIUM
        return self.HIGH

    def carbon_bucket_at_step(self, step: int) -> str:
        """Return low / medium / high for the current progress position.

        Averages the forecast over ``[step, step + min_dwell_steps]`` before
        classifying, so a single-row spike or dip cannot flip the bucket.
        """
        idx_start = self._bucket_idx_for_step(step)
        if self.min_dwell_steps == 0:
            ci = self._ci_hourly_gco2_per_kwh[idx_start]
            return self._label_for_ci(ci)

        idx_end = self._bucket_idx_for_step(step + self.min_dwell_steps)
        idx_end = max(idx_end, idx_start)
        window = self._ci_hourly_gco2_per_kwh[idx_start:idx_end + 1]
        avg_ci = sum(window) / len(window)
        return self._label_for_ci(avg_ci)

    def ci_at_step(self, step: int) -> float:
        """Raw (unsmoothed) CI in gCO2 / J at the current progress position."""
        return self._ci_hourly_gco2_per_kwh[self._bucket_idx_for_step(step)] / 3.6e6

    # ────────────────────────────────────────────────────────────
    # Decision
    # ────────────────────────────────────────────────────────────

    def decide(self, step: int, wall_time_s: float = 0.0) -> Strategy:
        """Pick the strategy matching the current carbon bucket.

        ``wall_time_s`` is accepted for API compatibility with the online
        scheduler but is intentionally unused in progress-based mode.
        """
        del wall_time_s
        bucket = self.carbon_bucket_at_step(step)
        self._current = self._strategy_by_bucket[bucket]
        return self._current