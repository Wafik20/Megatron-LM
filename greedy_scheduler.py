#!/usr/bin/env python3
"""
greedy_scheduler.py
-------------------
Online greedy carbon-aware elastic parallelism scheduler.

The scheduler is queried during training. Each call to `decide(step,
wall_time)` returns the strategy to run next, given the latest observed
state. Training calls `observe(strategy, power_w, step_time_s)` after
each measurement window to refine the cached operating points, and
`update_forecast(...)` whenever a fresh CI forecast arrives.

Sibling of online_elastic_pretrain.py — both live in the Megatron-LM
directory together. online_elastic_pretrain.py imports GreedyScheduler
and Strategy from this module.

The forecast is treated as a list of CI buckets, each lasting
`seconds_per_forecast_bucket` seconds of wall time. In production this
is 3600.0 (one row = one real hour). For smoke tests, set it lower so
a short run traverses meaningful CI variation.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
#  STRATEGY DATA TYPE
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Strategy:
    tp: int
    pp: int
    power_w: float        # P(π): mean cluster power, watts
    step_time_s: float    # τ(π): seconds per training step

    @property
    def energy_per_step_j(self) -> float:
        return self.power_w * self.step_time_s

    @property
    def key(self) -> Tuple[int, int]:
        return (self.tp, self.pp)


# ══════════════════════════════════════════════════════════════
#  ONLINE SCHEDULER
# ══════════════════════════════════════════════════════════════

class GreedyScheduler:
    """Online greedy carbon-aware scheduler.

    State carried across calls:
      - current strategy (the one training is actively running)
      - cached operating points P(π), τ(π) for every π, refined by observe()
      - CI forecast (replaceable via update_forecast())
    """

    def __init__(
        self,
        strategies: List[Strategy],
        switch_time_s: float,
        switch_power_w: float,
        total_steps: int,
        deadline_s: float,
        ci_forecast_hourly_gco2_per_kwh: List[float],
        lookahead_steps: int = 200,
        ema_alpha: float = 0.3,
        initial_strategy_idx: int = 0,
        seconds_per_forecast_bucket: float = 3600.0,
    ):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        # Index operating points by (tp, pp) key. These are mutable —
        # observe() refines them as real measurements come in.
        self._ops: Dict[Tuple[int, int], Strategy] = {
            s.key: s for s in strategies
        }
        self._current: Strategy = strategies[initial_strategy_idx]

        self.switch_time_s = switch_time_s
        self.switch_energy_j = switch_power_w * switch_time_s
        self.total_steps = total_steps
        self.deadline_s = deadline_s

        self._ci_hourly_gco2_per_kwh: List[float] = list(
            ci_forecast_hourly_gco2_per_kwh
        )
        self._ci_t_origin_s: float = 0.0  # forecast index 0 corresponds to this wall time

        self.lookahead_steps = lookahead_steps
        self.ema_alpha = ema_alpha

        # How many wall-clock seconds each forecast bucket represents.
        # 3600.0 = production (one row = one real hour). Smaller values
        # compress the forecast onto a short run for smoke testing.
        self.seconds_per_forecast_bucket = seconds_per_forecast_bucket

    # ── State ingestion ──────────────────────────────────────

    def observe(self, key: Tuple[int, int],
                power_w: float, step_time_s: float) -> None:
        """Refine the cached operating point for strategy `key` with a
        new measurement, using exponential moving average. Called by
        training after each measurement window."""
        if key not in self._ops:
            return  # ignore unknown strategies (defensive)
        old = self._ops[key]
        a = self.ema_alpha
        self._ops[key] = replace(
            old,
            power_w=(1 - a) * old.power_w + a * power_w,
            step_time_s=(1 - a) * old.step_time_s + a * step_time_s,
        )

    def update_forecast(self, ci_hourly_gco2_per_kwh: List[float],
                        t_origin_s: float = 0.0) -> None:
        """Replace the CI forecast. `t_origin_s` is the wall-clock time
        (in the same frame the scheduler is being called with) that
        index 0 of the forecast corresponds to."""
        self._ci_hourly_gco2_per_kwh = list(ci_hourly_gco2_per_kwh)
        self._ci_t_origin_s = t_origin_s

    # ── Read-only accessors ──────────────────────────────────

    @property
    def current(self) -> Strategy:
        return self._current

    @property
    def strategies(self) -> List[Strategy]:
        return list(self._ops.values())

    def ci_at(self, t_s: float) -> float:
        """CI in gCO₂ / J at wall-clock offset t_s. Hold last value flat
        past the forecast horizon. The wall-time → bucket mapping uses
        seconds_per_forecast_bucket so smoke tests can compress real
        diurnal variation onto short runs."""
        idx = int((t_s - self._ci_t_origin_s) // self.seconds_per_forecast_bucket)
        idx = max(0, min(idx, len(self._ci_hourly_gco2_per_kwh) - 1))
        return self._ci_hourly_gco2_per_kwh[idx] / 3.6e6

    # ── Decision kernel ──────────────────────────────────────

    def _lookahead_carbon(self, target: Strategy, current: Strategy,
                          t: float, H: int) -> float:
        """C(π | H, t) — total gCO₂ over the next H steps under `target`,
        charging the switch cost upfront if target ≠ current."""
        switching = (target.key != current.key)
        cost = self.switch_energy_j * self.ci_at(t) if switching else 0.0
        t0 = t + (self.switch_time_s if switching else 0.0)
        for j in range(H):
            cost += target.energy_per_step_j * self.ci_at(
                t0 + j * target.step_time_s
            )
        return cost

    def _feasible(self, remaining_steps: int,
                  remaining_time: float) -> List[Strategy]:
        """Strategies that, used exclusively from now to the end, would
        still meet the deadline."""
        return [
            s for s in self._ops.values()
            if s.step_time_s * remaining_steps <= remaining_time
        ]

    def decide(self, step: int, wall_time_s: float) -> Strategy:
        """Pick the strategy to run for the next phase, given live state.

        Side effect: updates `self._current` to the chosen strategy, so
        subsequent calls correctly account for whether the *next* call
        would be a switch.
        """
        remaining_steps = self.total_steps - step
        remaining_time = self.deadline_s - wall_time_s

        candidates = self._feasible(remaining_steps, remaining_time)
        if not candidates:
            # Behind schedule — abandon carbon objective, just finish.
            chosen = min(self._ops.values(), key=lambda s: s.step_time_s)
        else:
            H = min(self.lookahead_steps, remaining_steps)
            chosen = min(
                candidates,
                key=lambda s: self._lookahead_carbon(s, self._current,
                                                    wall_time_s, H),
            )

        self._current = chosen
        return chosen

    # ── Replay (for offline analysis / reporting) ────────────

    def replay_carbon(self, schedule: List[dict]) -> Tuple[float, float]:
        """Sum gCO₂ and wall time for a given phase schedule, using the
        current cached operating points."""
        total_gco2 = 0.0
        t = 0.0
        prev_key: Optional[Tuple[int, int]] = None

        for phase in schedule:
            key = (phase["tp"], phase["pp"])
            s = self._ops[key]
            n_steps = phase["end_step"] - phase["start_step"]

            if prev_key is not None and key != prev_key:
                total_gco2 += self.switch_energy_j * self.ci_at(t)
                t += self.switch_time_s

            for j in range(n_steps):
                total_gco2 += s.energy_per_step_j * self.ci_at(
                    t + j * s.step_time_s
                )
            t += n_steps * s.step_time_s
            prev_key = key

        return total_gco2, t