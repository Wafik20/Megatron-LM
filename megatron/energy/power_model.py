# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Simple phase-based power model for energy estimation."""

_PHASES = ("forward", "backward", "optimizer", "comm", "idle")


class PowerModel:
    """Map training phases to constant power draw in watts."""

    def __init__(
        self,
        forward_power_w: float = 280,
        backward_power_w: float = 320,
        optimizer_power_w: float = 250,
        comm_power_w: float = 180,
        idle_power_w: float = 70,
    ) -> None:
        self.power_map = {
            "forward": float(forward_power_w),
            "backward": float(backward_power_w),
            "optimizer": float(optimizer_power_w),
            "comm": float(comm_power_w),
            "idle": float(idle_power_w),
        }

    def get_power(self, phase: str) -> float:
        """Return power in watts for the requested phase."""
        if phase not in self.power_map:
            raise ValueError(f"Unknown phase: {phase}")
        return self.power_map[phase]
