#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications: online elastic scheduling driven by GreedyScheduler.

"""
online_elastic_pretrain.py
--------------------------
Online (greedy carbon-aware) sibling of elastic_pretrain.py.

Instead of consuming a fixed schedule JSON, this entry point queries a
GreedyScheduler at runtime to decide which (TP, PP) strategy to run next.
The schedule emerges from a sequence of decisions taken every
decide_every training steps, using:
  - a live NVML measurement of cluster power and the observed step time,
  - the carbon-intensity forecast supplied to the scheduler.

Single-node assumption: each rank samples its local GPU via NVML
(LOCAL_RANK == device index), and we all-reduce SUM across WORLD to get
cluster-total power.

Reuses the heavy-lifting helpers from elastic_pretrain.py unchanged
(NCCL group cleanup, microbatch reconfiguration, model/loss/forward
plumbing, parallel-state verifier, mock data, free_model_memory). Only
the outer phase-driving loop is rewritten.

Inputs (both required):
  --scheduler-config <path.json>   Strategies + scheduler scalars
  --carbon-forecast  <path.csv>    Hourly CI forecast (gCO2/kWh)

  scheduler_config.json schema:
    {
      "strategies": [
        {"tp": int, "pp": int, "power_w": float, "step_time_s": float},
        ...
      ],
      "switch_time_s":   float,
      "switch_power_w":  float,
      "deadline_s":      float,
      "lookahead_steps": int,
      "decide_every":    int
    }

  carbon-forecast CSV (matches the synthetic generator):
    columns: year, month, day, time, carbon
    The 'carbon' column is read in row order; row 0 is treated as
    wall-time t=0 (training start). Trim or shift the CSV upstream
    if you want a different anchor.

Outputs in --elastic-work-dir:
  phases.json            — per-phase log of what was actually executed
  elastic_results.json   — summary, same shape as elastic_pretrain.py
"""

import argparse
import csv
import gc
import json
import os
import sys
import threading
import time
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

# Local Megatron-LM modules (same dir as this file).
from gpt_builders import gpt_builder
from model_provider import model_provider

# Stable helpers from the original elastic_pretrain.py — reused as-is to
# avoid duplicating the careful work already done there. If those helpers
# move, update this import.
from elastic_pretrain import (
    destroy_model_parallel_with_nccl_cleanup,
    reconfigure_microbatches_for_phase,
    get_batch,
    loss_func,
    forward_step,
    infinite_data_iterator,
    get_loss_for_logging,
    verify_parallel_state,
    free_model_memory,
)

# Online scheduler. greedy_scheduler.py must be importable — easiest is
# to drop it next to this file in the Megatron-LM directory.
from greedy_scheduler import GreedyScheduler, Strategy


# ═══════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════

def load_scheduler_config(path: str) -> Dict[str, Any]:
    """Load and validate the scheduler JSON config.

    Returns a dict with parsed/typed values:
      strategies      : List[Strategy]
      switch_time_s   : float
      switch_power_w  : float
      deadline_s      : float
      lookahead_steps : int
      decide_every    : int

    Fails loudly on missing fields or wrong types — there's no sensible
    default for things like the deadline, so we'd rather hard-exit than
    train against silent fallbacks.
    """
    with open(path) as f:
        raw = json.load(f)

    required_top = ['strategies', 'switch_time_s', 'switch_power_w',
                    'deadline_s', 'lookahead_steps', 'decide_every']
    missing = [k for k in required_top if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")

    raw_strats = raw['strategies']
    if not isinstance(raw_strats, list) or not raw_strats:
        raise ValueError(f"{path}: 'strategies' must be a non-empty list")

    strategies: List[Strategy] = []
    for i, s in enumerate(raw_strats):
        for k in ('tp', 'pp', 'power_w', 'step_time_s'):
            if k not in s:
                raise ValueError(
                    f"{path}: strategy index {i} missing required key '{k}'"
                )
        strategies.append(Strategy(
            tp=int(s['tp']),
            pp=int(s['pp']),
            power_w=float(s['power_w']),
            step_time_s=float(s['step_time_s']),
        ))

    # Defensive: reject duplicate (tp, pp) entries — the scheduler
    # indexes operating points by key, so duplicates would silently
    # collapse and the user wouldn't see all candidates listed.
    keys = [s.key for s in strategies]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (tp, pp) entries in 'strategies'")

    return {
        'strategies':      strategies,
        'switch_time_s':   float(raw['switch_time_s']),
        'switch_power_w':  float(raw['switch_power_w']),
        'deadline_s':      float(raw['deadline_s']),
        'lookahead_steps': int(raw['lookahead_steps']),
        'decide_every':    int(raw['decide_every']),
    }


def load_carbon_forecast(path: str) -> List[float]:
    """Load hourly CI forecast from the synthetic-generator CSV format.

    Expected columns: year, month, day, time, carbon  (extras ignored).
    Returns the 'carbon' column as a list of floats in row order. The
    timestamp columns are not used here — row 0 is t=0 by convention.
    """
    values: List[float] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'carbon' not in reader.fieldnames:
            raise ValueError(
                f"{path}: missing 'carbon' column "
                f"(found: {reader.fieldnames})"
            )
        for row_idx, row in enumerate(reader):
            try:
                values.append(float(row['carbon']))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"{path} row {row_idx}: bad 'carbon' value {row.get('carbon')!r} ({e})"
                )

    if not values:
        raise ValueError(f"{path}: no rows found")
    return values


# ═══════════════════════════════════════════════════════════════
# NVML in-process power sampler (single-node, one rank per local GPU)
# ═══════════════════════════════════════════════════════════════

class NvmlPowerSampler:
    """Background thread that polls NVML for the local GPU's power.

    Each rank samples the device given by LOCAL_RANK. consume_mean()
    returns the mean since the last call and clears the buffer, so a
    decision-boundary read naturally aggregates the *just-finished*
    measurement window.

    Thread safety: a single threading.Lock guards the sample buffer.
    Sample reads are atomic floats so contention is negligible at
    sub-Hz sampling rates.
    """

    def __init__(self, gpu_index: int, sample_interval_s: float = 0.5):
        try:
            import pynvml
        except ImportError as e:
            raise RuntimeError(
                "pynvml is required for NvmlPowerSampler. "
                "Install with `pip install nvidia-ml-py` (already present "
                "in standard NeMo / Megatron containers)."
            ) from e

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._gpu_index = gpu_index

        self.sample_interval_s = sample_interval_s
        self._lock = threading.Lock()
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        pynvml = self._pynvml
        while not self._stop.is_set():
            try:
                # NVML returns power in milliwatts.
                p_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                with self._lock:
                    self._samples.append(p_w)
            except pynvml.NVMLError:
                # Transient driver hiccup — drop this sample, keep going.
                pass
            self._stop.wait(self.sample_interval_s)

    def consume_mean(self) -> float:
        """Mean power since the last call, in watts. Empties the buffer.
        Returns 0.0 if no samples have been collected (e.g., very short
        window) — caller should treat 0.0 as "no measurement"."""
        with self._lock:
            samples = self._samples
            self._samples = []
        return (sum(samples) / len(samples)) if samples else 0.0

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


def sample_cluster_power(sampler: NvmlPowerSampler) -> float:
    """All-reduce per-rank mean power across WORLD to get cluster total.

    On single-node multi-GPU setups, this is the cluster wattage during
    the just-finished window. If a rank produced no samples (e.g.,
    sub-sample-interval window), it contributes 0 W — the sum is still
    a valid lower bound but slightly under-counts. With decide_every=50
    and seconds-per-step >= 1s, every rank gets >= 50 samples per
    window, so this is not a practical concern.
    """
    local_mean = sampler.consume_mean()
    t = torch.tensor([local_mean], device=torch.cuda.current_device(),
                     dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ═══════════════════════════════════════════════════════════════
# Strategy validation (replaces load_and_validate_schedule)
# ═══════════════════════════════════════════════════════════════

def validate_strategies_or_die(strategies: List[Strategy], n_gpus: int,
                               num_layers: int, num_attention_heads: int,
                               num_query_groups: Optional[int],
                               rank: int) -> None:
    """Run all the divisibility checks that load_and_validate_schedule
    would have run, but applied to *every* strategy the scheduler might
    pick — since with online scheduling we don't know the schedule ahead
    of time. Any infeasible strategy is a hard exit; we'd rather fail
    loudly upfront than mid-run after a reshard."""
    failures = []
    for s in strategies:
        errs = []
        if n_gpus % (s.tp * s.pp) != 0:
            errs.append(f"TP={s.tp}*PP={s.pp} doesn't divide n_gpus={n_gpus}")
        if num_layers % s.pp != 0:
            errs.append(f"num_layers={num_layers} not divisible by PP={s.pp}")
        if num_attention_heads % s.tp != 0:
            errs.append(
                f"num_attention_heads={num_attention_heads} "
                f"not divisible by TP={s.tp}"
            )
        if num_query_groups is not None and num_query_groups % s.tp != 0:
            errs.append(
                f"num_query_groups={num_query_groups} "
                f"not divisible by TP={s.tp}"
            )
        if errs:
            failures.append((s, errs))

    if failures:
        if rank == 0:
            print("[validate] Strategy menu has infeasible entries:",
                  flush=True)
            for s, errs in failures:
                for e in errs:
                    print(f"  TP={s.tp} PP={s.pp}: {e}", flush=True)
        sys.exit(1)


def update_args_for_strategy(args, strategy: Strategy, ckpt_dir: str,
                             is_first_phase: bool) -> None:
    """Strategy-keyed counterpart to elastic_pretrain.update_args_for_phase."""
    args.tensor_model_parallel_size = strategy.tp
    args.pipeline_model_parallel_size = strategy.pp
    args.save = ckpt_dir
    args.save_interval = 999999

    if is_first_phase:
        args.load = None
        args.iteration = 0
        args.consumed_train_samples = 0
    else:
        args.load = ckpt_dir


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. Pull elastic-only args off sys.argv before Megatron sees it ──
    elastic_parser = argparse.ArgumentParser(add_help=False)
    elastic_parser.add_argument('--elastic-work-dir',
                                default='/tmp/elastic_training')
    elastic_parser.add_argument('--scheduler-config', required=True,
                                help='Path to JSON with strategies + scalars')
    elastic_parser.add_argument('--carbon-forecast', required=True,
                                help='Path to CSV with hourly CI forecast '
                                     '(columns: year,month,day,time,carbon)')
    elastic_args, megatron_argv = elastic_parser.parse_known_args()

    ckpt_dir = os.path.join(elastic_args.elastic_work_dir, 'elastic_ckpt')

    # Online scheduling assumes any (tp, pp) in strategies is reachable,
    # so we always need fully-reshardable optimizer checkpoints. That
    # forces the same ~10 GB allgather cost that the offline path only
    # paid for actually-reshardable schedules; keep that in mind when
    # choosing model size vs. GPU memory budget.
    extra_megatron = [
        '--ckpt-format', 'torch_dist',
        '--use-distributed-optimizer',
        '--auto-detect-ckpt-format',
        '--override-opt_param-scheduler',
        '--save', ckpt_dir,
        '--save-interval', '999999',
        '--dist-ckpt-optim-fully-reshardable',
    ]
    sys.argv = [sys.argv[0]] + megatron_argv + extra_megatron

    # ── 2. Load configs (cheap, before Megatron init so errors show fast) ──
    cfg = load_scheduler_config(elastic_args.scheduler_config)
    ci_hourly = load_carbon_forecast(elastic_args.carbon_forecast)

    strategies      = cfg['strategies']
    switch_time_s   = cfg['switch_time_s']
    switch_power_w  = cfg['switch_power_w']
    deadline_s      = cfg['deadline_s']
    lookahead_steps = cfg['lookahead_steps']
    decide_every    = cfg['decide_every']

    # ── 3. Initialize Megatron (one-time cost) ──────────────────
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer, train_step
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core import parallel_state as mpu
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.rerun_state_machine import RerunDataIterator

    model_provider_func = partial(model_provider, gpt_builder)

    t_init_start = time.perf_counter()
    initialize_megatron(allow_no_cuda=False)
    t_init = time.perf_counter() - t_init_start

    args = get_args()
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    total_steps = args.train_iters

    # ── 4. Validate every strategy upfront ──────────────────────
    validate_strategies_or_die(
        strategies, n_gpus,
        args.num_layers, args.num_attention_heads,
        getattr(args, 'num_query_groups', None),
        rank,
    )

    # ── 5. Find which strategy matches Megatron's initial state ─
    initial_idx = None
    for i, s in enumerate(strategies):
        if s.tp == args.tensor_model_parallel_size \
                and s.pp == args.pipeline_model_parallel_size:
            initial_idx = i
            break
    if initial_idx is None:
        if rank == 0:
            print(f"[fatal] Megatron initial state TP={args.tensor_model_parallel_size}, "
                  f"PP={args.pipeline_model_parallel_size} doesn't match any "
                  f"entry in scheduler_config 'strategies'. Add it to the menu "
                  f"or change the launcher's --tensor-model-parallel-size / "
                  f"--pipeline-model-parallel-size.", flush=True)
        sys.exit(1)

    # ── 6. Instantiate the online scheduler ─────────────────────
    sched = GreedyScheduler(
        strategies=strategies,
        switch_time_s=switch_time_s,
        switch_power_w=switch_power_w,
        total_steps=total_steps,
        deadline_s=deadline_s,
        ci_forecast_hourly_gco2_per_kwh=ci_hourly,
        lookahead_steps=lookahead_steps,
        initial_strategy_idx=initial_idx,
    )

    # ── 7. Start NVML power sampler (per-rank, local device) ────
    power_sampler = NvmlPowerSampler(gpu_index=local_rank)
    power_sampler.start()

    if rank == 0:
        print("\n" + "=" * 70)
        print("  ONLINE ELASTIC 3D PARALLELISM (greedy carbon-aware)")
        print("=" * 70)
        print(f"  GPUs:             {n_gpus}")
        print(f"  Total steps:      {total_steps}")
        print(f"  Deadline:         {deadline_s/3600:.2f} h")
        print(f"  Decide every:     {decide_every} steps")
        print(f"  Lookahead:        {lookahead_steps} steps")
        print(f"  Switch cost:      {switch_time_s}s @ {switch_power_w}W")
        print(f"  Megatron init:    {t_init:.1f}s (one-time cost)")
        print(f"  Checkpoint dir:   {ckpt_dir}")
        print(f"  Scheduler config: {elastic_args.scheduler_config}")
        print(f"  Carbon forecast:  {elastic_args.carbon_forecast} "
              f"({len(ci_hourly)} hourly buckets)")
        print(f"  Strategy menu:    {len(strategies)} candidates")
        for s in strategies:
            tag = "← initial" if s is strategies[initial_idx] else ""
            print(f"    TP={s.tp} PP={s.pp}  P≈{s.power_w}W  τ≈{s.step_time_s}s  {tag}")
        print("=" * 70 + "\n", flush=True)

    # ── 8. Drive phases until we reach total_steps ──────────────
    iteration = 0
    current = sched.current
    pending_strategy: Optional[Strategy] = None  # set by inner loop on switch
    phase_num = 0
    phase_results: List[dict] = []
    data_iter = RerunDataIterator(iter(infinite_data_iterator()))
    global_start = time.perf_counter()

    try:
        while iteration < total_steps:
            phase_num += 1
            is_first = (phase_num == 1)
            phase_start_step = iteration
            t_reconfig_start = time.perf_counter()

            if rank == 0:
                dp = n_gpus // (current.tp * current.pp)
                print(f"\n{'=' * 70}")
                print(f"  PHASE {phase_num}: starting at step {iteration}  "
                      f"(DP={dp}, TP={current.tp}, PP={current.pp})")
                print(f"{'=' * 70}", flush=True)

            # ── 8a. Reconfig (skipped on first phase) ───────────
            if not is_first:
                if rank == 0:
                    print("  [reconfig] Destroying old parallel state...",
                          flush=True)
                t0 = time.perf_counter()
                mpu.destroy_model_parallel()
                if rank == 0:
                    print(f"  [reconfig] destroy_model_parallel: "
                          f"{time.perf_counter()-t0:.2f}s", flush=True)

                update_args_for_strategy(args, current, ckpt_dir,
                                         is_first_phase=False)

                if rank == 0:
                    print(f"  [reconfig] Reinitializing model parallel: "
                          f"TP={current.tp}, PP={current.pp}", flush=True)
                t0 = time.perf_counter()
                dist.barrier()
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=current.tp,
                    pipeline_model_parallel_size=current.pp,
                )
                if rank == 0:
                    print(f"  [reconfig] initialize_model_parallel: "
                          f"{time.perf_counter()-t0:.2f}s", flush=True)

                reconfigure_microbatches_for_phase(args, rank)
            else:
                update_args_for_strategy(args, current, ckpt_dir,
                                         is_first_phase=True)

            # ── 8b. Build model + optimizer ─────────────────────
            if rank == 0:
                print("  [build] Constructing model and optimizer...",
                      flush=True)
            t0 = time.perf_counter()
            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                model_provider_func, ModelType.encoder_or_decoder
            )
            t_build = time.perf_counter() - t0

            # Megatron may advance args.iteration on checkpoint load.
            iteration = getattr(args, 'iteration', iteration)
            if rank == 0:
                print(f"  [build] Model + optimizer + checkpoint: {t_build:.2f}s")
                print(f"  [build] Resuming from iteration: {iteration}")
                alloc = torch.cuda.memory_allocated() / 1e9
                print(f"  [build] GPU memory allocated: {alloc:.2f}GB",
                      flush=True)

            verify_parallel_state(
                model,
                expected_tp=current.tp,
                expected_pp=current.pp,
                expected_dp=n_gpus // (current.tp * current.pp),
                rank=rank,
            )

            t_reconfig = time.perf_counter() - t_reconfig_start

            # Flush the sampler's buffer — it accumulated samples during
            # the (mostly idle) reconfig, which would skew the first
            # window's mean if we read it now.
            power_sampler.consume_mean()

            forward_backward_func = get_forward_backward_func()
            config = core_transformer_config_from_args(args)

            # ── 8c. Inner loop: train until decision says switch ────
            t_train_start = time.perf_counter()
            t_log_window = t_train_start
            losses_this_phase: List[float] = []
            log_interval = args.log_interval

            window_start_t = time.perf_counter()
            window_start_step = iteration
            phase_observations: List[tuple] = []  # (power_w, step_time_s)
            pending_strategy = None

            if rank == 0:
                print(f"\n  [train] Starting inner loop "
                      f"(decide every {decide_every} steps)...", flush=True)

            while iteration < total_steps:
                (loss_dict, skipped_iter, should_checkpoint, should_exit,
                 exit_code, grad_norm, num_zeros,
                 max_attn_logit) = train_step(
                    forward_step, data_iter, model, optimizer,
                    opt_param_scheduler, config, forward_backward_func,
                )
                iteration += 1

                lm_loss = get_loss_for_logging(loss_dict)
                losses_this_phase.append(lm_loss)

                if rank == 0 and iteration % log_interval == 0:
                    t_now = time.perf_counter()
                    iter_ms = (t_now - t_log_window) * 1000.0 / log_interval
                    t_log_window = t_now
                    avg_recent = (sum(losses_this_phase[-log_interval:])
                                  / min(log_interval, len(losses_this_phase)))
                    gn = grad_norm if grad_norm is not None else 0.0
                    skip = " [SKIPPED]" if skipped_iter else ""
                    print(f"    iter {iteration:>6}/{total_steps} | "
                          f"loss: {avg_recent:.4f} | grad_norm: {gn:.4f} | "
                          f"{iter_ms:5.0f}ms/it{skip}", flush=True)

                # Decision boundary?
                steps_in_window = iteration - window_start_step
                at_decision = (
                    steps_in_window >= decide_every
                    or iteration >= total_steps
                )
                if not at_decision:
                    continue

                # Observe what we just did. Both measurements are
                # cluster-aggregate by construction (NVML sum across
                # ranks; wall time is the same on every rank).
                window_dur = time.perf_counter() - window_start_t
                measured_step_time = window_dur / steps_in_window
                measured_power = sample_cluster_power(power_sampler)
                phase_observations.append((measured_power, measured_step_time))
                sched.observe(current.key, measured_power, measured_step_time)

                if rank == 0:
                    print(f"    [observe] window {steps_in_window} steps:  "
                          f"{measured_power:>6.1f}W cluster,  "
                          f"{measured_step_time:.3f}s/step", flush=True)

                # Decide. Skip the decision if we've already finished —
                # there's no next phase to schedule.
                if iteration >= total_steps:
                    break

                wall_time = time.perf_counter() - global_start
                next_strat = sched.decide(iteration, wall_time)

                if next_strat.key != current.key:
                    if rank == 0:
                        print(f"    [decide] SWITCH "
                              f"TP={current.tp}/PP={current.pp} → "
                              f"TP={next_strat.tp}/PP={next_strat.pp} "
                              f"@ step {iteration} "
                              f"(wall {wall_time/3600:.2f}h)", flush=True)
                    pending_strategy = next_strat
                    break

                if rank == 0:
                    print(f"    [decide] stay on "
                          f"TP={current.tp}/PP={current.pp}", flush=True)

                # Reset the window.
                window_start_t = time.perf_counter()
                window_start_step = iteration

            t_train = time.perf_counter() - t_train_start
            num_steps_phase = iteration - phase_start_step
            avg_iter_ms = (t_train / max(num_steps_phase, 1)) * 1000.0
            final_loss = losses_this_phase[-1] if losses_this_phase else None

            if phase_observations:
                obs_p = sum(p for p, _ in phase_observations) / len(phase_observations)
                obs_t = sum(t for _, t in phase_observations) / len(phase_observations)
            else:
                obs_p = obs_t = None

            # ── 8d. Save checkpoint iff there's a next phase ────
            is_last = (pending_strategy is None)  # we either hit total_steps or no switch
            if is_last:
                if rank == 0:
                    print(f"  [ckpt] Skipping final checkpoint "
                          f"(end of training, no next phase to load)",
                          flush=True)
                t_save = 0.0
            else:
                if rank == 0:
                    print(f"  [ckpt] Saving checkpoint at iteration {iteration}...",
                          flush=True)
                t0 = time.perf_counter()
                args.save = ckpt_dir
                save_checkpoint(iteration, model, optimizer,
                                opt_param_scheduler, 0)
                t_save = time.perf_counter() - t0
                if rank == 0:
                    print(f"  [ckpt] Saved in {t_save:.2f}s", flush=True)

            phase_results.append({
                "phase": phase_num,
                "start_step": phase_start_step,
                "end_step": iteration,
                "tp": current.tp,
                "pp": current.pp,
                "dp": n_gpus // (current.tp * current.pp),
                "num_steps": num_steps_phase,
                "reconfig_s": round(t_reconfig, 2),
                "train_s": round(t_train, 2),
                "save_s": round(t_save, 2),
                "avg_iter_ms": round(avg_iter_ms, 1),
                "observed_power_w": (round(obs_p, 1) if obs_p is not None
                                     else None),
                "observed_step_time_s": (round(obs_t, 4) if obs_t is not None
                                         else None),
                "n_observations": len(phase_observations),
                "final_loss": (round(final_loss, 6) if final_loss is not None
                               else None),
            })

            if not is_last:
                if rank == 0:
                    print(f"  [cleanup] Freeing model memory for next phase...",
                          flush=True)
                free_model_memory(model, optimizer, opt_param_scheduler)
                current = pending_strategy
            else:
                del model, optimizer, opt_param_scheduler

    finally:
        power_sampler.stop()

    # ── 9. Emit phases.json + summary results JSON ──────────────
    if rank == 0:
        os.makedirs(elastic_args.elastic_work_dir, exist_ok=True)

        phases_path = os.path.join(elastic_args.elastic_work_dir, 'phases.json')
        with open(phases_path, 'w') as f:
            json.dump(phase_results, f, indent=2)

        total_wall = sum(r['reconfig_s'] + r['train_s'] + r['save_s']
                         for r in phase_results)
        total_train = sum(r['train_s'] for r in phase_results)
        total_reconfig = sum(r['reconfig_s'] for r in phase_results)
        total_save = sum(r['save_s'] for r in phase_results)

        print(f"\n{'=' * 70}")
        print("  ONLINE ELASTIC TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Megatron init:       {t_init:.1f}s (one-time)")
        print(f"  Total wall:          {total_wall:.1f}s")
        print(f"  Total training:      {total_train:.1f}s")
        print(f"  Total reconfig:      {total_reconfig:.1f}s")
        print(f"  Total checkpoint:    {total_save:.1f}s")
        if total_wall > 0:
            print(f"  Training fraction:   "
                  f"{total_train/total_wall*100:.1f}%")
        print(f"  Phases scheduled:    {len(phase_results)}")

        print(f"\n  {'Phase':<6} {'Config':<22} {'Steps':>8} "
              f"{'Train':>8} {'Iter':>9} {'Pow(W)':>8} {'Loss':>10}")
        print("  " + "-" * 75)
        for r in phase_results:
            cfg_str = f"dp{r['dp']}_tp{r['tp']}_pp{r['pp']}"
            loss_s = f"{r['final_loss']:.4f}" if r['final_loss'] else "N/A"
            pow_s = (f"{r['observed_power_w']:.0f}"
                     if r['observed_power_w'] else "—")
            print(f"  {r['phase']:<6} {cfg_str:<22} {r['num_steps']:>8} "
                  f"{r['train_s']:>7.1f}s {r['avg_iter_ms']:>7.0f}ms/it "
                  f"{pow_s:>8} {loss_s:>10}")

        results_file = os.path.join(elastic_args.elastic_work_dir,
                                    'elastic_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'n_gpus': n_gpus,
                'megatron_init_s': round(t_init, 2),
                'total_wall_s': round(total_wall, 2),
                'total_train_s': round(total_train, 2),
                'total_reconfig_s': round(total_reconfig, 2),
                'total_save_s': round(total_save, 2),
                'total_steps': total_steps,
                'deadline_s': deadline_s,
                'decide_every': decide_every,
                'lookahead_steps': lookahead_steps,
                'switch_time_s': switch_time_s,
                'switch_power_w': switch_power_w,
                'scheduler_config_path': elastic_args.scheduler_config,
                'carbon_forecast_path': elastic_args.carbon_forecast,
                'phases': phase_results,
            }, f, indent=2)

        print(f"\n  Phases log: {phases_path}")
        print(f"  Summary:    {results_file}")
        print(f"{'=' * 70}\n", flush=True)

    dist.barrier()


if __name__ == '__main__':
    main()