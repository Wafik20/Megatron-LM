#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications: online elastic scheduling driven by GreedyScheduler.

"""
online_elastic_pretrain.py
--------------------------
Online (greedy carbon-aware) sibling of elastic_pretrain.py.

Inputs (required):
  --scheduler-config <path.json>   Calibration JSON from profile_strategies.py
                                   (provides strategies + average_reshard_wall_time_s)
  --carbon-forecast  <path.csv>    Hourly CI forecast (gCO2/kWh)

Policy parameters (required, from CLI or JSON):
  --switch-power-w     <float>     Power draw during reshard window (W)
  --deadline-seconds   <float>     Total training deadline (s)
  --lookahead-steps    <int>       Greedy lookahead horizon (steps)
  --decide-every       <int>       Decision interval (steps)

Optional overrides:
  --switch-time-seconds <float>    Override JSON's average_reshard_wall_time_s
  --bucket-seconds      <float>    Wall seconds per CI forecast row (default 3600)

The scheduler-config JSON minimally needs:
  {
    "strategies": [
      {"tp": int, "pp": int, "power_w": float, "step_time_s": float, ...},
      ...
    ],
    "average_reshard_wall_time_s": float    (used as switch_time_s)
  }

Extra fields in the calibration (dp, warmup_iters, save_s, mpu_reshard_s,
etc.) are ignored. Policy fields (switch_power_w, deadline_s,
lookahead_steps, decide_every, seconds_per_forecast_bucket) MAY also be
embedded in the JSON for convenience — CLI args take precedence when
both are present. This separation lets the same calibration drive
multiple comparison runs (greedy / fast-only / slow-only) by varying
only the CLI flags.
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

from gpt_builders import gpt_builder
from model_provider import model_provider

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

from greedy_scheduler import GreedyScheduler, Strategy


# ═══════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════

def load_scheduler_config(path: str) -> Dict[str, Any]:
    """Load calibration/scheduler JSON.

    Accepts the output of profile_strategies.py, which has:
      - strategies: list of {tp, pp, power_w, step_time_s, ...}
      - average_reshard_wall_time_s: used as switch_time_s if no CLI override

    Policy fields (switch_power_w, deadline_s, lookahead_steps, decide_every,
    seconds_per_forecast_bucket) MAY be embedded for convenience but are
    optional here — main() merges them with CLI args. Returned values are
    None when absent so the caller can detect missing fields.
    """
    with open(path) as f:
        raw = json.load(f)

    raw_strats = raw.get('strategies')
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

    keys = [s.key for s in strategies]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (tp, pp) entries in 'strategies'")

    # switch_time_s: prefer 'average_reshard_wall_time_s' (output of the
    # profiler), fall back to 'switch_time_s' for backwards compatibility
    # with older hand-written scheduler configs.
    switch_time_s = raw.get('average_reshard_wall_time_s',
                            raw.get('switch_time_s'))

    return {
        'strategies':                  strategies,
        # All scalar fields below may be None — main() resolves against CLI.
        'switch_time_s':               (float(switch_time_s)
                                        if switch_time_s is not None else None),
        'switch_power_w':              (float(raw['switch_power_w'])
                                        if 'switch_power_w' in raw else None),
        'deadline_s':                  (float(raw['deadline_s'])
                                        if 'deadline_s' in raw else None),
        'lookahead_steps':             (int(raw['lookahead_steps'])
                                        if 'lookahead_steps' in raw else None),
        'decide_every':                (int(raw['decide_every'])
                                        if 'decide_every' in raw else None),
        'seconds_per_forecast_bucket': (float(raw['seconds_per_forecast_bucket'])
                                        if 'seconds_per_forecast_bucket' in raw
                                        else None),
    }


def load_carbon_forecast(path: str) -> List[float]:
    """Load hourly CI forecast from the synthetic-generator CSV format."""
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
    """Background thread that polls NVML for the local GPU's power."""

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
                p_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                with self._lock:
                    self._samples.append(p_w)
            except pynvml.NVMLError:
                pass
            self._stop.wait(self.sample_interval_s)

    def consume_mean(self) -> float:
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
    """All-reduce per-rank mean power across WORLD to get cluster total."""
    local_mean = sampler.consume_mean()
    t = torch.tensor([local_mean], device=torch.cuda.current_device(),
                     dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ═══════════════════════════════════════════════════════════════
# Strategy validation
# ═══════════════════════════════════════════════════════════════

def validate_strategies_or_die(strategies: List[Strategy], n_gpus: int,
                               num_layers: int, num_attention_heads: int,
                               num_query_groups: Optional[int],
                               rank: int) -> None:
    """Run divisibility checks against every strategy in the menu."""
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
                                help='Path to calibration JSON from '
                                     'profile_strategies.py')
    elastic_parser.add_argument('--carbon-forecast', required=True,
                                help='Path to CSV with hourly CI forecast '
                                     '(columns: year,month,day,time,carbon)')
    # Policy parameters — required (CLI or JSON), CLI takes precedence.
    elastic_parser.add_argument('--switch-power-w',     type=float, default=None,
                                help='Power draw during reshard window (W). '
                                     'Overrides JSON switch_power_w.')
    elastic_parser.add_argument('--deadline-seconds',   type=float, default=None,
                                help='Total training deadline (s). '
                                     'Overrides JSON deadline_s.')
    elastic_parser.add_argument('--lookahead-steps',    type=int,   default=None,
                                help='Greedy lookahead horizon (steps). '
                                     'Overrides JSON lookahead_steps.')
    elastic_parser.add_argument('--decide-every',       type=int,   default=None,
                                help='Decision interval (steps). '
                                     'Overrides JSON decide_every.')
    # Optional overrides — sensible defaults / derivable from JSON.
    elastic_parser.add_argument('--switch-time-seconds', type=float, default=None,
                                help='Override the calibration\'s '
                                     'average_reshard_wall_time_s. Useful '
                                     'for ablations.')
    elastic_parser.add_argument('--bucket-seconds',     type=float, default=None,
                                help='Wall seconds per CI forecast row '
                                     '(default 3600). Overrides JSON '
                                     'seconds_per_forecast_bucket.')
    elastic_args, megatron_argv = elastic_parser.parse_known_args()

    ckpt_dir = os.path.join(elastic_args.elastic_work_dir, 'elastic_ckpt')

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

    # Resolve policy params: CLI > JSON > built-in default. Required params
    # error if absent from both sources.
    def resolve(cli_val, json_val, cli_flag_name, default=None):
        if cli_val is not None:
            return cli_val
        if json_val is not None:
            return json_val
        if default is not None:
            return default
        raise ValueError(
            f"Required parameter must be provided via --{cli_flag_name} CLI "
            f"arg or as a field in {elastic_args.scheduler_config}"
        )

    strategies                  = cfg['strategies']
    switch_time_s               = resolve(elastic_args.switch_time_seconds,
                                          cfg['switch_time_s'],
                                          'switch-time-seconds')
    switch_power_w              = resolve(elastic_args.switch_power_w,
                                          cfg['switch_power_w'],
                                          'switch-power-w')
    deadline_s                  = resolve(elastic_args.deadline_seconds,
                                          cfg['deadline_s'],
                                          'deadline-seconds')
    lookahead_steps             = resolve(elastic_args.lookahead_steps,
                                          cfg['lookahead_steps'],
                                          'lookahead-steps')
    decide_every                = resolve(elastic_args.decide_every,
                                          cfg['decide_every'],
                                          'decide-every')
    seconds_per_forecast_bucket = resolve(elastic_args.bucket_seconds,
                                          cfg['seconds_per_forecast_bucket'],
                                          'bucket-seconds',
                                          default=3600.0)

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
    # Only validate num_query_groups % tp when GQA is actually enabled.
    # Megatron sets args.num_query_groups=1 by default whether or not GQA
    # is on, so checking for None isn't enough — gate on the flag instead.
    gqa_enabled = getattr(args, 'group_query_attention', False)
    num_query_groups = args.num_query_groups if gqa_enabled else None
    validate_strategies_or_die(
        strategies, n_gpus,
        args.num_layers, args.num_attention_heads,
        num_query_groups,
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
        seconds_per_forecast_bucket=seconds_per_forecast_bucket,
    )

    # ── 7. Start NVML power sampler (per-rank, local device) ────
    power_sampler = NvmlPowerSampler(gpu_index=local_rank)
    power_sampler.start()

    if rank == 0:
        print("\n" + "=" * 70)
        print("  ONLINE ELASTIC 3D PARALLELISM (greedy carbon-aware)")
        print("=" * 70)
        print(f"  GPUs:              {n_gpus}")
        print(f"  Total steps:       {total_steps}")
        print(f"  Deadline:          {deadline_s/3600:.2f} h ({deadline_s:.0f}s)")
        print(f"  Decide every:      {decide_every} steps")
        print(f"  Lookahead:         {lookahead_steps} steps")
        print(f"  Switch cost:       {switch_time_s}s @ {switch_power_w}W")
        print(f"  Bucket size:       {seconds_per_forecast_bucket}s "
              f"(forecast row → wall-time mapping)")
        print(f"  Megatron init:     {t_init:.1f}s (one-time cost)")
        print(f"  Checkpoint dir:    {ckpt_dir}")
        print(f"  Scheduler config:  {elastic_args.scheduler_config}")
        print(f"  Carbon forecast:   {elastic_args.carbon_forecast} "
              f"({len(ci_hourly)} buckets)")
        print(f"  Strategy menu:     {len(strategies)} candidates")
        for s in strategies:
            tag = "← initial" if s is strategies[initial_idx] else ""
            print(f"    TP={s.tp} PP={s.pp}  P≈{s.power_w}W  τ≈{s.step_time_s}s  {tag}")
        print("=" * 70 + "\n", flush=True)

    # ── 8. Drive phases until we reach total_steps ──────────────
    iteration = 0
    current = sched.current
    pending_strategy: Optional[Strategy] = None
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

            if rank == 0:
                print("  [build] Constructing model and optimizer...",
                      flush=True)
            t0 = time.perf_counter()
            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                model_provider_func, ModelType.encoder_or_decoder
            )
            t_build = time.perf_counter() - t0

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
            power_sampler.consume_mean()

            forward_backward_func = get_forward_backward_func()
            config = core_transformer_config_from_args(args)

            t_train_start = time.perf_counter()
            t_log_window = t_train_start
            losses_this_phase: List[float] = []
            log_interval = args.log_interval

            window_start_t = time.perf_counter()
            window_start_step = iteration
            phase_observations: List[tuple] = []
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

                steps_in_window = iteration - window_start_step
                at_decision = (
                    steps_in_window >= decide_every
                    or iteration >= total_steps
                )
                if not at_decision:
                    continue

                window_dur = time.perf_counter() - window_start_t
                measured_step_time = window_dur / steps_in_window
                measured_power = sample_cluster_power(power_sampler)
                phase_observations.append((measured_power, measured_step_time))

                # Capture the cached operating point before observe() runs the
                # EMA, so we can show the refinement step-by-step in the log.
                pre = sched._ops[current.key]
                sched.observe(current.key, measured_power, measured_step_time)
                post = sched._ops[current.key]

                if rank == 0:
                    wall_time = time.perf_counter() - global_start
                    ci_now = sched.ci_at(wall_time) * 3.6e6
                    print(f"    [observe] {steps_in_window} steps  "
                          f"meas: P={measured_power:.0f}W τ={measured_step_time:.3f}s  |  "
                          f"EMA: P {pre.power_w:.0f}→{post.power_w:.0f}W "
                          f"τ {pre.step_time_s:.3f}→{post.step_time_s:.3f}s  |  "
                          f"CI≈{ci_now:.0f}", flush=True)

                if iteration >= total_steps:
                    break

                wall_time = time.perf_counter() - global_start

                # Diagnostic: what does the scheduler think each strategy would
                # cost? Iterate sched.strategies (EMA-updated) and use
                # sched.current — both must match what decide() sees internally,
                # otherwise the displayed ★ can disagree with the actual choice.
                if rank == 0:
                    H_eff = min(lookahead_steps, total_steps - iteration)
                    remaining_steps_cur = total_steps - iteration
                    remaining_time_cur = deadline_s - wall_time
                    fastest_tau = min(s.step_time_s for s in sched.strategies)
                    slack = remaining_time_cur - remaining_steps_cur * fastest_tau

                    # Sample CI across the *current* strategy's lookahead window
                    # so we can see whether temporal variation is visible to the
                    # policy at this decision point. Use sched.current's τ for
                    # the window length.
                    cur = sched.current
                    window_dur_la = H_eff * cur.step_time_s
                    ci_samples = [sched.ci_at(wall_time + frac * window_dur_la) * 3.6e6
                                  for frac in (0.0, 0.5, 1.0)]

                    print(f"    [scheduler] H={H_eff}, t={wall_time:.1f}s, "
                          f"CI in window: {ci_samples[0]:.0f}→{ci_samples[1]:.0f}→"
                          f"{ci_samples[2]:.0f} gCO2/kWh, "
                          f"slack={slack:.1f}s", flush=True)

                    # Compute cost for each candidate using the SAME state
                    # decide() will use: EMA-updated operating points
                    # (sched.strategies) and the scheduler's internal current
                    # (sched.current). This guarantees the displayed ★ matches
                    # the action taken by decide() on the next line.
                    costs = []
                    for s in sched.strategies:
                        cost = sched._lookahead_carbon(s, sched.current,
                                                      wall_time, H_eff)
                        feas = (s.step_time_s * remaining_steps_cur
                                <= remaining_time_cur)
                        costs.append((s, cost, feas))

                    # Sort cheapest-first so the rank ordering jumps out at a glance.
                    costs.sort(key=lambda x: x[1])
                    for rank_idx, (s, cost, feas) in enumerate(costs):
                        action = "(stay)  " if s.key == sched.current.key else "(switch)"
                        feas_tag = "        " if feas else "[INFEAS]"
                        marker = "★" if rank_idx == 0 else " "
                        print(f"    [scheduler] {marker} TP={s.tp} PP={s.pp} {action} {feas_tag} "
                              f"→ {cost*1000:.2f} mgCO2 over {H_eff} steps", flush=True)

                next_strat = sched.decide(iteration, wall_time)

                if next_strat.key != current.key:
                    if rank == 0:
                        print(f"    [decide] SWITCH "
                              f"TP={current.tp}/PP={current.pp} → "
                              f"TP={next_strat.tp}/PP={next_strat.pp} "
                              f"@ step {iteration} "
                              f"(wall {wall_time:.1f}s)", flush=True)
                    pending_strategy = next_strat
                    break

                if rank == 0:
                    print(f"    [decide] stay on "
                          f"TP={current.tp}/PP={current.pp}", flush=True)

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

            is_last = (pending_strategy is None)
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

    # ── 9. Emit phases.json + summary ──────────────────────────
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
                'seconds_per_forecast_bucket': seconds_per_forecast_bucket,
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