#!/usr/bin/env python3
"""
profile_strategies.py
---------------------
Profiler that runs each (TP, PP) candidate, measures cluster power, mean
step time, AND full switch wall time (including checkpoint save+load),
then emits a calibration JSON ready to drop into scheduler_config.json.

Reshards in-process between configs (no Megatron re-init) and saves a
checkpoint after each config so the next one loads it — this captures
the real switch cost the scheduler will pay in production, not just the
MPU state-change overhead.

Top-level field `average_reshard_wall_time_s` aggregates the observed
transitions; that's the value to drop into scheduler_config.json's
`switch_time_s`.

Usage:
    python profile_strategies.py \
        --profile-menu profile_menu.json \
        --warmup-seconds 60 \
        --measure-seconds 60 \
        --output calibration.json \
        [megatron args...]
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import List

import torch
import torch.distributed as dist

from gpt_builders import gpt_builder
from model_provider import model_provider

from elastic_pretrain import (
    reconfigure_microbatches_for_phase,
    forward_step,
    infinite_data_iterator,
    verify_parallel_state,
    free_model_memory,
)
from online_elastic_pretrain import (
    NvmlPowerSampler,
    sample_cluster_power,
    validate_strategies_or_die,
    update_args_for_strategy,
)
from greedy_scheduler import Strategy


# ═══════════════════════════════════════════════════════════════
# Menu loading
# ═══════════════════════════════════════════════════════════════

def load_profile_menu(path: str) -> List[Strategy]:
    """Load list of {tp, pp} from JSON. Power/step_time placeholders
    are accepted but ignored — the profiler will measure them."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get('strategies', raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected non-empty list of strategies")

    out: List[Strategy] = []
    for i, s in enumerate(raw):
        if 'tp' not in s or 'pp' not in s:
            raise ValueError(f"{path} entry {i}: needs 'tp' and 'pp'")
        out.append(Strategy(
            tp=int(s['tp']),
            pp=int(s['pp']),
            power_w=float(s.get('power_w', 0.0)),
            step_time_s=float(s.get('step_time_s', 0.0)),
        ))

    keys = [s.key for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (tp, pp) entries")
    return out


# ═══════════════════════════════════════════════════════════════
# Run-for-N-seconds helper
# ═══════════════════════════════════════════════════════════════

def run_for_seconds(target_s, train_step_fn, *args):
    """Drive train_step in a loop until wall time exceeds target_s.
    Returns (n_iters, elapsed_s). Final dist.barrier() ensures all
    ranks finish together so timing reflects all-rank completion."""
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < target_s:
        train_step_fn(*args)
        n += 1
    dist.barrier()
    elapsed = time.perf_counter() - t0
    return n, elapsed


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. Pull profiler-only args off sys.argv ─────────────────
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--profile-menu',     required=True)
    p.add_argument('--warmup-seconds',   type=float, default=60.0,
                   help='Discard this many seconds of training before measuring '
                        '(GPU thermal/clock ramp-up, default 60s).')
    p.add_argument('--measure-seconds',  type=float, default=60.0,
                   help='Measurement window length in wall seconds (default 60s).')
    p.add_argument('--output',           required=True)
    p.add_argument('--elastic-work-dir', default='/tmp/profile_workdir')
    profile_args, megatron_argv = p.parse_known_args()

    work_ckpt = os.path.join(profile_args.elastic_work_dir, 'profile_ckpt')
    sys.argv = [sys.argv[0]] + megatron_argv + [
        '--ckpt-format', 'torch_dist',
        '--use-distributed-optimizer',
        '--auto-detect-ckpt-format',
        '--override-opt_param-scheduler',
        '--save', work_ckpt,
        '--save-interval', '999999',
        '--dist-ckpt-optim-fully-reshardable',
    ]

    # ── 2. Load menu (cheap, before Megatron init) ──────────────
    profile_menu = load_profile_menu(profile_args.profile_menu)

    # ── 3. Megatron init ────────────────────────────────────────
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
    initialize_megatron(allow_no_cuda=False)

    args = get_args()
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    # ── 4. Validate every config upfront ────────────────────────
    gqa_enabled = getattr(args, 'group_query_attention', False)
    nqg = args.num_query_groups if gqa_enabled else None
    validate_strategies_or_die(
        profile_menu, n_gpus,
        args.num_layers, args.num_attention_heads, nqg, rank,
    )

    # ── 5. Set up NVML + data iter ──────────────────────────────
    sampler = NvmlPowerSampler(gpu_index=local_rank)
    sampler.start()
    data_iter = RerunDataIterator(iter(infinite_data_iterator()))

    if rank == 0:
        print("\n" + "=" * 70)
        print("  STRATEGY PROFILER")
        print("=" * 70)
        print(f"  GPUs:               {n_gpus}")
        print(f"  Warmup window:      {profile_args.warmup_seconds:.0f}s")
        print(f"  Measure window:     {profile_args.measure_seconds:.0f}s")
        print(f"  Profile menu:       {len(profile_menu)} configs")
        print(f"  Output:             {profile_args.output}")
        print(f"  Checkpoint dir:     {work_ckpt} (used to measure full switch cost)")
        per_cfg = profile_args.warmup_seconds + profile_args.measure_seconds
        n_transitions = len(profile_menu) - 1
        print(f"  Estimated total:    ~{(per_cfg * len(profile_menu) + 37 * n_transitions) / 60:.1f} min "
              f"(measurement + {n_transitions} transitions × ~37s save+load)")
        print("=" * 70 + "\n", flush=True)

    # ── 6. Per-config measurement loop ──────────────────────────
    results = []
    last_measure_end_t = None  # for full reshard wall-time accounting

    try:
        for idx, strat in enumerate(profile_menu):
            is_first = (idx == 0)

            if rank == 0:
                print(f"\n[{idx+1}/{len(profile_menu)}] "
                      f"Profiling TP={strat.tp} PP={strat.pp} "
                      f"(DP={n_gpus // (strat.tp * strat.pp)})", flush=True)

            # Reshard for non-first configs. is_first_phase=False → checkpoint
            # loads on rebuild, which is the realistic production cost.
            if not is_first:
                t0 = time.perf_counter()
                mpu.destroy_model_parallel()
                update_args_for_strategy(args, strat, work_ckpt,
                                         is_first_phase=False)
                dist.barrier()
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=strat.tp,
                    pipeline_model_parallel_size=strat.pp,
                )
                reconfigure_microbatches_for_phase(args, rank)
                t_mpu_reshard = time.perf_counter() - t0
                if rank == 0:
                    print(f"  MPU state change: {t_mpu_reshard:.2f}s",
                          flush=True)
            else:
                update_args_for_strategy(args, strat, work_ckpt,
                                         is_first_phase=True)
                t_mpu_reshard = 0.0

            # Build model. For non-first configs this loads the checkpoint
            # saved by the previous config — that's the dominant switch cost.
            t0 = time.perf_counter()
            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                model_provider_func, ModelType.encoder_or_decoder
            )
            t_build = time.perf_counter() - t0
            if rank == 0:
                print(f"  build (incl ckpt load if applicable): "
                      f"{t_build:.2f}s", flush=True)

            verify_parallel_state(
                model,
                expected_tp=strat.tp, expected_pp=strat.pp,
                expected_dp=n_gpus // (strat.tp * strat.pp),
                rank=rank,
            )

            # Capture timestamp right before first train_step of warmup.
            # Combined with last_measure_end_t (set at end of previous config),
            # this measures the full reshard wall time the production scheduler
            # would pay to switch into this config.
            first_step_start_t = time.perf_counter()
            if last_measure_end_t is not None:
                full_reshard_s = first_step_start_t - last_measure_end_t
                if rank == 0:
                    print(f"  full reshard wall time (since last measure end): "
                          f"{full_reshard_s:.2f}s", flush=True)
            else:
                full_reshard_s = None  # first config — no preceding transition

            forward_backward_func = get_forward_backward_func()
            config = core_transformer_config_from_args(args)
            step_args = (forward_step, data_iter, model, optimizer,
                         opt_param_scheduler, config, forward_backward_func)

            # Warmup — discard timing and power
            if rank == 0:
                print(f"  warmup ({profile_args.warmup_seconds:.0f}s)...",
                      flush=True)
            n_warmup, t_warmup = run_for_seconds(
                profile_args.warmup_seconds, train_step, *step_args
            )
            sampler.consume_mean()  # drain NVML samples accumulated in warmup

            # Measure window
            if rank == 0:
                print(f"  measure ({profile_args.measure_seconds:.0f}s)...",
                      flush=True)
            n_measure, t_measure = run_for_seconds(
                profile_args.measure_seconds, train_step, *step_args
            )
            measure_end_t = time.perf_counter()

            mean_step_time = t_measure / max(n_measure, 1)
            mean_cluster_power = sample_cluster_power(sampler)

            # Save checkpoint so the next config can load it (this is part of
            # the wall time cost the scheduler pays at every switch).
            t_save = 0.0
            if idx < len(profile_menu) - 1:
                if rank == 0:
                    print(f"  saving checkpoint for next config...",
                          flush=True)
                t0 = time.perf_counter()
                save_checkpoint(0, model, optimizer,
                                opt_param_scheduler, 0)
                t_save = time.perf_counter() - t0
                if rank == 0:
                    print(f"  checkpoint saved: {t_save:.2f}s", flush=True)

            # Mark the end-of-this-config moment for next iteration's
            # reshard accounting. We use measure_end_t (not after save) so the
            # next config's full_reshard_s captures save → reshard → build.
            last_measure_end_t = measure_end_t

            results.append({
                'tp':              strat.tp,
                'pp':              strat.pp,
                'power_w':         round(mean_cluster_power, 1),
                'step_time_s':     round(mean_step_time, 4),
                # Diagnostic / provenance
                'dp':              n_gpus // (strat.tp * strat.pp),
                'warmup_seconds':  profile_args.warmup_seconds,
                'measure_seconds': profile_args.measure_seconds,
                'warmup_iters':    n_warmup,
                'measure_iters':   n_measure,
                'measure_wall_s':  round(t_measure, 2),
                'build_s':         round(t_build, 2),
                'save_s':          round(t_save, 2),
                'mpu_reshard_s':   round(t_mpu_reshard, 2),
                # Full wall-clock gap from end-of-measure of previous config
                # to first-step of this config. None for the first config.
                'reshard_wall_time_s': (round(full_reshard_s, 2)
                                         if full_reshard_s is not None else None),
            })

            if rank == 0:
                E = mean_cluster_power * mean_step_time
                print(f"  → P = {mean_cluster_power:.1f}W cluster, "
                      f"τ = {mean_step_time:.4f}s/step, "
                      f"E = {E:.1f}J/step "
                      f"({n_warmup} warmup + {n_measure} measure iters)",
                      flush=True)

            if idx < len(profile_menu) - 1:
                free_model_memory(model, optimizer, opt_param_scheduler)
            else:
                del model, optimizer, opt_param_scheduler

    finally:
        sampler.stop()

    # ── 7. Aggregate transitions and emit calibration JSON ──────
    if rank == 0:
        # Average reshard wall time over all observed transitions
        transition_times = [r['reshard_wall_time_s'] for r in results
                            if r['reshard_wall_time_s'] is not None]
        avg_reshard_s = (sum(transition_times) / len(transition_times)
                         if transition_times else 0.0)

        out_dir = os.path.dirname(profile_args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_doc = {
            'n_gpus':                       n_gpus,
            'warmup_seconds':               profile_args.warmup_seconds,
            'measure_seconds':              profile_args.measure_seconds,
            'average_reshard_wall_time_s':  round(avg_reshard_s, 2),
            'reshard_observations':         len(transition_times),
            'strategies':                   results,
        }
        with open(profile_args.output, 'w') as f:
            json.dump(out_doc, f, indent=2)

        # Pretty summary
        print(f"\n{'=' * 70}")
        print("  PROFILE COMPLETE")
        print(f"{'=' * 70}")
        print(f"  {'(TP,PP)':>10}  {'P (W)':>10}  {'τ (s)':>10}  "
              f"{'E (J)':>10}  {'reshard':>10}")
        print("  " + "-" * 60)
        for r in results:
            E = r['power_w'] * r['step_time_s']
            tag = f"({r['tp']},{r['pp']})"
            rs = (f"{r['reshard_wall_time_s']:.1f}s"
                  if r['reshard_wall_time_s'] is not None else "—")
            print(f"  {tag:>10}  {r['power_w']:>10.1f}  "
                  f"{r['step_time_s']:>10.4f}  {E:>10.1f}  {rs:>10}")
        print("  " + "-" * 60)
        print(f"\n  Average reshard wall time: {avg_reshard_s:.2f}s "
              f"({len(transition_times)} transitions observed)")
        print(f"  → Drop into scheduler_config.json as `switch_time_s`")
        print(f"\n  Calibration written to: {profile_args.output}")
        print(f"{'=' * 70}\n", flush=True)

    dist.barrier()


if __name__ == '__main__':
    main()