#!/usr/bin/env python3
"""
profile_strategies.py
---------------------
Profiler that runs each (TP, PP, micro_batch_size) candidate, measures
cluster power, mean step time, throughput in tokens/sec, energy per token,
AND full switch wall time including checkpoint save+load. It then emits a
calibration JSON ready to drop into scheduler_config.json.

Reshards in-process between configs (no Megatron re-init) and saves a
checkpoint after each config so the next one loads it. This captures the
real switch cost the scheduler will pay in production, not just the MPU
state-change overhead.

Top-level field `average_reshard_wall_time_s` aggregates the observed
transitions. That is the value to drop into scheduler_config.json's
`switch_time_s`.

Warmup is CONVERGENCE-BASED: training proceeds until the rolling window
of step times stabilizes, bounded by `--warmup-min-seconds` and
`--warmup-max-seconds`. Rank 0 decides; the exit flag is broadcast each
iteration so ranks stay in lockstep.

The first config does a discarded `save_checkpoint` after the warmup
train_steps have run so Adam has lazily allocated exp_avg/exp_avg_sq.
This warms the dist-checkpoint subsystem so the first MEASURED transition's
wall time reflects steady-state save cost, not one-time setup.

After the main loop, the profiler does a WRAPAROUND: reshard from the
last config back to the first and time the transition. This gives the
first config a `reshard_wall_time_s` and adds one more transition
observation to `average_reshard_wall_time_s`.

Usage:
    python profile_strategies.py \
        --profile-menu profile_menu.json \
        --warmup-min-seconds 10 \
        --warmup-max-seconds 60 \
        --warmup-window 10 \
        --warmup-tol 0.02 \
        --measure-seconds 60 \
        --output calibration.json \
        [megatron args...]
"""

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

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

# ═══════════════════════════════════════════════════════════════
# Menu loading
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProfileCase:
    tp: int
    pp: int
    micro_batch_size: Optional[int] = None
    order: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (self.tp, self.pp, self.micro_batch_size, self.order)

    def effective_micro_batch_size(self, default_micro_batch_size: int) -> int:
        if self.micro_batch_size is None:
            return default_micro_batch_size
        return self.micro_batch_size


def _coerce_positive_int(value, field: str, path: str, entry_idx: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} entry {entry_idx}: '{field}' must be a positive integer")

    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} entry {entry_idx}: '{field}' must be a positive integer"
        ) from exc

    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path} entry {entry_idx}: '{field}' must be a positive integer")

    if out <= 0:
        raise ValueError(f"{path} entry {entry_idx}: '{field}' must be a positive integer")

    return out


def validate_profile_order(order):
    """Only plain TP/PP/DP permutations; singleton CP/EP axes are implicit."""
    if order is None:
        return None
    if not isinstance(order, str) or sorted(order.split("-")) != ["dp", "pp", "tp"]:
        raise ValueError("order must be a permutation of tp-dp-pp")
    return order


def collect_rank_mapping():
    """Gather observed devices and memberships, not predicted rank layouts."""
    from megatron.core import parallel_state as mpu

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    local = {
        "rank": dist.get_rank(),
        "hostname": socket.gethostname(),
        "local_rank": int(os.environ["SLURM_LOCALID"] if os.environ.get("PROFILE_SLURM_LAUNCH") == "1"
                          else os.environ.get("LOCAL_RANK", "0")),
        "cuda_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": str(properties.uuid),
        "gpu_name": properties.name,
        "tp_group": dist.get_process_group_ranks(mpu.get_tensor_model_parallel_group()),
        "pp_group": dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()),
        "dp_group": dist.get_process_group_ranks(mpu.get_data_parallel_group()),
    }
    mapping = [None] * dist.get_world_size()
    dist.all_gather_object(mapping, local)
    devices = {(entry["hostname"], entry["gpu_uuid"]) for entry in mapping}
    if len(devices) != dist.get_world_size():
        raise RuntimeError("Profiler requires one distinct GPU per global rank")
    return mapping


def load_profile_menu(path: str) -> List[ProfileCase]:
    """Load TP/PP cases with optional micro_batch_size and one static order."""
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get('strategies', raw)

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected non-empty list of strategies")

    out: List[ProfileCase] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(f"{path} entry {i}: expected an object")

        if 'tp' not in s or 'pp' not in s:
            raise ValueError(f"{path} entry {i}: needs 'tp' and 'pp'")

        micro_batch_size = None
        if 'micro_batch_size' in s:
            micro_batch_size = _coerce_positive_int(
                s['micro_batch_size'],
                'micro_batch_size',
                path,
                i,
            )

        out.append(ProfileCase(
            tp=_coerce_positive_int(s['tp'], 'tp', path, i),
            pp=_coerce_positive_int(s['pp'], 'pp', path, i),
            micro_batch_size=micro_batch_size,
            order=validate_profile_order(s.get("order")),
        ))

    if len({case.order for case in out}) != 1:
        raise ValueError("Use separate profiler runs for different orders; dynamic placement switching is unsupported")

    keys = [s.key for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (tp, pp, micro_batch_size, order) entries")

    return out


def apply_micro_batch_size_for_case(args,
                                    case: ProfileCase,
                                    default_micro_batch_size: int) -> int:
    """Set args.micro_batch_size explicitly for every case."""
    effective_micro_batch_size = case.effective_micro_batch_size(
        default_micro_batch_size
    )
    args.micro_batch_size = effective_micro_batch_size
    return effective_micro_batch_size


def validate_micro_batch_cases_or_die(cases: List[ProfileCase],
                                      n_gpus: int,
                                      global_batch_size: int,
                                      default_micro_batch_size: int,
                                      rank: int) -> None:
    """Fail early if a case cannot preserve the requested global batch size."""
    failures = []

    for case in cases:
        model_parallel_size = case.tp * case.pp
        if n_gpus % model_parallel_size != 0:
            continue

        dp = n_gpus // model_parallel_size
        micro_batch_size = case.effective_micro_batch_size(
            default_micro_batch_size
        )
        divisor = micro_batch_size * dp

        if global_batch_size % divisor != 0:
            failures.append((
                case,
                dp,
                micro_batch_size,
                (
                    f"global_batch_size={global_batch_size} is not divisible "
                    f"by micro_batch_size={micro_batch_size} * dp={dp}"
                ),
            ))

    if failures:
        if rank == 0:
            print("[validate] Profile menu has infeasible microbatch entries:",
                  flush=True)
            for case, dp, micro_batch_size, err in failures:
                print(
                    f"  TP={case.tp} PP={case.pp} DP={dp} "
                    f"MBS={micro_batch_size}: {err}",
                    flush=True,
                )
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Convergence-based warmup
# ═══════════════════════════════════════════════════════════════

def run_until_stable(min_s: float,
                     max_s: float,
                     window: int,
                     tol: float,
                     train_step_fn,
                     *args) -> Tuple[int, float, bool, float]:
    """Drive train_step until the MEAN step time has stopped drifting.

    Convergence criterion:
        | mean(last `window` steps) - mean(prior `window` steps) |
        --------------------------------------------------------- < tol
                       mean(last `window` steps)

    This tests whether the mean is stationary across two consecutive
    non-overlapping windows. It does not require every individual step
    to be close to the mean, because real training has per-step noise.

    Rank 0 makes the decision. The exit flag is broadcast to all ranks
    every iteration so they stay in lockstep.

    Returns:
        (n_iters, elapsed_s, converged, final_window_mean)
    """
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    exit_flag = torch.zeros(1, dtype=torch.int32, device=device)
    converged_flag = torch.zeros(1, dtype=torch.int32, device=device)

    t0 = time.perf_counter()
    step_times: List[float] = []

    while True:
        t_step = time.perf_counter()
        train_step_fn(*args)
        step_times.append(time.perf_counter() - t_step)

        elapsed = time.perf_counter() - t0

        if rank == 0:
            should_exit = False
            did_converge = False

            if elapsed >= max_s:
                should_exit = True

            elif elapsed >= min_s and len(step_times) >= 2 * window:
                recent = step_times[-window:]
                prior = step_times[-2 * window:-window]

                recent_mean = sum(recent) / window
                prior_mean = sum(prior) / window

                if recent_mean > 0:
                    drift = abs(recent_mean - prior_mean) / recent_mean
                    if drift < tol:
                        should_exit = True
                        did_converge = True

            exit_flag[0] = 1 if should_exit else 0
            converged_flag[0] = 1 if did_converge else 0

        dist.broadcast(exit_flag, src=0)

        if exit_flag.item():
            break

    dist.broadcast(converged_flag, src=0)
    dist.barrier()

    elapsed_final = time.perf_counter() - t0
    win = min(window, len(step_times))
    final_mean = (sum(step_times[-win:]) / win) if step_times else 0.0

    return (
        len(step_times),
        elapsed_final,
        bool(converged_flag.item()),
        final_mean,
    )


# ═══════════════════════════════════════════════════════════════
# Fixed-time measurement window
# ═══════════════════════════════════════════════════════════════

def run_for_seconds(target_s, train_step_fn, *args):
    """Drive train_step in a loop until wall time exceeds target_s.

    Used for the measurement window. Returns (n_iters, elapsed_s).
    Final dist.barrier() ensures all ranks finish together.
    """
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    n = 0

    stop = torch.zeros(1, dtype=torch.int32, device=torch.cuda.current_device())
    while True:
        train_step_fn(*args)
        n += 1
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            stop[0] = int(time.perf_counter() - t0 >= target_s)
        dist.broadcast(stop, src=0)
        if stop.item():
            break

    dist.barrier()
    elapsed = time.perf_counter() - t0

    return n, elapsed


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def select_profile_cuda_device(args):
    """Select a visible ordinal, not a physical GPU ID or global rank."""
    local_rank = int(os.environ["SLURM_LOCALID"] if os.environ.get("PROFILE_SLURM_LAUNCH") == "1"
                     else os.environ.get("LOCAL_RANK", args.local_rank))
    count = torch.cuda.device_count()
    if os.environ.get("PROFILE_TASK_GPUS") == "1" and count != 1:
        raise ValueError(f"Profiler requires one task-bound visible CUDA device, got {count}; "
                         "use --gpus-per-task=1 --gpu-bind=single:1 "
                         "--gres-flags=allow-task-sharing")
    device = 0 if count == 1 else local_rank
    if local_rank < 0 or not 0 <= device < count:
        raise ValueError("Profiler local rank cannot map to a visible CUDA device")
    torch.cuda.set_device(device)
    args.local_rank = device
    os.environ["LOCAL_RANK"] = str(device)
    print(f"[profile CUDA] host={socket.gethostname()} rank={args.rank} local_rank={local_rank} "
          f"visible_device_count={count} cuda_device={device} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    return device


def main():
    # ── 1. Pull profiler-only args off sys.argv ─────────────────
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument('--profile-menu', required=True)

    p.add_argument(
        '--warmup-min-seconds',
        type=float,
        default=10.0,
        help='Minimum warmup wall time before the convergence check can fire.',
    )
    p.add_argument(
        '--warmup-max-seconds',
        type=float,
        default=60.0,
        help='Cap on warmup wall time if step times never stabilize.',
    )
    p.add_argument(
        '--warmup-window',
        type=int,
        default=10,
        help='Rolling-window size for step-time stability check.',
    )
    p.add_argument(
        '--warmup-tol',
        type=float,
        default=0.02,
        help='Fractional tolerance for stability.',
    )
    p.add_argument(
        '--measure-seconds',
        type=float,
        default=60.0,
        help='Measurement window length in wall seconds.',
    )
    p.add_argument('--output', required=True)
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

    # ── 2. Load menu before Megatron init ───────────────────────
    profile_menu = load_profile_menu(profile_args.profile_menu)

    # ── 3. Megatron init ────────────────────────────────────────
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer, train_step
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training import get_args, get_timers
    from megatron.core.utils import get_model_config
    from megatron.core.distributed import finalize_model_grads
    from megatron.core import parallel_state as mpu
    from megatron.core.enums import ModelType
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.rerun_state_machine import RerunDataIterator

    model_provider_func = partial(model_provider, gpt_builder)
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.global_vars import initialize_runtime_services, set_run_config

    args = parse_and_validate_args()
    if os.environ.get("PROFILE_SLURM_LAUNCH") == "1":
        from carbon_aware_3d.gpu_preflight import run_preflight
        run_preflight(args, profile_args.elastic_work_dir, select_profile_cuda_device)
    else:
        select_profile_cuda_device(args)
    if (args.tensor_model_parallel_size, args.pipeline_model_parallel_size) != (
        profile_menu[0].tp, profile_menu[0].pp
    ):
        raise ValueError('Initial TP/PP must match the first profile-menu case')
    if profile_menu[0].order is not None:
        if (args.context_parallel_size != 1 or args.expert_model_parallel_size != 1
                or args.gtp_weight_remat_size != 1 or args.expert_gtp_weight_remat_size != 1
                or args.gtp_remat_num_sequence_shards != 1
                or args.expert_tensor_parallel_size != args.tensor_model_parallel_size
                or args.virtual_pipeline_model_parallel_size is not None):
            raise ValueError("Explicit order profiling supports plain TP/PP/DP only")
    set_run_config(pretrain_cfg_container_from_args(args))
    initialize_runtime_services(args, training=True)
    # Keep normal initialization (including RNG setup) in the requested groups.
    requested_order = profile_menu[0].order
    initialize_megatron(
        allow_no_cuda=False, training=True, model_parallel_order=requested_order
    )
    initial_order = requested_order or (
        "tp-cp-ep-pp-dp" if args.use_tp_pp_dp_mapping else "tp-cp-ep-dp-pp"
    )

    args = get_args()
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()
    default_micro_batch_size = int(args.micro_batch_size)

    # ── 4. Validate every config upfront ────────────────────────
    gqa_enabled = getattr(args, 'group_query_attention', False)
    nqg = args.num_query_groups if gqa_enabled else None

    validate_strategies_or_die(
        profile_menu,
        n_gpus,
        args.num_layers,
        args.num_attention_heads,
        nqg,
        rank,
    )
    validate_micro_batch_cases_or_die(
        profile_menu,
        n_gpus,
        int(args.global_batch_size),
        default_micro_batch_size,
        rank,
    )

    # ── 5. Set up NVML + data iter ──────────────────────────────
    sampler = NvmlPowerSampler(gpu_index=torch.cuda.current_device())
    sampler.start()

    data_iter = RerunDataIterator(iter(infinite_data_iterator()))

    if rank == 0:
        print("\n" + "=" * 70)
        print("  STRATEGY PROFILER")
        print("=" * 70)
        print(f"  GPUs:               {n_gpus}")
        print(f"  Global batch size:  {args.global_batch_size}")
        print(f"  Default microbatch: {default_micro_batch_size}")
        print(f"  Sequence length:    {args.seq_length}")
        print(f"  Tokens / iter:      {int(args.global_batch_size) * int(args.seq_length):,}")
        print(f"  Warmup min/max:     "
              f"{profile_args.warmup_min_seconds:.0f}s / "
              f"{profile_args.warmup_max_seconds:.0f}s")
        print(f"  Warmup convergence: window={profile_args.warmup_window} "
              f"iters, tol=±{profile_args.warmup_tol * 100:.1f}%")
        print(f"  Measure window:     {profile_args.measure_seconds:.0f}s")
        print(f"  Profile menu:       {len(profile_menu)} cases")
        print(f"  Output:             {profile_args.output}")
        print(f"  Checkpoint dir:     {work_ckpt}")
        print("=" * 70 + "\n", flush=True)

    # ── 6. Per-config measurement loop ──────────────────────────
    results = []
    last_measure_end_t = None

    try:
        for idx, case in enumerate(profile_menu):
            is_first = (idx == 0)
            case_micro_batch_size = case.effective_micro_batch_size(
                default_micro_batch_size
            )
            case_dp = n_gpus // (case.tp * case.pp)

            if rank == 0:
                print(f"\n[{idx + 1}/{len(profile_menu)}] "
                      f"Profiling TP={case.tp} PP={case.pp} "
                      f"DP={case_dp} MBS={case_micro_batch_size}",
                      flush=True)

            # Reshard for non-first configs.
            if not is_first:
                t0 = time.perf_counter()

                mpu.destroy_model_parallel()
                update_args_for_strategy(
                    args,
                    case,
                    work_ckpt,
                    is_first_phase=False,
                )
                apply_micro_batch_size_for_case(
                    args,
                    case,
                    default_micro_batch_size,
                )

                dist.barrier()

                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=case.tp,
                    pipeline_model_parallel_size=case.pp,
                    order=case.order or "tp-cp-ep-dp-pp",
                )
                reconfigure_microbatches_for_phase(args, rank)
                num_microbatches = get_num_microbatches()

                t_mpu_reshard = time.perf_counter() - t0

                if rank == 0:
                    print(f"  MPU state change: {t_mpu_reshard:.2f}s",
                          flush=True)
            else:
                update_args_for_strategy(
                    args,
                    case,
                    work_ckpt,
                    is_first_phase=True,
                )
                apply_micro_batch_size_for_case(
                    args,
                    case,
                    default_micro_batch_size,
                )
                reconfigure_microbatches_for_phase(args, rank)
                num_microbatches = get_num_microbatches()
                t_mpu_reshard = 0.0

            rank_mapping = collect_rank_mapping()
            actual_order = initial_order if is_first else (case.order or "tp-cp-ep-dp-pp")
            if rank == 0:
                os.makedirs(profile_args.elastic_work_dir, exist_ok=True)
                mapping_path = os.path.join(
                    profile_args.elastic_work_dir,
                    f"rank_mapping_tp{case.tp}_pp{case.pp}_mbs{args.micro_batch_size}.json",
                )
                with open(mapping_path, "w") as f:
                    json.dump({"order": actual_order, "tp": case.tp, "pp": case.pp,
                               "ranks": rank_mapping}, f, indent=2)

            # Build model. For non-first configs, this loads the checkpoint
            # saved by the previous config.
            t0 = time.perf_counter()

            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                model_provider_func=model_provider_func,
                model_type=ModelType.encoder_or_decoder,
            )

            t_build = time.perf_counter() - t0

            if rank == 0:
                print(f"  build (incl ckpt load if applicable): "
                      f"{t_build:.2f}s", flush=True)

            verify_parallel_state(
                model,
                expected_tp=case.tp,
                expected_pp=case.pp,
                expected_dp=case_dp,
                rank=rank,
            )

            # Capture timestamp right before first train_step of warmup.
            first_step_start_t = time.perf_counter()

            if last_measure_end_t is not None:
                full_reshard_s = first_step_start_t - last_measure_end_t
                if rank == 0:
                    print(f"  full reshard wall time "
                          f"(since last measure end): "
                          f"{full_reshard_s:.2f}s",
                          flush=True)
            else:
                full_reshard_s = None

            forward_backward_func = get_forward_backward_func()
            config = get_model_config(model[0])
            config.grad_scale_func = optimizer.scale_loss
            config.timers = get_timers()
            config.finalize_model_grads_func = finalize_model_grads
            if args.overlap_grad_reduce:
                config.no_sync_func = model[0].no_sync
                if args.align_grad_reduce:
                    config.grad_sync_func = model[0].start_grad_sync
            if args.overlap_param_gather and args.align_param_gather:
                config.param_sync_func = model[0].start_param_sync

            step_args = (
                forward_step,
                data_iter,
                model,
                optimizer,
                opt_param_scheduler,
                config,
                forward_backward_func,
            )

            # Warmup.
            if rank == 0:
                print(f"  warmup (converge or "
                      f"{profile_args.warmup_max_seconds:.0f}s max)...",
                      flush=True)

            n_warmup, t_warmup, converged, warmup_final_mean = run_until_stable(
                profile_args.warmup_min_seconds,
                profile_args.warmup_max_seconds,
                profile_args.warmup_window,
                profile_args.warmup_tol,
                train_step,
                *step_args,
            )

            if rank == 0:
                if converged:
                    status = (
                        f"converged after {t_warmup:.1f}s "
                        f"({n_warmup} iters), "
                        f"window mean {warmup_final_mean * 1000:.1f}ms"
                    )
                else:
                    status = (
                        f"HIT MAX after {t_warmup:.1f}s "
                        f"({n_warmup} iters). Measurement may be noisy"
                    )

                print(f"  warmup: {status}", flush=True)

            # Warmup save for first config only.
            if is_first:
                if rank == 0:
                    print("  warmup save (discarded, warms dist-ckpt subsystem)...",
                          flush=True)

                t0 = time.perf_counter()

                save_checkpoint(
                    0,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    0,
                )

                t_warmup_save = time.perf_counter() - t0

                if rank == 0:
                    print(f"  warmup save: {t_warmup_save:.2f}s "
                          f"(not counted)",
                          flush=True)

            # Drain NVML samples accumulated during warmup and warmup save.
            dist.barrier()
            sampler.consume_mean()

            # Measurement window.
            if rank == 0:
                print(f"  measure ({profile_args.measure_seconds:.0f}s)...",
                      flush=True)

            n_measure, t_measure = run_for_seconds(
                profile_args.measure_seconds,
                train_step,
                *step_args,
            )

            measure_end_t = time.perf_counter()

            mean_step_time = t_measure / max(n_measure, 1)
            mean_cluster_power = sample_cluster_power(sampler)
            if not 0.0 < mean_cluster_power < float('inf'):
                raise RuntimeError('Missing or invalid GPU power samples; refusing calibration')

            # -----------------------------
            # New explicit throughput fields
            # -----------------------------
            # In Megatron, global_batch_size is the total number of sequences
            # processed per training iteration. This already accounts for DP.
            # TP and PP split model work, not independent token streams.
            tokens_per_iter = int(args.global_batch_size) * int(args.seq_length)

            # Throughput measured directly from the measurement window.
            measured_tokens = n_measure * tokens_per_iter
            throughput_tok_s = (
                measured_tokens / t_measure
                if t_measure > 0
                else 0.0
            )

            # Since watt = J/s and throughput = tok/s:
            #     (J/s) / (tok/s) = J/tok
            joules_per_token = (
                mean_cluster_power / throughput_tok_s
                if throughput_tok_s > 0
                else None
            )

            # Save checkpoint so the next config or wraparound can load it.
            if rank == 0:
                print("  saving checkpoint for next config...", flush=True)

            t0 = time.perf_counter()

            save_checkpoint(
                0,
                model,
                optimizer,
                opt_param_scheduler,
                0,
            )

            t_save = time.perf_counter() - t0

            if rank == 0:
                print(f"  checkpoint saved: {t_save:.2f}s",
                      flush=True)

            # We use measure_end_t instead of after-save time so the next
            # config's full_reshard_s captures save -> reshard -> build.
            last_measure_end_t = measure_end_t

            results.append({
                'tp': case.tp,
                'pp': case.pp,
                'dp': case_dp,
                'order': actual_order,
                'rank_mapping': rank_mapping,
                'micro_batch_size': int(args.micro_batch_size),
                'num_microbatches': int(num_microbatches),

                # Main calibration quantities.
                'power_w': round(mean_cluster_power, 1),
                'step_time_s': round(mean_step_time, 4),
                'throughput_tok_s': round(throughput_tok_s, 2),
                'joules_per_token': (
                    round(joules_per_token, 6)
                    if joules_per_token is not None
                    else None
                ),

                # Token accounting provenance.
                'global_batch_size': int(args.global_batch_size),
                'seq_length': int(args.seq_length),
                'tokens_per_iter': tokens_per_iter,
                'measured_tokens': int(measured_tokens),

                # Diagnostic / provenance.
                'warmup_converged': converged,
                'warmup_wall_s': round(t_warmup, 2),
                'warmup_iters': n_warmup,
                'warmup_window_mean_s': round(warmup_final_mean, 4),
                'measure_seconds': profile_args.measure_seconds,
                'measure_iters': n_measure,
                'measure_wall_s': round(t_measure, 2),
                'build_s': round(t_build, 2),
                'save_s': round(t_save, 2),
                'mpu_reshard_s': round(t_mpu_reshard, 2),
                'reshard_wall_time_s': (
                    round(full_reshard_s, 2)
                    if full_reshard_s is not None
                    else None
                ),
                'reshard_is_wraparound': False,
            })

            if rank == 0:
                e_step = mean_cluster_power * mean_step_time
                conv_tag = "✓" if converged else "⚠"
                jtok_text = (
                    f"{joules_per_token:.6f}"
                    if joules_per_token is not None
                    else "nan"
                )

                print(
                    f"  → {conv_tag} "
                    f"P = {mean_cluster_power:.1f}W cluster, "
                    f"τ = {mean_step_time:.4f}s/step, "
                    f"r = {throughput_tok_s:,.0f} tok/s, "
                    f"E = {e_step:.1f}J/step, "
                    f"J/tok = {jtok_text} "
                    f"({n_warmup} warmup + {n_measure} measure iters)",
                    flush=True,
                )

            del step_args, config, model, optimizer, opt_param_scheduler
            free_model_memory(None, None, None)

        # ── Wraparound ───────────────────────────────────────────
        if len(profile_menu) >= 2:
            case0 = profile_menu[0]
            case0_micro_batch_size = case0.effective_micro_batch_size(
                default_micro_batch_size
            )

            if rank == 0:
                print(f"\n[wraparound] Reshard back to TP={case0.tp} "
                      f"PP={case0.pp} MBS={case0_micro_batch_size} "
                      f"for first-case reshard timing",
                      flush=True)

            mpu.destroy_model_parallel()

            update_args_for_strategy(
                args,
                case0,
                work_ckpt,
                is_first_phase=False,
            )
            apply_micro_batch_size_for_case(
                args,
                case0,
                default_micro_batch_size,
            )

            dist.barrier()

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=case0.tp,
                pipeline_model_parallel_size=case0.pp,
                order=case0.order or "tp-cp-ep-dp-pp",
            )
            reconfigure_microbatches_for_phase(args, rank)

            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                model_provider_func=model_provider_func,
                model_type=ModelType.encoder_or_decoder,
            )

            verify_parallel_state(
                model,
                expected_tp=case0.tp,
                expected_pp=case0.pp,
                expected_dp=n_gpus // (case0.tp * case0.pp),
                rank=rank,
            )

            wraparound_end_t = time.perf_counter()
            wraparound_reshard_s = wraparound_end_t - last_measure_end_t

            results[0]['reshard_wall_time_s'] = round(wraparound_reshard_s, 2)
            results[0]['reshard_is_wraparound'] = True

            if rank == 0:
                print(f"  wraparound reshard wall time: "
                      f"{wraparound_reshard_s:.2f}s "
                      f"(written into case 0's reshard_wall_time_s)",
                      flush=True)

            del model, optimizer, opt_param_scheduler
            free_model_memory(None, None, None)

    finally:
        sampler.stop()

    # ── 7. Aggregate transitions and emit calibration JSON ──────
    if rank == 0:
        transition_times = [
            r['reshard_wall_time_s']
            for r in results
            if r['reshard_wall_time_s'] is not None
        ]

        avg_reshard_s = (
            sum(transition_times) / len(transition_times)
            if transition_times
            else 0.0
        )

        n_unconverged = sum(
            1 for r in results
            if not r['warmup_converged']
        )
        micro_batch_sizes_profiled = sorted({
            int(r['micro_batch_size'])
            for r in results
        })

        out_dir = os.path.dirname(profile_args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_doc = {
            'n_gpus': n_gpus,
            'global_batch_size': int(args.global_batch_size),
            'default_micro_batch_size': default_micro_batch_size,
            'micro_batch_sizes_profiled': micro_batch_sizes_profiled,
            'seq_length': int(args.seq_length),
            'tokens_per_iter': int(args.global_batch_size) * int(args.seq_length),
            'warmup_min_seconds': profile_args.warmup_min_seconds,
            'warmup_max_seconds': profile_args.warmup_max_seconds,
            'warmup_window': profile_args.warmup_window,
            'warmup_tol': profile_args.warmup_tol,
            'measure_seconds': profile_args.measure_seconds,
            'average_reshard_wall_time_s': round(avg_reshard_s, 2),
            'reshard_observations': len(transition_times),
            'n_unconverged_warmups': n_unconverged,
            'strategies': results,
        }

        with open(profile_args.output, 'w') as f:
            json.dump(out_doc, f, indent=2)

        # Pretty summary.
        print(f"\n{'=' * 100}")
        print("  PROFILE COMPLETE")
        print(f"{'=' * 100}")
        print(
            f"  {'(TP,PP)':>10}  "
            f"{'DP':>4}  "
            f"{'MBS':>5}  "
            f"{'P (W)':>10}  "
            f"{'τ (s)':>10}  "
            f"{'tok/s':>14}  "
            f"{'J/tok':>10}  "
            f"{'warm':>5}  "
            f"{'reshard':>10}"
        )
        print("  " + "-" * 96)

        for r in results:
            tag = f"({r['tp']},{r['pp']})"

            if r['reshard_wall_time_s'] is None:
                rs = "—"
            else:
                suffix = "w" if r.get('reshard_is_wraparound') else ""
                rs = f"{r['reshard_wall_time_s']:.1f}s{suffix}"

            wc = "✓" if r['warmup_converged'] else "⚠"

            jtok = (
                f"{r['joules_per_token']:.6f}"
                if r['joules_per_token'] is not None
                else "nan"
            )

            print(
                f"  {tag:>10}  "
                f"{r['dp']:>4}  "
                f"{r['micro_batch_size']:>5}  "
                f"{r['power_w']:>10.1f}  "
                f"{r['step_time_s']:>10.4f}  "
                f"{r['throughput_tok_s']:>14,.0f}  "
                f"{jtok:>10}  "
                f"{wc:>5}  "
                f"{rs:>10}"
            )

        print("  " + "-" * 96)

        n_wrap = sum(
            1 for r in results
            if r.get('reshard_is_wraparound')
        )

        if n_wrap:
            print("  (w = wraparound observation: reshard back to first "
                  "case after the last)")

        if n_unconverged:
            print(f"\n  ⚠ {n_unconverged} case(s) did NOT converge in warmup. "
                  f"Their step_time_s and throughput_tok_s may be biased.")
            print("    Consider raising --warmup-max-seconds, --warmup-tol, "
                  "or --warmup-window.")

        print(f"\n  Average reshard wall time: {avg_reshard_s:.2f}s "
              f"({len(transition_times)} transitions observed)")
        print("  → Drop into scheduler_config.json as `switch_time_s`")
        print("  → Use `throughput_tok_s` as r(pi) in the scheduler/writeup")
        print(f"\n  Calibration written to: {profile_args.output}")
        print(f"{'=' * 100}\n", flush=True)

    dist.barrier()


if __name__ == '__main__':
    main()
