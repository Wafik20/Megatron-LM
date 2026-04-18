#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications: Wafik Tawfik (wat29@pitt.edu) — elastic parallelism reconfiguration

"""
elastic_pretrain.py
-------------------
In-process elastic 3D parallelism reconfiguration for GPT training.

Single torchrun launch, switches (TP, PP) configurations between training phases
without restarting processes. Eliminates process spawn, import, CUDA init, and
JIT compilation overhead on every reconfiguration.

Usage:
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 elastic_pretrain.py \
        --elastic-schedule schedule.json \
        [standard megatron args: --num-layers, --hidden-size, etc.]

Schedule JSON format:
    [
        {"phase": 1, "start_step": 0,   "end_step": 100, "tp": 1, "pp": 1},
        {"phase": 2, "start_step": 100, "end_step": 200, "tp": 2, "pp": 2}
    ]

Reconfiguration sequence between phases:
    save checkpoint -> free GPU memory -> destroy NCCL groups ->
    reinitialize parallel state -> RECONFIGURE microbatch calculator ->
    rebuild model/optimizer -> load+reshard checkpoint
"""

import argparse
import gc
import json
import os
import sys
import time
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

from gpt_builders import gpt_builder
from model_provider import model_provider


# ═══════════════════════════════════════════════════════════════
# Schedule
# ═══════════════════════════════════════════════════════════════

def load_and_validate_schedule(path, n_gpus, num_layers, num_attention_heads):
    with open(path) as f:
        raw = json.load(f)

    phases = []
    for i, entry in enumerate(raw):
        tp, pp = entry["tp"], entry["pp"]
        dp = n_gpus // (tp * pp)
        assert n_gpus % (tp * pp) == 0, \
            f"Phase {i+1}: TP={tp}*PP={pp} doesn't divide N={n_gpus}"
        assert num_layers % pp == 0, \
            f"Phase {i+1}: {num_layers} layers not divisible by PP={pp}"
        assert num_attention_heads % tp == 0, \
            f"Phase {i+1}: {num_attention_heads} heads not divisible by TP={tp}"

        phases.append({
            "phase": entry.get("phase", i + 1),
            "start_step": entry["start_step"],
            "end_step": entry["end_step"],
            "tp": tp, "pp": pp, "dp": dp,
        })

    for i in range(1, len(phases)):
        assert phases[i]["start_step"] == phases[i-1]["end_step"], \
            f"Gap between phase {phases[i-1]['phase']} and {phases[i]['phase']}"

    return phases


def schedule_needs_reshard(path):
    """
    Return True if any phase boundary in the schedule changes (TP, PP).

    Used to decide whether to request a fully-reshardable optimizer checkpoint.
    Reshardable saves require an allgather of the full optimizer state onto
    each rank, which can OOM on tight-memory configs (e.g. LLaMA-3B on DP=4
    with only 46 GB/GPU — the 10+ GB allgather temp buffer on top of training
    state exceeds the ceiling). We only pay that cost when an actual reshard
    is coming.

    Note: this runs BEFORE initialize_megatron, so we parse the raw JSON
    without any of the validation in load_and_validate_schedule.
    """
    with open(path) as f:
        raw = json.load(f)
    for i in range(1, len(raw)):
        if raw[i].get('tp') != raw[i-1].get('tp') \
                or raw[i].get('pp') != raw[i-1].get('pp'):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# NCCL Group Cleanup
# ═══════════════════════════════════════════════════════════════

def destroy_model_parallel_with_nccl_cleanup():
    """
    Enhanced destroy_model_parallel that actually destroys NCCL process groups.

    Stock Megatron only destroys Gloo groups and nulls NCCL group references,
    leaking NCCL communicators. We collect group objects before stock destroy
    nulls them, then explicitly destroy each one.
    """
    from megatron.core import parallel_state as mpu

    nccl_group_names = [
        '_MODEL_PARALLEL_GROUP',
        '_TENSOR_MODEL_PARALLEL_GROUP',
        '_PIPELINE_MODEL_PARALLEL_GROUP',
        '_DATA_PARALLEL_GROUP',
        '_DATA_PARALLEL_GROUP_WITH_CP',
        '_DATA_PARALLEL_GROUP_WITH_CP_AG',
        '_CONTEXT_PARALLEL_GROUP',
        '_EMBEDDING_GROUP',
        '_POSITION_EMBEDDING_GROUP',
        '_TENSOR_AND_DATA_PARALLEL_GROUP',
        '_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP',
        '_TENSOR_AND_CONTEXT_PARALLEL_GROUP',
        '_EXPERT_MODEL_PARALLEL_GROUP',
        '_EXPERT_TENSOR_PARALLEL_GROUP',
        '_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
        '_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP',
        '_EXPERT_DATA_PARALLEL_GROUP',
        '_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP',
        '_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP',
        '_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP',
    ]

    groups_to_destroy = set()
    for name in nccl_group_names:
        group = getattr(mpu, name, None)
        if group is not None:
            groups_to_destroy.add(group)

    pg_list = getattr(mpu, '_global_process_group_list', None)
    if pg_list:
        for g in pg_list:
            if g is not None:
                groups_to_destroy.add(g)

    mpu.destroy_model_parallel()

    world_pg = dist.GroupMember.WORLD
    pg_map = dist.distributed_c10d._world.pg_map

    destroyed = 0
    for group in groups_to_destroy:
        if group == world_pg:
            continue
        if pg_map.get(group, None) is None:
            continue
        try:
            dist.destroy_process_group(group)
            destroyed += 1
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"  [cleanup] Warning: failed to destroy group: {e}")

    if dist.get_rank() == 0:
        print(f"  [cleanup] Destroyed {destroyed} NCCL process groups")


# ═══════════════════════════════════════════════════════════════
# Microbatch calculator reset
# ═══════════════════════════════════════════════════════════════

def reconfigure_microbatches_for_phase(args, rank):
    """
    Refresh Megatron's num_microbatches_calculator singleton for the new phase.

    CRITICAL: without this call, phase 2 keeps phase 1's num_microbatches value
    because the calculator is a process-wide singleton initialized at first
    Megatron startup. When switching from e.g. DP=4,PP=1 (num_mb=2) to
    DP=1,PP=4 (num_mb=8), the stale value of 2 causes the pipeline to run with
    only 2 microbatches flowing through 4 stages — leaving GPUs idle in the
    pipeline bubble and producing silently-faster-but-less-work iterations.

    reconfigure_num_microbatches_calculator is the same helper Megatron uses
    internally for rampup schedules; it rebuilds the singleton in-place
    without disturbing the rest of the framework.
    """
    from megatron.core import num_microbatches_calculator as mbc
    from megatron.core import parallel_state as mpu

    dp_world = mpu.get_data_parallel_world_size()
    mbc.reconfigure_num_microbatches_calculator(
        rank=rank,
        rampup_batch_size=args.rampup_batch_size,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=dp_world,
    )

    if rank == 0:
        print(f"  [reconfig] microbatch calculator -> "
              f"num_microbatches = {mbc.get_num_microbatches()} "
              f"(global={args.global_batch_size}, micro={args.micro_batch_size}, "
              f"dp={dp_world})",
              flush=True)


# ═══════════════════════════════════════════════════════════════
# Mock Data + Forward Step (matching pretrain_gpt.py signatures)
# ═══════════════════════════════════════════════════════════════

def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a mock training batch."""
    from megatron.training import get_args
    args = get_args()

    device = torch.cuda.current_device()
    seq_len = args.seq_length
    mbs = args.micro_batch_size
    vocab = args.padded_vocab_size

    tokens = torch.randint(0, vocab, (mbs, seq_len), dtype=torch.long, device=device)
    labels = torch.randint(0, vocab, (mbs, seq_len), dtype=torch.long, device=device)
    loss_mask = torch.ones(mbs, seq_len, dtype=torch.float, device=device)
    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device) \
                        .unsqueeze(0).expand(mbs, -1).contiguous()
    packed_seq_params = None

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """Loss function matching pretrain_gpt.py: returns (loss, num_tokens, report)."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    return loss, num_tokens, report


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward step matching pretrain_gpt.py signature."""
    from megatron.training import get_timers
    from megatron.core.utils import get_attr_wrapped_model
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
        data_iterator, vp_stage
    )
    timers('batch-generator').stop()

    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels,
        loss_mask=loss_mask, packed_seq_params=packed_seq_params
    )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def infinite_data_iterator():
    while True:
        yield None


# ═══════════════════════════════════════════════════════════════
# Loss gathering across pipeline stages
# ═══════════════════════════════════════════════════════════════

def get_loss_for_logging(loss_dict):
    """
    Extract scalar loss value, broadcasting from the last PP stage to all ranks.

    train_step returns the loss on the last pipeline-parallel rank only; other
    PP ranks receive an empty loss_dict. In a PP=1 config every rank is the last
    stage so this is a no-op, but with PP>1 rank 0 would print 0.0 without this
    broadcast. We broadcast within the pipeline group so every rank in the
    pipeline (and transitively global rank 0, which is PP-rank-0 of its pipeline)
    sees the correct value.
    """
    from megatron.core import parallel_state as mpu

    lm_loss_tensor = loss_dict.get('lm loss', None)
    device = torch.cuda.current_device()

    if lm_loss_tensor is not None and torch.is_tensor(lm_loss_tensor):
        local_loss = lm_loss_tensor.detach().float().view(1)
    else:
        local_loss = torch.zeros(1, device=device, dtype=torch.float32)

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        torch.distributed.broadcast(
            local_loss,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )

    return local_loss.item()


# ═══════════════════════════════════════════════════════════════
# Parallel-state verification diagnostic
# ═══════════════════════════════════════════════════════════════

def verify_parallel_state(model, expected_tp, expected_pp, expected_dp, rank):
    """
    Print what Megatron ACTUALLY built vs what the schedule asked for.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core import num_microbatches_calculator as mbc

    actual_tp = mpu.get_tensor_model_parallel_world_size()
    actual_pp = mpu.get_pipeline_model_parallel_world_size()
    actual_dp = mpu.get_data_parallel_world_size()
    actual_num_mb = mbc.get_num_microbatches()

    def count_decoder_layers(m):
        inner = m
        for attr in ('module', 'module'):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            else:
                break
        layers = getattr(getattr(inner, 'decoder', None), 'layers', None)
        if layers is None:
            return -1
        try:
            return len(layers)
        except TypeError:
            return sum(1 for _ in layers)

    models = model if isinstance(model, list) else [model]
    layers_per_chunk = [count_decoder_layers(m) for m in models]
    total_local_layers = sum(l for l in layers_per_chunk if l >= 0)

    total_params = 0
    for m in models:
        total_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

    my_info = torch.tensor(
        [rank, actual_tp, actual_pp, actual_dp, total_local_layers, total_params // 1_000_000],
        device=torch.cuda.current_device(), dtype=torch.long,
    )
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(my_info) for _ in range(world_size)]
    dist.all_gather(gathered, my_info)

    if rank == 0:
        ok_tp = (actual_tp == expected_tp)
        ok_pp = (actual_pp == expected_pp)
        ok_dp = (actual_dp == expected_dp)
        tag_tp = "✓" if ok_tp else "✗"
        tag_pp = "✓" if ok_pp else "✗"
        tag_dp = "✓" if ok_dp else "✗"

        print(f"  [verify] Parallel sizes  "
              f"TP: {actual_tp}{tag_tp}(want {expected_tp})  "
              f"PP: {actual_pp}{tag_pp}(want {expected_pp})  "
              f"DP: {actual_dp}{tag_dp}(want {expected_dp})  "
              f"|  num_microbatches = {actual_num_mb}",
              flush=True)

        print(f"  [verify] Per-rank layer & param counts:")
        for g in gathered:
            r, tp_, pp_, dp_, nlayers, nparams_m = [x.item() for x in g]
            print(f"  [verify]   rank {r}: {nlayers:>3} decoder layers, "
                  f"~{nparams_m}M params", flush=True)

        from megatron.training import get_args
        args = get_args()
        expected_per_rank = args.num_layers // actual_pp
        any_rank_full = any(g[4].item() == args.num_layers for g in gathered) \
                        and actual_pp > 1
        if any_rank_full:
            print(f"  [verify] ⚠ WARNING: PP={actual_pp} but some rank holds all "
                  f"{args.num_layers} layers. Reshard may not have applied!",
                  flush=True)

        expected_num_mb = args.global_batch_size // (args.micro_batch_size * actual_dp)
        if actual_num_mb != expected_num_mb:
            print(f"  [verify] ⚠ WARNING: num_microbatches = {actual_num_mb} "
                  f"but expected {expected_num_mb} for "
                  f"global={args.global_batch_size} / "
                  f"(micro={args.micro_batch_size} × dp={actual_dp}). "
                  f"Microbatch calculator may be stale!",
                  flush=True)


# ═══════════════════════════════════════════════════════════════
# Args Helpers
# ═══════════════════════════════════════════════════════════════

def update_args_for_phase(args, phase, ckpt_dir, is_first_phase):
    args.tensor_model_parallel_size = phase['tp']
    args.pipeline_model_parallel_size = phase['pp']
    args.save = ckpt_dir
    args.save_interval = 999999

    if is_first_phase:
        args.load = None
        args.iteration = 0
        args.consumed_train_samples = 0
    else:
        args.load = ckpt_dir


def free_model_memory(model, optimizer, opt_param_scheduler):
    del model
    del optimizer
    del opt_param_scheduler

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if dist.get_rank() == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  [memory] After cleanup: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. Extract elastic args before Megatron sees sys.argv ─────
    elastic_parser = argparse.ArgumentParser(add_help=False)
    elastic_parser.add_argument('--elastic-schedule', required=True,
                                help='Path to schedule JSON')
    elastic_parser.add_argument('--elastic-work-dir',
                                default='/tmp/elastic_training',
                                help='Directory for checkpoints and results')
    elastic_args, megatron_argv = elastic_parser.parse_known_args()

    ckpt_dir = os.path.join(elastic_args.elastic_work_dir, 'elastic_ckpt')

    # Only request fully-reshardable optimizer checkpoints when the schedule
    # actually changes (TP, PP) between phases. Reshardable saves require an
    # allgather of the full optimizer state onto each rank (~10 GB for
    # LLaMA-3B), which can OOM on tight-memory configs (e.g. DP=4 on 46 GB
    # L40 with a large model, where training state alone is ~36 GB). We only
    # pay that cost when we're actually going to reshard.
    #
    # torchrun exports RANK; use it here because dist isn't initialized yet.
    _launch_rank = int(os.environ.get("RANK", "0"))
    needs_reshard = schedule_needs_reshard(elastic_args.elastic_schedule)

    extra_megatron = [
        '--ckpt-format', 'torch_dist',
        '--use-distributed-optimizer',
        '--auto-detect-ckpt-format',
        '--override-opt_param-scheduler',
        '--save', ckpt_dir,
        '--save-interval', '999999',
    ]
    if needs_reshard:
        extra_megatron.append('--dist-ckpt-optim-fully-reshardable')
        if _launch_rank == 0:
            print("[elastic] schedule has reshards between phases; "
                  "using fully-reshardable optimizer checkpoints",
                  flush=True)
    else:
        if _launch_rank == 0:
            print("[elastic] schedule is single-config (no reshards); "
                  "using standard optimizer checkpoints "
                  "(~10 GB saved during checkpoint save)",
                  flush=True)

    sys.argv = [sys.argv[0]] + megatron_argv + extra_megatron

    # ── 2. Initialize Megatron (one-time cost) ────────────────────
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer, train_step
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training import get_args, get_timers
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

    # ── 3. Load and validate schedule ─────────────────────────────
    schedule = load_and_validate_schedule(
        elastic_args.elastic_schedule, n_gpus,
        args.num_layers, args.num_attention_heads,
    )

    assert schedule[0]['tp'] == args.tensor_model_parallel_size, \
        f"First phase TP={schedule[0]['tp']} != Megatron TP={args.tensor_model_parallel_size}"
    assert schedule[0]['pp'] == args.pipeline_model_parallel_size, \
        f"First phase PP={schedule[0]['pp']} != Megatron PP={args.pipeline_model_parallel_size}"

    if rank == 0:
        print("\n" + "=" * 70)
        print("  ELASTIC 3D PARALLELISM - IN-PROCESS RECONFIGURATION")
        print("=" * 70)
        print(f"  GPUs:           {n_gpus}")
        print(f"  Phases:         {len(schedule)}")
        print(f"  Total steps:    {schedule[0]['start_step']} -> {schedule[-1]['end_step']}")
        print(f"  Megatron init:  {t_init:.1f}s (one-time cost)")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"  Reshardable:    {needs_reshard}")
        print()
        for p in schedule:
            print(f"  Phase {p['phase']}: steps {p['start_step']:>5}->{p['end_step']:<5} "
                  f"  DP={p['dp']} TP={p['tp']} PP={p['pp']}")
        print("=" * 70 + "\n", flush=True)

    # ── 4. Run phases ─────────────────────────────────────────────
    phase_results = []
    data_iter = RerunDataIterator(iter(infinite_data_iterator()))

    for phase_idx, phase in enumerate(schedule):
        tp, pp, dp = phase['tp'], phase['pp'], phase['dp']
        is_first = (phase_idx == 0)

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"  PHASE {phase['phase']}: steps {phase['start_step']}->{phase['end_step']}  "
                  f"(DP={dp}, TP={tp}, PP={pp})")
            print(f"{'=' * 70}", flush=True)

        t_reconfig_start = time.perf_counter()

        # ── 4a. Reconfigure parallel state (skip for first phase) ──
        if not is_first:
            if rank == 0:
                print("  [reconfig] Destroying old parallel state...", flush=True)
            t0 = time.perf_counter()
            mpu.destroy_model_parallel()
            if rank == 0:
                print(f"  [reconfig] destroy_model_parallel: {time.perf_counter()-t0:.2f}s",
                      flush=True)

            update_args_for_phase(args, phase, ckpt_dir, is_first)

            if rank == 0:
                print(f"  [reconfig] Reinitializing model parallel: TP={tp}, PP={pp}",
                      flush=True)
            t0 = time.perf_counter()

            dist.barrier()
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
            )
            if rank == 0:
                print(f"  [reconfig] initialize_model_parallel: {time.perf_counter()-t0:.2f}s",
                      flush=True)

            # CRITICAL: refresh the microbatch calculator for the new DP size.
            # Without this, phase 2 keeps phase 1's num_microbatches and
            # silently under-fills the pipeline.
            reconfigure_microbatches_for_phase(args, rank)
        else:
            update_args_for_phase(args, phase, ckpt_dir, is_first)

        # ── 4b. Build model + optimizer ───────────────────────────
        if rank == 0:
            print("  [build] Constructing model and optimizer...", flush=True)
        t0 = time.perf_counter()

        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider_func, ModelType.encoder_or_decoder
        )
        t_build = time.perf_counter() - t0

        iteration = getattr(args, 'iteration', 0)

        if rank == 0:
            print(f"  [build] Model + optimizer + checkpoint: {t_build:.2f}s")
            print(f"  [build] Resuming from iteration: {iteration}")
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"  [build] GPU memory allocated: {alloc:.2f}GB", flush=True)

        # Verify the parallel state + microbatch count actually match the schedule
        verify_parallel_state(model, expected_tp=tp, expected_pp=pp,
                              expected_dp=dp, rank=rank)

        t_reconfig = time.perf_counter() - t_reconfig_start

        forward_backward_func = get_forward_backward_func()

        # ── 4c. Training loop ─────────────────────────────────────
        config = core_transformer_config_from_args(args)
        num_steps = phase['end_step'] - iteration
        if rank == 0:
            print(f"\n  [train] Training {num_steps} steps "
                  f"(iter {iteration}->{phase['end_step']})...", flush=True)

        t_train_start = time.perf_counter()
        t_log_window = t_train_start
        losses_this_phase = []
        log_interval = args.log_interval

        for step_i in range(num_steps):
            loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros, max_attn_logit = train_step(
                forward_step, data_iter, model, optimizer, opt_param_scheduler,
                config, forward_backward_func
            )
            iteration += 1

            lm_loss = get_loss_for_logging(loss_dict)
            losses_this_phase.append(lm_loss)

            if rank == 0 and iteration % log_interval == 0:
                t_now = time.perf_counter()
                iter_ms = (t_now - t_log_window) * 1000.0 / log_interval
                t_log_window = t_now

                avg_recent = sum(losses_this_phase[-log_interval:]) / \
                             min(log_interval, len(losses_this_phase))
                gn = grad_norm if grad_norm is not None else 0.0
                skip = " [SKIPPED]" if skipped_iter else ""
                print(f"    iter {iteration:>6}/{phase['end_step']} | "
                      f"loss: {avg_recent:.4f} | grad_norm: {gn:.4f} | "
                      f"{iter_ms:5.0f}ms/it{skip}",
                      flush=True)

        t_train = time.perf_counter() - t_train_start
        avg_iter_ms = (t_train / max(num_steps, 1)) * 1000

        final_loss = losses_this_phase[-1] if losses_this_phase else None
        if rank == 0:
            print(f"\n  [train] Done. {num_steps} steps in {t_train:.1f}s "
                  f"({avg_iter_ms:.0f}ms/iter)")
            if final_loss:
                print(f"  [train] Final loss: {final_loss:.4f}", flush=True)

        # ── 4d. Save checkpoint ───────────────────────────────────
        # Skip the save entirely if we don't need to reshard AND this is the
        # last phase. Saves ~20s + ~10 GB of VRAM headroom. We still save
        # between phases that reshard, because the next phase has to load.
        is_last_phase = (phase_idx == len(schedule) - 1)
        # Skip the final save unconditionally. By definition, no subsequent phase
        # will load it, so the reshardable allgather burns memory for nothing —
        # and for tight-memory configs (e.g. DP=4 on LLaMA-3B) that allgather
        # can OOM. Mid-schedule saves still happen (the reshard depends on them).
        skip_save = is_last_phase

        if skip_save:
            if rank == 0:
                print(f"  [ckpt] Skipping final checkpoint "
                      f"(single-config run, no reshard needed)", flush=True)
            t_save = 0.0
        else:
            if rank == 0:
                print(f"  [ckpt] Saving checkpoint at iteration {iteration}...",
                      flush=True)
            t0 = time.perf_counter()

            args.save = ckpt_dir
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)

            t_save = time.perf_counter() - t0
            if rank == 0:
                print(f"  [ckpt] Saved in {t_save:.2f}s", flush=True)

        result = {
            "phase": phase['phase'],
            "dp": dp, "tp": tp, "pp": pp,
            "start_step": phase['start_step'],
            "end_step": phase['end_step'],
            "reconfig_s": round(t_reconfig, 2),
            "train_s": round(t_train, 2),
            "save_s": round(t_save, 2),
            "avg_iter_ms": round(avg_iter_ms, 1),
            "final_loss": round(final_loss, 6) if final_loss is not None else None,
            "num_steps": num_steps,
        }
        phase_results.append(result)

        if phase_idx < len(schedule) - 1:
            if rank == 0:
                print(f"  [cleanup] Freeing model memory for next phase...", flush=True)
            free_model_memory(model, optimizer, opt_param_scheduler)
        else:
            del model, optimizer, opt_param_scheduler

    # ── 5. Summary ────────────────────────────────────────────────
    if rank == 0:
        total_wall = sum(r['reconfig_s'] + r['train_s'] + r['save_s']
                         for r in phase_results)
        total_train = sum(r['train_s'] for r in phase_results)
        total_reconfig = sum(r['reconfig_s'] for r in phase_results)
        total_save = sum(r['save_s'] for r in phase_results)

        print(f"\n{'=' * 70}")
        print("  ELASTIC TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Megatron init:       {t_init:.1f}s (one-time)")
        print(f"  Total wall:          {total_wall:.1f}s")
        print(f"  Total training:      {total_train:.1f}s")
        print(f"  Total reconfig:      {total_reconfig:.1f}s")
        print(f"  Total checkpoint:    {total_save:.1f}s")
        if total_wall > 0:
            print(f"  Training fraction:   {total_train/total_wall*100:.1f}%")

        print(f"\n  {'Phase':<6} {'Config':<22} {'Reconfig':>9} {'Train':>8} "
              f"{'Iter':>9} {'Loss':>10}")
        print("  " + "-" * 67)
        for r in phase_results:
            config_str = f"dp{r['dp']}_tp{r['tp']}_pp{r['pp']}"
            loss_str = f"{r['final_loss']:.4f}" if r['final_loss'] else "N/A"
            print(f"  {r['phase']:<6} {config_str:<22} {r['reconfig_s']:>8.1f}s "
                  f"{r['train_s']:>7.1f}s {r['avg_iter_ms']:>7.0f}ms/it {loss_str:>10}")

        results_file = os.path.join(elastic_args.elastic_work_dir, 'elastic_results.json')
        os.makedirs(elastic_args.elastic_work_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                'n_gpus': n_gpus,
                'megatron_init_s': round(t_init, 2),
                'total_wall_s': round(total_wall, 2),
                'total_train_s': round(total_train, 2),
                'total_reconfig_s': round(total_reconfig, 2),
                'phases': phase_results,
            }, f, indent=2)
        print(f"\n  Results: {results_file}")
        print(f"{'=' * 70}\n", flush=True)

    dist.barrier()


if __name__ == '__main__':
    main()