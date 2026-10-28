"""CPU-only tests of device selection and process-group device binding."""
import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import os
import socket

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("slurm", [False, True])
@pytest.mark.parametrize("count,local,expected", [(1, 3, 0), (4, 3, 3), (4, 0, 0)])
def test_selection_precedes_nccl_initialization(monkeypatch, slurm, count, local, expected):
    events = []
    monkeypatch.setenv("PROFILE_SLURM_LAUNCH", "1" if slurm else "0")
    monkeypatch.setenv("SLURM_LOCALID", str(local))
    monkeypatch.setenv("LOCAL_RANK", str(local))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bound" if count == 1 else "2,3,4,6")
    cuda = SimpleNamespace(device_count=lambda: count,
                           set_device=lambda d: events.append(("select", d)))
    dist = SimpleNamespace(is_initialized=lambda: False,
                           init_process_group=lambda **kw: events.append(("init", kw)))
    torch = SimpleNamespace(cuda=cuda, distributed=dist, device=lambda d: d)
    args = SimpleNamespace(rank=7, world_size=8, local_rank=local, cuda_graph_impl="none",
                           distributed_backend="nccl", distributed_timeout_minutes=10,
                           fake_process_group=False, flight_recorder_dump_path=None)
    select = load_function(ROOT / "profile_strategies.py", "select_profile_cuda_device",
                           dict(os=os, socket=socket, torch=torch))
    assert select(args) == expected
    assert args.local_rank == expected
    assert os.environ["SLURM_LOCALID"] == str(local)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ("GPU-bound" if count == 1 else "2,3,4,6")
    for key in ("TORCH_FR_DUMP_TEMP_FILE", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"):
        monkeypatch.delenv(key, raising=False)
    initialize = load_function(ROOT / "megatron/training/initialize.py", "_initialize_distributed",
        dict(torch=torch, os=os, get_args=lambda: args, print_rank_0=lambda *a: None,
             timedelta=timedelta,
             inprocess_restart=SimpleNamespace(maybe_force_nccl_backend_init=lambda d: None)))
    initialize(None, None, None, skip_model_parallel_init=True)
    assert events[:2] == [("select", expected), ("select", expected)]
    assert events[2][0] == "init"
    # PyTorch uses this bound device for its first NCCL barrier, before any tensors.
    assert events[2][1]["device_id"] == f"cuda:{expected}"
    assert events[2][1]["rank"] == 7


@pytest.mark.parametrize("count,local", [(0, 0), (4, 4), (1, -1)])
def test_invalid_selection_fails_before_cuda_set_device(monkeypatch, count, local):
    monkeypatch.setenv("PROFILE_SLURM_LAUNCH", "1")
    monkeypatch.setenv("SLURM_LOCALID", str(local))
    events = []
    select = load_function(ROOT / "profile_strategies.py", "select_profile_cuda_device",
        dict(os=os, torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count,
                     set_device=lambda d: events.append(d)))))
    with pytest.raises(ValueError, match="visible CUDA device"):
        select(SimpleNamespace(local_rank=local, rank=7))
    assert events == []


def test_profiler_selects_before_runtime_and_megatron_init():
    tree = ast.parse((ROOT / "profile_strategies.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {n.func.id: n.lineno for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls["parse_and_validate_args"] < calls["run_preflight"]
    assert calls["run_preflight"] < calls["initialize_runtime_services"]
    assert calls["run_preflight"] < calls["initialize_megatron"]
    assert calls["parse_and_validate_args"] < calls["select_profile_cuda_device"]
    assert calls["select_profile_cuda_device"] < calls["initialize_runtime_services"]
    assert calls["select_profile_cuda_device"] < calls["initialize_megatron"]


@pytest.mark.parametrize("count", [2, 4])
def test_task_bound_contract_rejects_partial_or_shared_visibility(monkeypatch, count):
    monkeypatch.setenv("PROFILE_TASK_GPUS", "1")
    monkeypatch.setenv("PROFILE_SLURM_LAUNCH", "1")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    select = load_function(ROOT / "profile_strategies.py", "select_profile_cuda_device",
        dict(os=os, socket=socket, torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count))))
    with pytest.raises(ValueError, match="one task-bound visible CUDA device"):
        select(SimpleNamespace(local_rank=0, rank=0))


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("duplicate", [False, True])
def test_task_bound_ranks_resolve_distinct_uuids(monkeypatch, duplicate, world_size):
    from types import ModuleType
    import sys
    monkeypatch.setenv("PROFILE_TASK_GPUS", "1")
    monkeypatch.setenv("PROFILE_SLURM_LAUNCH", "1")
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_group=lambda: "tp",
        get_pipeline_model_parallel_group=lambda: "pp",
        get_data_parallel_group=lambda: "dp")
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    records = []
    # Slurm cyclic placement: even global ranks on node76, odd on node77.
    # singleton CVD values are job-relative IDs, never CUDA ordinals.
    masks = (["0", "1", "2", "3"], ["2", "0", "3", "1"])
    for rank in range(world_size):
        node, local = rank % 2, rank // 2
        monkeypatch.setenv("SLURM_LOCALID", str(local))
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", masks[node][local])
        selected = []
        uuid = f"GPU-node{node}-{local if not duplicate or rank != world_size - 2 else 0}"
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=selected.append,
            current_device=lambda: selected[-1],
            get_device_properties=lambda device: SimpleNamespace(uuid=uuid, name="RTX6000"))
        namespace = dict(os=os, socket=SimpleNamespace(gethostname=lambda: f"gpu-n{76+node}"),
                         torch=SimpleNamespace(cuda=cuda))
        select = load_function(ROOT / "profile_strategies.py", "select_profile_cuda_device", namespace)
        assert select(SimpleNamespace(rank=rank, local_rank=0)) == 0
        def gather(output, record):
            records.append(record.copy())
            # Each call exercises the real collector with all observed UUIDs.
            output[:] = [dict(record, rank=r, hostname=f"gpu-n{76+r%2}",
                              gpu_uuid=f"GPU-node{r%2}-{r//2}") for r in range(world_size)]
            output[rank] = record
        namespace["dist"] = SimpleNamespace(get_rank=lambda: rank, get_world_size=lambda: world_size,
            all_gather_object=gather, get_process_group_ranks=lambda group: [rank])
        collect = load_function(ROOT / "profile_strategies.py", "collect_rank_mapping", namespace)
        if duplicate and rank == world_size - 2:
            with pytest.raises(RuntimeError, match="distinct GPU"):
                collect()
        else:
            collect()
    if not duplicate:
        assert len({r["gpu_uuid"] for r in records}) == world_size
        for node in ("gpu-n76", "gpu-n77"):
            group = [r for r in records if r["hostname"] == node]
            assert {r["local_rank"] for r in group} == set(range(world_size // 2))
            assert {r["cuda_device"] for r in group} == {0}
