"""CPU checks for restored profiler control logic (not GPU integration)."""
import ast
from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import List, Optional, Tuple

import pytest


@pytest.fixture
def profiler_logic():
    path = Path(__file__).resolve().parents[1] / "profile_strategies.py"
    wanted = {"ProfileCase", "_coerce_positive_int", "load_profile_menu",
              "validate_micro_batch_cases_or_die", "run_for_seconds", "validate_profile_order",
              "collect_rank_mapping"}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    namespace = dict(dataclass=dataclass, List=List, Optional=Optional, Tuple=Tuple,
                     json=json, sys=sys)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_menu_and_batch_validation(profiler_logic, tmp_path):
    path = tmp_path / "menu.json"
    path.write_text(json.dumps([{"tp": 1, "pp": 1, "micro_batch_size": 2},
                               {"tp": 2, "pp": 2, "micro_batch_size": 1}]))
    cases = profiler_logic["load_profile_menu"](path)
    profiler_logic["validate_micro_batch_cases_or_die"](cases, 4, 16, 2, 0)
    with pytest.raises(SystemExit):
        profiler_logic["validate_micro_batch_cases_or_die"](cases, 4, 3, 2, 0)
    path.write_text(json.dumps([{"tp": 1, "pp": 1}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        profiler_logic["load_profile_menu"](path)


def test_measurement_uses_broadcast_stop_on_nonzero_rank(profiler_logic):
    from types import SimpleNamespace
    class Flag(list):
        def item(self):
            return self[0]
    flag = Flag([0])
    broadcasts = []
    steps = []
    barriers = []
    def broadcast(value, src):
        broadcasts.append(src)
        value[0] = int(len(broadcasts) == 3)
    profiler_logic.update(
        torch=SimpleNamespace(zeros=lambda *a, **kw: flag, int32=None,
                              cuda=SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None)),
        dist=SimpleNamespace(get_rank=lambda: 1, broadcast=broadcast,
                             barrier=lambda: barriers.append(len(steps))),
        time=SimpleNamespace(perf_counter=lambda: 1000.0),
    )
    count, _ = profiler_logic["run_for_seconds"](0.01, lambda: steps.append(1))
    assert count == 3 and len(steps) == 3 and broadcasts == [0, 0, 0]
    assert barriers == [0, 3]


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_strategy_refreshes_parallel_defaults(sequence_parallel, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    # Load the real config modules without Megatron's CUDA-dependent package init.
    root = Path(__file__).resolve().parents[1]
    for name, relative_path in (
        ("megatron.core.tensor_parallel.gtp_parallel_layout",
         "megatron/core/tensor_parallel/gtp_parallel_layout.py"),
        ("megatron.core.model_parallel_config", "megatron/core/model_parallel_config.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / relative_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    ModelParallelConfig = module.ModelParallelConfig

    path = Path(__file__).resolve().parents[1] / "online_elastic_pretrain.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "update_args_for_strategy"]
    namespace = {"Strategy": SimpleNamespace}
    exec(compile(tree, str(path), "exec"), namespace)
    update = namespace["update_args_for_strategy"]
    args = SimpleNamespace(world_size=4 if sequence_parallel else 2,
                           sequence_parallel=sequence_parallel)
    # Include PP changes and the wraparound to the initial TP.
    layouts = [(2, 1), (4, 1), (2, 2), (2, 1)] if sequence_parallel else [
        (1, 1), (2, 1), (1, 2), (1, 1)]
    for index, (tp, pp) in enumerate(layouts):
        update(args, SimpleNamespace(tp=tp, pp=pp), "/checkpoint", index == 0)
        assert args.tensor_parallel_num_weight_shards == tp
        assert args.expert_tensor_parallel_size == tp
        assert args.expert_tensor_parallel_num_weight_shards == tp
        assert args.gtp_weight_remat_size == args.expert_gtp_weight_remat_size == 1
        assert args.tensor_parallel_num_sequence_shards == (tp if sequence_parallel else 1)
        assert args.gtp_remat_num_sequence_shards == 1
        assert args.transformer_pipeline_model_parallel_size == pp
        assert args.data_parallel_size == args.batch_parallel_size == args.world_size // (tp * pp)
        assert args.save == "/checkpoint"
        assert args.load == (None if index == 0 else "/checkpoint")
        import dataclasses
        import torch
        fields = {field.name for field in dataclasses.fields(ModelParallelConfig)}
        config = ModelParallelConfig(**{key: value for key, value in vars(args).items()
                                       if key in fields}, pipeline_dtype=torch.bfloat16)
        assert config.gtp_weight_remat_size == 1


@pytest.mark.parametrize("order", ["tp-dp-pp", "tp-pp-dp", "dp-tp-pp",
                                   "dp-pp-tp", "pp-tp-dp", "pp-dp-tp"])
def test_static_order_menu(profiler_logic, tmp_path, order):
    path = tmp_path / "menu.json"
    path.write_text(json.dumps([{"tp": 2, "pp": 2, "order": order},
                               {"tp": 1, "pp": 2, "order": order}]))
    cases = profiler_logic["load_profile_menu"](path)
    assert [case.order for case in cases] == [order, order]


@pytest.mark.parametrize("order", ["tp-dp", "tp-tp-pp", "tp-cp-dp-pp", "", 123])
def test_invalid_profile_order(profiler_logic, order):
    with pytest.raises(ValueError):
        profiler_logic["validate_profile_order"](order)


def test_menu_rejects_order_switching(profiler_logic, tmp_path):
    path = tmp_path / "menu.json"
    for second_order in ("pp-tp-dp", None):
        path.write_text(json.dumps([{"tp": 2, "pp": 2, "order": "tp-dp-pp"},
                                   {"tp": 1, "pp": 2, "order": second_order}]))
        with pytest.raises(ValueError, match="separate profiler runs"):
            profiler_logic["load_profile_menu"](path)


def test_real_rank_generator_order_mappings():
    from itertools import permutations

    path = Path(__file__).resolve().parents[1] / "megatron/core/parallel_state.py"
    tree = ast.parse(path.read_text())
    wanted = {"RankGenerator", "generate_masked_orthogonal_rank_groups"}
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    namespace = {"List": List}
    exec(compile(tree, str(path), "exec"), namespace)
    RankGenerator = namespace["RankGenerator"]
    axes = ("tp", "dp", "pp")
    for sizes, expected_count in [((2, 2, 2), 6), ((2, 1, 2), 2),
                                  ((1, 2, 1), 1), ((1, 1, 1), 1)]:
        signatures = set()
        for order in permutations(axes):
            generator = RankGenerator(**dict(zip(axes, sizes)), cp=1, ep=1,
                                      order="-".join(order))
            signature = tuple(tuple(tuple(group) for group in generator.get_ranks(axis))
                              for axis in axes)
            signatures.add(signature)
            stride = 1
            for axis in order:
                size = dict(zip(axes, sizes))[axis]
                assert generator.get_ranks(axis)[0] == list(range(0, size * stride, stride))
                stride *= size
        assert len(signatures) == expected_count


def test_mapping_collector_uses_observed_groups(profiler_logic, monkeypatch):
    from types import ModuleType, SimpleNamespace

    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_group=lambda: "tp",
        get_pipeline_model_parallel_group=lambda: "pp",
        get_data_parallel_group=lambda: "dp",
    )
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    groups = {"tp": [1, 3], "pp": [0, 1], "dp": [1]}
    def gather(output, local):
        output[:] = [{"rank": 0, "hostname": "node-b", "gpu_uuid": "GPU-0"}, local]
    profiler_logic.update(
        torch=SimpleNamespace(cuda=SimpleNamespace(
            current_device=lambda: 7,
            get_device_properties=lambda device: SimpleNamespace(uuid=f"GPU-{device}", name="test GPU"))),
        dist=SimpleNamespace(get_rank=lambda: 1, get_world_size=lambda: 2,
                             get_process_group_ranks=groups.__getitem__, all_gather_object=gather),
        socket=SimpleNamespace(gethostname=lambda: "node-a"),
        os=SimpleNamespace(environ={"PROFILE_SLURM_LAUNCH": "1", "LOCAL_RANK": "0", "SLURM_LOCALID": "1",
                                      "CUDA_VISIBLE_DEVICES": "4,7"}),
    )
    mapping = profiler_logic["collect_rank_mapping"]()
    assert mapping == [{"rank": 0, "hostname": "node-b", "gpu_uuid": "GPU-0"}, {
        "rank": 1, "hostname": "node-a", "local_rank": 1, "cuda_device": 7,
        "cuda_visible_devices": "4,7", "gpu_uuid": "GPU-7", "gpu_name": "test GPU",
        "tp_group": [1, 3], "pp_group": [0, 1], "dp_group": [1],
    }]


def test_power_is_summed_over_world():
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "online_elastic_pretrain.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "sample_cluster_power"]
    class Scalar(list):
        def item(self):
            return self[0]
    calls = []
    def reduce(value, op):
        calls.append(op)
        value[0] += 7 * 100  # Seven other GPUs, including the remote node.
    namespace = dict(NvmlPowerSampler=object,
        torch=SimpleNamespace(tensor=lambda values, **kw: Scalar(values), float32=None,
                              cuda=SimpleNamespace(current_device=lambda: 0)),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(SUM="sum")))
    exec(compile(tree, str(path), "exec"), namespace)
    assert namespace["sample_cluster_power"](SimpleNamespace(consume_mean=lambda: 100)) == 800
    assert calls == ["sum"]
