# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Check rank grouping and conservative admission before any native work."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pypto.runtime.operator_recovery import OperatorRecoveryConfig, _get_recovery, _RecoveryGraph


class Args:
    def __init__(self):
        self.dependencies = []
        self.bindings = []

    def add_dep_wait(self, dependency):
        self.dependencies.append(dependency)

    def add_domain_binding(self, binding):
        self.bindings.append(binding)

    def set_recovery_metadata(self, metadata):
        self.metadata = metadata


@pytest.fixture
def graph(monkeypatch):
    api = ModuleType("simpler.task_interface")
    for name in ("RecoveryArgBinding", "RecoveryMetadata", "TaskDomainBinding"):
        setattr(api, name, SimpleNamespace)
    api.RecoveryArgKind = SimpleNamespace(WINDOW_TENSOR="window", COMM_CONTEXT="context")
    api.ReplaySafety = SimpleNamespace(REPLAY_SAFE="safe", UNKNOWN="unknown")
    monkeypatch.setitem(sys.modules, "simpler.task_interface", api)
    capture = _RecoveryGraph(OperatorRecoveryConfig(replay_safe=("exchange",)), 2, {"exchange": 2})
    contexts = [
        SimpleNamespace(
            domain_rank=rank,
            domain_size=2,
            recovery_binding=SimpleNamespace(
                allocation_id=0, generation=4, local_window_base=10000 + rank * 1000
            ),
        )
        for rank in range(2)
    ]
    return capture, contexts


def test_rank_major_capture_keeps_graph_order_and_explicit_safety(graph):
    capture, contexts = graph
    config = object()
    all_args = {}
    for rank in range(2):
        previous = None
        for sequence, name in enumerate(("warmup", "exchange", "finish")):
            args = Args()
            ctx = contexts[rank]
            capture.annotate(
                args,
                [
                    ("window", 2, ctx, SimpleNamespace(base=ctx.recovery_binding.local_window_base + 256)),
                    ("context", 1, ctx, None),
                ],
            )
            if previous is not None:
                capture.add_dependency(args, previous)
            previous = capture.submit(sequence, args, config, rank, name)
            all_args[rank, sequence] = args
    orch = MagicMock()
    orch.submit_next_level_group.side_effect = ["A", "B", "C"]
    capture.flush(orch)
    assert orch.submit_next_level_group.call_count == 3
    for sequence, call in enumerate(orch.submit_next_level_group.call_args_list):
        assert call.kwargs["workers"] == [0, 1]
        assert call.args[1] == [all_args[0, sequence], all_args[1, sequence]]
        for args in call.args[1]:
            assert args.dependencies == ([] if sequence == 0 else ["ABC"[sequence - 1]])
            assert args.metadata.participants == [0, 1]
            assert args.metadata.op_seq == sequence + 1
            assert args.metadata.replay_safety == ("safe" if sequence == 1 else "unknown")
            assert [(e.kind, e.index, e.offset) for e in args.bindings[0].arguments] == [
                ("window", 2, 256),
                ("context", 1, 0),
            ]
    assert all_args[0, 0].metadata.logical_run_id == all_args[1, 2].metadata.logical_run_id


@pytest.mark.parametrize("misuse", ["missing-rank", "different-callable", "forward-dependency", "wrong-rank"])
def test_unsafe_capture_never_admits_a_task(graph, misuse):
    capture, contexts = graph
    config, orch = object(), MagicMock()
    a, b = Args(), Args()
    capture.annotate(a, [("context", 0, contexts[0], None)])
    capture.annotate(b, [("context", 0, contexts[0 if misuse == "wrong-rank" else 1], None)])
    first = capture.submit(2, a, config, 0, "exchange")
    with pytest.raises(ValueError):
        if misuse != "missing-rank":
            capture.submit(99 if misuse == "different-callable" else 2, b, config, 1, "exchange")
        if misuse == "forward-dependency":
            capture.add_dependency(b, first)
        capture.flush(orch)
    orch.submit_next_level_group.assert_not_called()


def test_cross_graph_dependency_and_invalid_contract_are_rejected(graph):
    capture, _ = graph
    other = _RecoveryGraph(OperatorRecoveryConfig(("exchange",)), 2, {"exchange": 2})
    with pytest.raises(ValueError, match="same captured graph"):
        other.add_dependency(Args(), capture.submit(2, Args(), object(), 0, "exchange"))
    with pytest.raises(ValueError, match="not registered"):
        _RecoveryGraph(OperatorRecoveryConfig(("typo",)), 2, {"exchange": 2})
    for kwargs in ({"max_retries": -1}, {"timeout_s": 0}, {"backoff_s": float("nan")}):
        with pytest.raises(ValueError):
            OperatorRecoveryConfig(("exchange",), **kwargs)
    assert _get_recovery(MagicMock()) is None


def test_generated_program_native_argument_contract(tmp_path):
    """Compile the actual hardware program, then run its generated host code on CPU.

    No device execution occurs. Real native TaskArgs validate scalar/tensor binding
    indices and full-rank metadata; CPU mailbox endpoints complete the generated native groups.
    """
    api = pytest.importorskip("simpler.task_interface")
    import ctypes  # noqa: PLC0415
    import threading  # noqa: PLC0415

    import simpler.worker as worker_module  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from _task_interface import _mailbox_load_i32, _mailbox_store_i32  # noqa: PLC0415
    from pypto.ir import DistributedConfig  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415
    from pypto.runtime.distributed_runner import (  # noqa: PLC0415
        _call_alloc_intermediates,
        _load_orch_entry,
        _make_dispatch_orchestration,
    )
    from simpler.buffer import mint_owner_instance_id, wrap_vmm_window  # noqa: PLC0415
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    test_file = Path(__file__).resolve().parents[2] / "st/distributed/test_l3_operator_recovery.py"
    spec = importlib.util.spec_from_file_location("_recovery_program_contract", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = RunConfig(
        platform="a5",
        save_kernels=True,
        save_kernels_dir=str(tmp_path),
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0, aicpu_thread_num=2),
    )
    compiled = module.recovery_program.compile(config=config)
    entry, alloc = _load_orch_entry(compiled._output_dir)
    tensors = {
        "x": torch.zeros((2, 1, module.N)),
        "y": torch.zeros((2, 1, module.N)),
        "stats": torch.zeros((2, 8), dtype=torch.int32),
        "mode": 1,
    }
    tensors = {info.name: tensors[info.name.split("__ssa_")[0]] for info in compiled._get_metadata()[0]}
    _call_alloc_intermediates(alloc, tensors, 2)
    for value in tensors.values():
        if isinstance(value, torch.Tensor):
            value.share_memory_()
    owner = mint_owner_instance_id()
    captured = []
    native = api._Worker(3, 1 << 20)
    mailboxes = [ctypes.create_string_buffer(api.MAILBOX_SIZE) for _ in range(2)]
    stopping = threading.Event()
    completed = [0, 0]

    def complete_on_cpu(rank, address):
        task = address + api.MAILBOX_FRAME_SIZE
        while not stopping.wait(0.0001):
            if _mailbox_load_i32(task) == worker_module._TASK_READY:
                _mailbox_store_i32(task + worker_module._OFF_ACCEPTED, 1)
                ctypes.c_int32.from_address(task + worker_module._OFF_ERROR).value = 0
                completed[rank] += 1
                _mailbox_store_i32(task, worker_module._TASK_DONE)

    threads = []
    for rank, memory in enumerate(mailboxes):
        address = ctypes.addressof(memory)
        native.add_next_level_worker_at(rank, address)
        thread = threading.Thread(target=complete_on_cpu, args=(rank, address))
        thread.start()
        threads.append(thread)
    native.init()
    native_orch = native.get_orchestrator()
    run = native_orch._begin_run()

    class Host(Orchestrator):
        def __init__(self):
            super().__init__(
                native_orch, Worker(level=3, platform="a5", device_ids=[0, 1], num_sub_workers=0)
            )

        @contextmanager
        def allocate_domain(self, **kwargs):
            contexts = []
            for rank in range(2):
                binding = api.RecoveryDomainBinding()
                binding.allocation_id, binding.generation = 0, 1
                binding.local_window_base = 0x100000 + rank * 0x100000
                binding.window_size = kwargs["window_size"]
                binding.device_ctx, binding.workspace = binding.local_window_base + 0x10000, 0x900000
                offset, buffers = 0, {}
                for index, slot in enumerate(kwargs["buffers"]):
                    buffers[slot.name] = wrap_vmm_window(
                        binding.local_window_base + offset,
                        slot.nbytes,
                        owner,
                        rank * 100 + index,
                        owner_worker_id=rank,
                    )
                    offset += slot.nbytes
                contexts.append(
                    SimpleNamespace(
                        domain_rank=rank,
                        domain_size=2,
                        recovery_binding=binding,
                        device_ctx=binding.device_ctx,
                        buffers=buffers,
                    )
                )
            yield contexts

        def submit_next_level_group(self, callable_id, args_list, cfg, *, workers):
            captured.append((callable_id, args_list))
            # CPU mailboxes complete these native groups without executing kernels.
            return native_orch.submit_next_level_group(
                bytes([callable_id]) * 32,
                "CHIP_CALLABLE",
                "LOCAL_CHIP",
                args_list,
                cfg,
                workers,
            )

    host, keep = Host(), []
    callback = _make_dispatch_orchestration(
        entry,
        tensors,
        {"warmup": 1, "exchange": 2, "finish": 3},
        {},
        api.CallConfig(),
        2,
        keep,
        OperatorRecoveryConfig(("exchange",)),
    )
    try:
        callback(host, None, None)
        assert completed == [0, 0], "native work escaped graph-build admission"
        with pytest.raises(RuntimeError, match="precede"):
            host.defer_dispatch_until_submitted()
        assert host._pypto_recovery is None
        assert [cid for cid, _ in captured] == [1, 2, 3]
        for sequence, (_, members) in enumerate(captured):
            for rank, args in enumerate(members):
                assert args.recovery_metadata.rank == rank
                assert args.recovery_metadata.op_seq == sequence + 1
                assert args.recovery_metadata.replay_safety == (
                    api.ReplaySafety.REPLAY_SAFE if sequence == 1 else api.ReplaySafety.UNKNOWN
                )
                for binding in args.domain_bindings:
                    assert binding.domain.device_ctx == 0x110000 + rank * 0x100000
                    for argument in binding.arguments:
                        if argument.kind == api.RecoveryArgKind.COMM_CONTEXT:
                            assert args.scalar(argument.index) == binding.domain.device_ctx
                        else:
                            assert argument.index < args.tensor_count()
        print("[PYPTO_RECOVERY_TEST] GENERATED_NATIVE_GRAPH_PASS")
    finally:
        native_orch._close_run_submission(run)
        try:
            assert native_orch._wait_run_for(run, 3), "CPU native graph did not drain"
            assert completed == [3, 3]
        finally:
            native.close()
            stopping.set()
            for thread in threads:
                thread.join(3)
                assert not thread.is_alive()
            host._worker.close()
