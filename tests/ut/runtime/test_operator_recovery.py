# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Rank publication and concrete alias checks for compiler-derived contracts."""

import json
import logging
import runpy
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.pypto_core import passes
from pypto.runtime.operator_recovery import OperatorRecoveryConfig, _bind_recovery_domain, _RecoveryGraph


class Args:
    def __init__(self):
        self.dependencies = []
        self.bindings = []

    def tensor_count(self):
        return 2

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
    proof = {"safe": True, "complete": True, "tensor_effects": [1, 2], "reasons": []}
    capture = _RecoveryGraph(2, {"task": proof}, MagicMock())
    contexts = [
        SimpleNamespace(
            domain_rank=rank,
            domain_size=2,
            recovery_binding=SimpleNamespace(
                allocation_id=0, generation=1, local_window_base=10000 + rank * 1000
            ),
        )
        for rank in range(2)
    ]
    capture.bind_domain(contexts)
    return capture, contexts


def add(capture, contexts, rank, config, *, alias=False, output=None):
    args = Args()
    source = torch.zeros(4)
    output = source if alias else torch.empty(4) if output is None else output
    capture.annotate(args, [("tensor", 0, None, source), ("tensor", 1, None, output)])
    handle = capture.submit(1, args, config, rank, "task")
    return args, handle


def test_compiler_proof_requires_concrete_non_aliasing_arguments(graph):
    capture, contexts = graph
    config, orch = object(), capture.orch
    a, _ = add(capture, contexts, 0, config)
    b, _ = add(capture, contexts, 1, config)
    capture.finish()
    orch.submit_next_level_group.assert_called_once()
    assert a.metadata.replay_safety == b.metadata.replay_safety == "safe"
    assert a.metadata.participants == [0, 1]
    assert a.bindings == b.bindings == []


def test_actual_alias_rejects_an_otherwise_safe_function(graph):
    capture, contexts = graph
    config = object()
    a, _ = add(capture, contexts, 0, config, alias=True)
    b, _ = add(capture, contexts, 1, config)
    capture.finish()
    assert a.metadata.replay_safety == b.metadata.replay_safety == "unknown"


def test_overlapping_rank_outputs_are_not_approved(graph):
    capture, contexts = graph
    config, output = object(), torch.empty(4)
    a, _ = add(capture, contexts, 0, config, output=output)
    b, _ = add(capture, contexts, 1, config, output=output)
    capture.finish()
    assert a.metadata.replay_safety == b.metadata.replay_safety == "unknown"


def test_incomplete_group_is_never_dispatched(graph):
    capture, contexts = graph
    orch = capture.orch
    add(capture, contexts, 0, object())
    with pytest.raises(ValueError, match="full-rank"):
        capture.finish()
    orch.submit_next_level_group.assert_not_called()


def test_manual_safety_override_is_removed():
    with pytest.raises(TypeError):
        OperatorRecoveryConfig(replay_safe=("task",))
    for options in ({"max_retries": -1}, {"timeout_s": 0}, {"backoff_s": float("nan")}):
        with pytest.raises(ValueError):
            OperatorRecoveryConfig(**options)


@pytest.mark.parametrize("with_binding", [False, True])
def test_generated_urma_domain_requirement_checks_every_rank_without_recovery(with_binding):
    """Use real domain/native binding types, including when recovery is disabled."""
    api = pytest.importorskip("simpler.task_interface")

    contexts = {}
    for rank in range(2):
        context = api.ChipDomainContext(
            name="urma",
            domain_rank=rank,
            domain_size=2,
            device_ctx=100 + rank,
            local_window_base=1000 + rank * 1000,
            actual_window_size=64,
            buffers={},
            recovery_binding=api.RecoveryDomainBinding() if with_binding else None,
        )
        context.workspace_kind = api.DomainRecoveryEngine.URMA.value
        contexts[rank] = context
    domain = api.CommDomainHandle(
        name="urma",
        workers=(0, 1),
        contexts=contexts,
        allocation_id=0,
        _release_fn=lambda _: None,
    )
    _bind_recovery_domain(SimpleNamespace(), domain, required_engine="URMA")
    contexts[1].workspace_kind = api.DomainRecoveryEngine.SDMA.value
    with pytest.raises(ValueError, match="requires URMA, rank 1"):
        _bind_recovery_domain(SimpleNamespace(), domain, required_engine="URMA")
    contexts[1].workspace_kind = api.DomainRecoveryEngine.NONE.value
    with pytest.raises(ValueError, match="requires URMA, rank 1"):
        _bind_recovery_domain(SimpleNamespace(), domain, required_engine="URMA")
    _bind_recovery_domain(SimpleNamespace(), domain)


def test_urma_program_codegen_keeps_replay_proof_and_actual_get(tmp_path, monkeypatch):
    """Run the full compiler pipeline for the same JIT program exercised on A5."""
    source = Path(__file__).resolve().parents[2] / "st/distributed/test_l3_operator_recovery.py"
    definitions = runpy.run_path(str(source))
    definitions["instrument_wrappers"](monkeypatch, backend="urma")
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        compiled = ir.compile(
            definitions["urma_program"].specialize(),
            platform="a5",
            skip_ptoas=True,
            output_dir=str(tmp_path / "program"),
            distributed_config=ir.DistributedConfig(device_ids=[0, 1]),
        )
    meta = json.loads((compiled.output_dir / "distributed_meta.json").read_text())
    assert meta["recovery_effects"]["urma_exchange"]["safe"], meta["recovery_effects"]
    code = (compiled.output_dir / "orchestration/host_orch.py").read_text()
    assert 'required_engine="URMA"' in code
    mlir = "\n".join(path.read_text() for path in compiled.output_dir.rglob("*.pto"))
    assert "func.call @pypto_urma_get_completion" in mlir
    assert "pto.comm.tget(" not in mlir
    orchestration = (
        compiled.output_dir / "next_levels/urma_exchange/orchestration/urma_exchange.cpp"
    ).read_text()
    # The notification injection and payload share a single-producer URMA queue.
    # A TaskId dependency must gate payload admission on full notification completion.
    assert "TaskOutputTensors urma_protocol_ready = rt_submit_aiv_task(1, params_t1);" in orchestration
    assert "const TaskId urma_protocol_deps[] = {urma_protocol_ready.task_id()};" in orchestration
    assert "params_t2.set_dependencies(urma_protocol_deps, 1);" in orchestration
    assert orchestration.index("params_t2.set_dependencies(") < orchestration.index("rt_submit_aiv_task(2,")
    print("[PYPTO_RECOVERY_TEST] URMA_PROTOCOL_ORDER_PASS", flush=True)
    print("[PYPTO_RECOVERY_TEST] URMA_GENERATED_GRAPH_PASS", flush=True)


@pytest.mark.parametrize("misuse", ["different-callable", "wrong-rank", "forward-dependency"])
def test_group_validation_precedes_publication(graph, misuse):
    capture, contexts = graph
    config = object()
    a, handle = add(capture, contexts, 0, config)
    b = Args()
    entries = [("tensor", 0, None, torch.zeros(4)), ("tensor", 1, None, torch.empty(4))]
    if misuse == "wrong-rank":
        entries.append(("context", 0, contexts[0], None))
    elif misuse == "forward-dependency":
        capture.add_dependency(b, handle)
    capture.annotate(b, entries)
    with pytest.raises(ValueError):
        capture.submit(99 if misuse == "different-callable" else 1, b, config, 1, "task")
    capture.orch.submit_next_level_group.assert_not_called()
    with pytest.raises(ValueError, match="already submitted"):
        capture.add_dependency(a, handle)


@pytest.mark.parametrize("case", ["independent", "raw", "war", "waw", "disjoint"])
def test_only_actual_memory_hazards_order_groups(graph, case):
    graph, _contexts = graph
    first, second, config = object(), object(), object()
    graph.orch.submit_next_level_group.side_effect = [first, second]
    inputs = [torch.zeros(4), torch.zeros(4)]
    storage = [torch.empty(8), torch.empty(8)]
    outputs = [value[:4] for value in storage]
    for rank in range(2):
        args = Args()
        graph.annotate(args, [("tensor", 0, None, inputs[rank]), ("tensor", 1, None, outputs[rank])])
        graph.submit(1, args, config, rank, "task")
    members = []
    for rank in range(2):
        source = outputs[rank] if case == "raw" else inputs[rank] if case == "independent" else torch.zeros(4)
        output = (
            inputs[rank]
            if case == "war"
            else outputs[rank]
            if case == "waw"
            else storage[rank][4:]
            if case == "disjoint"
            else torch.empty(4)
        )
        args = Args()
        graph.annotate(args, [("tensor", 0, None, source), ("tensor", 1, None, output)])
        graph.submit(1, args, config, rank, "task")
        members.append(args)
    graph.finish()
    expected = [first] if case in ("raw", "war", "waw") else []
    assert all(args.dependencies == expected for args in members)
    assert all(not args.metadata.uses_communication for args in members)
    print(f"[PYPTO_RECOVERY_TEST] HAZARD_FRONTIER_PASS case={case}", flush=True)


@pytest.mark.parametrize("reason", ["opaque", "extra-output"])
def test_opaque_effects_and_unaccounted_outputs_remain_barriers(graph, reason):
    graph, contexts = graph
    config = object()
    handles = [object() for _ in range(3)]
    graph.orch.submit_next_level_group.side_effect = handles
    for rank in range(2):
        add(graph, contexts, rank, config)
    opaque = []
    for rank in range(2):
        args = Args()
        if reason == "extra-output":
            args.tensor_count = lambda: 3
        graph.proofs["opaque"] = {"safe": False, "complete": False}
        graph.annotate(args, [("tensor", 0, None, torch.zeros(4)), ("tensor", 1, None, torch.empty(4))])
        graph.submit(1, args, config, rank, "opaque" if reason == "opaque" else "task")
        opaque.append(args)
    assert all(
        args.metadata.replay_safety == "unknown" and args.dependencies == [handles[0]] for args in opaque
    )
    successors = [add(graph, contexts, rank, config)[0] for rank in range(2)]
    assert all(args.dependencies == [handles[1]] for args in successors)


def test_explicit_dependency_survives_without_a_memory_hazard(graph):
    graph, contexts = graph
    config, first = object(), object()
    graph.orch.submit_next_level_group.side_effect = [first, object()]
    _, producer = add(graph, contexts, 0, config)
    add(graph, contexts, 1, config)
    consumers = []
    for rank in range(2):
        args = Args()
        graph.annotate(args, [("tensor", 0, None, torch.zeros(4)), ("tensor", 1, None, torch.empty(4))])
        graph.add_dependency(args, producer)
        graph.submit(1, args, config, rank, "task")
        consumers.append(args)
    assert all(args.dependencies == [first] for args in consumers)


@pytest.mark.parametrize("transport", ["sdma", "urma"])
def test_generated_program_native_argument_contract(tmp_path, transport):
    """The real generated host graph reaches native mailboxes before graph closure."""
    api = pytest.importorskip("simpler.task_interface")
    import ctypes  # noqa: PLC0415
    import runpy  # noqa: PLC0415
    import threading  # noqa: PLC0415
    from contextlib import contextmanager  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import simpler.worker as worker_module  # noqa: PLC0415
    from _task_interface import _mailbox_load_i32, _mailbox_store_i32  # noqa: PLC0415
    from pypto import ir  # noqa: PLC0415
    from pypto.pypto_core import passes  # noqa: PLC0415
    from pypto.runtime.distributed_runner import (  # noqa: PLC0415
        _call_alloc_intermediates,
        _load_orch_entry,
        _make_dispatch_orchestration,
    )
    from simpler.buffer import mint_owner_instance_id, wrap_vmm_window  # noqa: PLC0415
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    source = Path(__file__).resolve().parents[2] / "st/distributed/test_l3_operator_recovery.py"
    definitions = runpy.run_path(str(source))
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        compiled = ir.compile(
            definitions["urma_program"].specialize()
            if transport == "urma"
            else definitions["build_program"](),
            platform="a5",
            skip_ptoas=True,
            output_dir=str(tmp_path / "program"),
            distributed_config=ir.DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
        )
    import json  # noqa: PLC0415

    proofs = json.loads((compiled.output_dir / "distributed_meta.json").read_text())["recovery_effects"]
    entry, alloc = _load_orch_entry(compiled.output_dir)
    values = {
        "x": torch.zeros((2, 1, definitions["N"])),
        "y": torch.zeros((2, 1, definitions["N"])),
        "stats": torch.zeros((2, 1, 8), dtype=torch.int32),
        "mode": 0,
    }
    tensors = {info.name: values[info.name.split("__ssa_")[0]] for info in compiled._get_metadata()[0]}
    _call_alloc_intermediates(alloc, tensors, 2)
    for value in tensors.values():
        if isinstance(value, torch.Tensor):
            value.share_memory_()
    owner = mint_owner_instance_id()
    captured = []
    native = api._Worker(3, 1 << 20)
    mailboxes = [ctypes.create_string_buffer(api.MAILBOX_SIZE) for _ in range(2)]
    stopping = threading.Event()
    first_done = [threading.Event(), threading.Event()]
    completed = [0, 0]

    def complete_on_cpu(rank, address):
        task = address + api.MAILBOX_FRAME_SIZE
        while not stopping.wait(0.0001):
            if _mailbox_load_i32(task) == worker_module._TASK_READY:
                _mailbox_store_i32(task + worker_module._OFF_ACCEPTED, 1)
                ctypes.c_int32.from_address(task + worker_module._OFF_ERROR).value = 0
                completed[rank] += 1
                _mailbox_store_i32(task, worker_module._TASK_DONE)
                first_done[rank].set()

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
            contexts = {}
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
                contexts[rank] = api.ChipDomainContext(
                    name=kwargs["name"],
                    domain_rank=rank,
                    domain_size=2,
                    recovery_binding=binding,
                    device_ctx=binding.device_ctx,
                    local_window_base=binding.local_window_base,
                    actual_window_size=binding.window_size,
                    workspace_kind=(
                        api.DomainRecoveryEngine.URMA
                        if transport == "urma"
                        else api.DomainRecoveryEngine.SDMA
                    ).value,
                    buffers=buffers,
                )
            with api.CommDomainHandle(
                name=kwargs["name"],
                workers=tuple(kwargs["workers"]),
                contexts=contexts,
                allocation_id=0,
                _release_fn=lambda _: None,
            ) as domain:
                yield domain

        def submit_next_level_group(self, callable_id, args_list, cfg, *, workers):
            captured.append((callable_id, args_list))
            handle = native_orch.submit_next_level_group(
                bytes([callable_id]) * 32,
                "CHIP_CALLABLE",
                "LOCAL_CHIP",
                args_list,
                cfg,
                workers,
            )
            if len(captured) == 1:
                assert all(event.wait(3) for event in first_done), "first group waited for graph closure"
            return handle

    host, keep = Host(), []
    callback = _make_dispatch_orchestration(
        entry,
        tensors,
        {
            ("urma_" if transport == "urma" else "") + name: index
            for index, name in enumerate(("warmup", "exchange", "finish"), 1)
        },
        {},
        api.CallConfig(),
        2,
        keep,
        OperatorRecoveryConfig(),
        proofs,
    )
    try:
        callback(host, None, None)
        assert all(event.is_set() for event in first_done)
        assert not native_orch._wait_run_for(run, 0.001), "an open graph retired early"
        with pytest.raises(RuntimeError, match="precede"):
            host.enable_incremental_recovery()
        assert host._pypto_recovery is None
        assert [cid for cid, _ in captured] == [1, 2, 3]
        for sequence, (_, members) in enumerate(captured):
            for rank, args in enumerate(members):
                assert args.recovery_metadata.rank == rank
                assert args.recovery_metadata.op_seq == sequence + 1
                assert args.recovery_metadata.replay_safety == api.ReplaySafety.REPLAY_SAFE
                for binding in args.domain_bindings:
                    assert binding.domain.device_ctx == 0x110000 + rank * 0x100000
                    for argument in binding.arguments:
                        if argument.kind == api.RecoveryArgKind.COMM_CONTEXT:
                            assert args.scalar(argument.index) == binding.domain.device_ctx
                        else:
                            assert argument.index < args.tensor_count()
        print(f"[PYPTO_RECOVERY_TEST] INCREMENTAL_NATIVE_GRAPH_PASS backend={transport}", flush=True)
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


@pytest.mark.parametrize("backend", ["sdma", "urma"])
def test_dfx_hardware_setup_compiles_before_device_prepare(tmp_path, monkeypatch, caplog, backend):
    """Compile both DFX backends through their actual setup before device preparation."""
    source = Path(__file__).resolve().parents[2] / "st/distributed/test_l3_operator_recovery.py"
    definitions = runpy.run_path(str(source))
    run_test = definitions["test_recovery_dfx_labels"]
    monkeypatch.setitem(run_test.__globals__, "_urma_enabled", lambda: backend == "urma")
    # Conversion is beyond the device boundary; keep this CPU test runtime-optional.
    setup = ModuleType("simpler_setup")
    tools = ModuleType("simpler_setup.tools")
    converter = ModuleType("simpler_setup.tools.swimlane_converter")
    converter.load_kernel_config = MagicMock(side_effect=AssertionError("device boundary crossed"))
    setup.tools, tools.swimlane_converter = tools, converter
    for module in (setup, tools, converter):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    compile_module = import_module("pypto.ir.compile")
    original_compile = compile_module.compile

    def compile_on_cpu(*args, **kwargs):
        kwargs["skip_ptoas"] = True
        # Match the existing distributed codegen tests: verify every pass;
        # printer roundtripping of window names is a separate compiler check.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            return original_compile(*args, **kwargs)

    monkeypatch.setattr(compile_module, "compile", compile_on_cpu)
    monkeypatch.setattr(ir, "compile", compile_on_cpu)
    prepared = []

    class ReachedDevicePrepare(Exception):
        pass

    def stop_before_device(compiled, **options):
        if backend == "urma":
            assert logging.getLogger("simpler").getEffectiveLevel() == logging.INFO
        assert options["operator_recovery"] == OperatorRecoveryConfig()
        assert compiled.output_dir == tmp_path / "program"
        metadata = json.loads((compiled.output_dir / "distributed_meta.json").read_text())
        name = "urma_exchange" if backend == "urma" else "exchange"
        assert metadata["recovery_effects"][name]["safe"]
        assert (compiled.output_dir / "orchestration/host_orch.py").is_file()
        prepared.append(compiled)
        raise ReachedDevicePrepare

    monkeypatch.setattr(DistributedCompiledProgram, "prepare", stop_before_device)
    with pytest.raises(ReachedDevicePrepare):
        run_test(SimpleNamespace(platform="a5"), [0, 1], "healthy", None, caplog, monkeypatch, tmp_path)
    assert len(prepared) == 1
    converter.load_kernel_config.assert_not_called()


def test_dfx_hardware_validation_reads_child_log_files(tmp_path, monkeypatch, capfd, caplog):
    """Exercise the hardware validator across its console-to-file log transition."""
    source = Path(__file__).resolve().parents[2] / "st/distributed/test_l3_operator_recovery.py"
    run_test = runpy.run_path(str(source))["test_recovery_dfx_labels"]
    monkeypatch.setitem(run_test.__globals__, "_urma_enabled", lambda: True)
    monkeypatch.setitem(run_test.__globals__, "instrument_wrappers", lambda *args, **kwargs: None)
    compiled = MagicMock(output_dir=tmp_path / "program")
    monkeypatch.setitem(
        run_test.__globals__, "urma_program", SimpleNamespace(compile=lambda *args, **kwargs: compiled)
    )
    runtime = compiled.prepare.return_value.__enter__.return_value
    runtime._states = {compiled: {"chip_cids": {"task": SimpleNamespace(digest=b"task")}}}
    dfx = compiled.output_dir / "dfx_outputs"
    capture = dfx / "rank0/d0"
    capture.mkdir(parents=True)
    (capture / "chip_swimlane_records.json").write_text("{}")
    (capture / "dispatch_identity.json").write_text(json.dumps({"callable_digest": b"task".hex()}))
    (capture / "name_map.json").write_text("{}")
    missing_destroy = False

    def execute(x, y, stats, fault, *, config):
        y.copy_(x.flip(0) + 4)
        # Initial creation precedes the first dispatch; subsequent native records
        # have exactly one file destination, rather than also reaching stderr.
        for rank in range(2):
            records = []
            for allocation in range(2):
                domain = f"0x{16 + 2 * rank + allocation:x}"
                created = (
                    f"URMA_DOMAIN_OWNER_CREATE_RESULT rank={rank} allocation_id={allocation} "
                    f"base_hccl=0x1 domain_hccl={domain} rc=0\n"
                )
                if allocation == 0:
                    print(created, end="", file=sys.stderr)
                else:
                    records.append(created)
                if not (missing_destroy and rank == 1 and allocation == 1):
                    records.append(
                        f"URMA_DOMAIN_OWNER_DESTROY_RESULT rank={rank} allocation_id={allocation} "
                        f"domain_hccl={domain} rc=0\n"
                    )
            (dfx / f"host.{rank}.log").write_text("".join(records))
        (dfx / "host.2.log").write_text("[PTO_RECOVERY] REPLAY_DISPATCH \n[PTO_RECOVERY] REPLAY_COMMIT \n")

    runtime.side_effect = execute
    setup, tools = ModuleType("simpler_setup"), ModuleType("simpler_setup.tools")
    converter = ModuleType("simpler_setup.tools.swimlane_converter")

    class ReachedCaptureValidation(Exception):
        pass

    converter.load_kernel_config = MagicMock(side_effect=ReachedCaptureValidation)
    setup.tools, tools.swimlane_converter = tools, converter
    for module in (setup, tools, converter):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ReachedCaptureValidation):
        run_test(SimpleNamespace(platform="a5"), [0, 1], "transient", capfd, caplog, monkeypatch, tmp_path)
    missing_destroy = True
    with pytest.raises(AssertionError) as failure:
        run_test(SimpleNamespace(platform="a5"), [0, 1], "transient", capfd, caplog, monkeypatch, tmp_path)
    created, destroyed = failure.value.args[0]
    assert len(created) == 4 and len(destroyed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
