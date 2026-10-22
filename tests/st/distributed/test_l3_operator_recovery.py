# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Generated SDMA tasks, compiler-owned replay proof and automatic continuation.

Only test-generated wrappers carry injection/counters. Business operations are
compiled normally; no external source is granted a semantic safety override.
"""

import json
import logging
import os
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.runtime import OperatorRecoveryConfig, RunConfig

N = 256


def _urma_enabled():
    return os.environ.get("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "").upper() in {"1", "ON", "TRUE", "YES"}


def build_program():
    @pl.program
    class RecoveryProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def warm_kernel(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            source: pl.Out[pld.DistributedTensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> tuple[pld.DistributedTensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            source = pl.store(pl.add(pl.load(x, [0, 0], [1, N]), 1.0), [0, 0], source)
            return source, stats

        @pl.function(type=pl.FunctionType.InCore)
        def entry_kernel(
            self,
            source: pld.DistributedTensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            out = pl.store(pl.load(source, [0, 0], [1, N]), [0, 0], out)
            return out, stats

        @pl.function(type=pl.FunctionType.InCore)
        def get_kernel(
            self,
            source: pld.DistributedTensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            pld.tensor.get(out, peer=peer, src=source)
            return out, stats

        @pl.function(type=pl.FunctionType.InCore)
        def finish_kernel(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            out = pl.store(pl.add(pl.load(x, [0, 0], [1, N]), 3.0), [0, 0], out)
            return out, stats

        @pl.function(type=pl.FunctionType.Orchestration)
        def warmup(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            source: pl.Out[pld.DistributedTensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 8], pl.INT32]:
            source, stats = self.warm_kernel(x, source, flags, stats, mode)
            return stats

        @pl.function(type=pl.FunctionType.Orchestration)
        def exchange(
            self,
            source: pld.DistributedTensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            out, stats = self.entry_kernel(source, out, flags, stats, mode)
            out, stats = self.get_kernel(source, out, flags, stats, mode, peer)
            return out, stats

        @pl.function(type=pl.FunctionType.Orchestration)
        def finish(
            self,
            x: pl.Tensor[[1, N], pl.FP32],
            out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            flags: pld.DistributedTensor[[1024], pl.UINT8],
            stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
            return self.finish_kernel(x, out, flags, stats, mode)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(
            self,
            x: pl.Tensor[[2, 1, N], pl.FP32],
            y: pl.Out[pl.Tensor[[2, 1, N], pl.FP32]],
            stats: pl.InOut[pl.Tensor[[2, 1, 8], pl.INT32]],
            mode: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[2, 1, N], pl.FP32]:
            data_buf = pld.alloc_window_buffer([1, N], dtype=pl.FP32)
            flags_buf = pld.alloc_window_buffer([1024], dtype=pl.UINT8)
            received = pl.create_tensor([2, 1, N], dtype=pl.FP32)
            for rank in pl.range(pld.world_size()):
                source = pld.window(data_buf, [1, N], dtype=pl.FP32)
                flags = pld.window(flags_buf, [1024], dtype=pl.UINT8)
                local_stats = stats[rank]
                local_stats = self.warmup(x[rank], source, flags, local_stats, mode, device=rank)
                source = pld.window(data_buf, [1, N], dtype=pl.FP32)
                recv, local_stats = self.exchange(
                    source, received[rank], flags, local_stats, mode, 1 - rank, device=rank
                )
                out, local_stats = self.finish(recv, y[rank], flags, local_stats, mode, device=rank)
            return y

    return RecoveryProgram


def instrument_wrappers(monkeypatch, backend="sdma"):
    """Inject before the generated body, without a production flag or hook."""
    from pypto.backend import pto_backend  # noqa: PLC0415

    original = pto_backend._generate_kernel_wrapper
    phases = {"warm_kernel": 2, "entry_kernel": 0, "get_kernel": 1, "finish_kernel": 3}
    header = Path(__file__).with_name("kernels") / "recovery_injection.h"
    if backend == "urma":
        phases = {f"urma_{name}": phase for name, phase in phases.items()}
        header = header.with_name("urma_recovery_injection.h")

    def generate(function, *args, **kwargs):
        code = original(function, *args, **kwargs)
        matches = [phase for name, phase in phases.items() if function.name.startswith(name)]
        if not matches:
            return code
        tensors = [p for p in function.params if isinstance(p.type, ir.TensorType)]
        scalars = [p for p in function.params if isinstance(p.type, (ir.ScalarType, ir.CommCtxType))]
        indices = {p.name_hint.split("__ssa_")[0]: i for i, p in enumerate(tensors + scalars)}
        assert {"flags", "stats", "mode"} <= indices.keys(), indices
        windows = [p for p in tensors if isinstance(p.type, ir.DistributedTensorType)]
        contexts = [p for p in scalars if isinstance(p.type, ir.CommCtxType)]
        assert len(windows) == len(contexts), function.name
        flag_position = next(i for i, p in enumerate(windows) if p.name_hint.startswith("flags"))
        context_index = next(
            i for i, p in enumerate(tensors + scalars) if p.unique_id == contexts[flag_position].unique_id
        )
        call = (
            "    if (inject_recovery_fault(args, "
            f"{indices['flags']}, {indices['stats']}, {context_index}, {indices['mode']}, "
            f"{matches[0]})) return;\n"
        )
        marker = "    // Forward to ptoas-generated function"
        assert code.count(marker) == 1
        return f'#include "{header.as_posix()}"\n' + code.replace(marker, call + marker)

    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", generate)
    if backend == "urma":
        original_orchestration = pto_backend._codegen_core.generate_orchestration

        def generate_orchestration(program, function):
            result = original_orchestration(program, function)
            if function.name != "urma_exchange":
                return result
            # The injected protocol adds a second producer of the payload's
            # per-peer URMA queue. Tensor analysis cannot see injected C++:
            # add its matching completion dependency at the same test seam.
            code = result.code
            submissions = {}
            for name in ("urma_get_kernel", "urma_payload"):
                kernel_id = result.func_name_to_id[name]
                matches = list(
                    re.finditer(rf"(?m)^([ \t]*)rt_submit_aiv_task\({kernel_id}, (params_t\d+)\);$", code)
                )
                assert len(matches) == 1, (name, code)
                submissions[name] = matches[0]
            ready = submissions["urma_get_kernel"]
            payload = submissions["urma_payload"]
            assert ready.start() < payload.start(), code
            indent, arguments = payload.groups()
            code = code.replace(
                payload[0],
                f"{indent}const TaskId urma_protocol_deps[] = {{urma_protocol_ready.task_id()}};\n"
                f"{indent}{arguments}.set_dependencies(urma_protocol_deps, 1);\n{payload[0]}",
                1,
            )
            code = code.replace(
                ready[0],
                ready[0].replace(
                    "rt_submit_aiv_task", "TaskOutputTensors urma_protocol_ready = rt_submit_aiv_task"
                ),
                1,
            )
            return SimpleNamespace(
                code=code,
                func_name_to_id=result.func_name_to_id,
                func_name_to_core_type=result.func_name_to_core_type,
                func_name_to_signature=result.func_name_to_signature,
                orchestration_signature=result.orchestration_signature,
            )

        monkeypatch.setattr(pto_backend._codegen_core, "generate_orchestration", generate_orchestration)


def gate_graph_builder(monkeypatch):
    """Delay a successor until a fault is visible; the framework owns all recovery."""
    from pypto.runtime.operator_recovery import _RecoveryGraph  # noqa: PLC0415

    original, observed = _RecoveryGraph.submit, []

    def submit(graph, callable_id, args, config, rank, name):
        if name in ("finish", "urma_finish") and rank == graph.ranks - 1:
            deadline = time.monotonic() + 30
            while not (episodes := graph.orch._worker.pending_recoveries()):
                assert time.monotonic() < deadline, "fault was not observed while the graph was open"
                time.sleep(0.001)
            observed.append(episodes[0].ticket.recovery_id)
            print(f"[PYPTO_RECOVERY_TEST] OPEN_GRAPH_RECOVERY_SEEN recovery_id={observed[-1]}", flush=True)
        return original(graph, callable_id, args, config, rank, name)

    monkeypatch.setattr(_RecoveryGraph, "submit", submit)
    return observed


@pytest.mark.timeout(240)
@pytest.mark.skipif(_urma_enabled(), reason="SDMA recovery requires the default SDMA runtime build")
@pytest.mark.parametrize("mode", ["baseline", "healthy", "transient"])
def test_program_continues(test_config, device_ids, mode, capfd, monkeypatch, tmp_path):
    if test_config.platform != "a5" or len(device_ids) != 2:
        pytest.skip("operator recovery requires exactly two local A5 devices")
    instrument_wrappers(monkeypatch)
    compiled = ir.compile(
        build_program(),
        platform="a5",
        output_dir=str(tmp_path / "program"),
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0, aicpu_thread_num=2),
    )
    proofs = json.loads((compiled.output_dir / "distributed_meta.json").read_text())["recovery_effects"]
    assert proofs["exchange"]["safe"], proofs["exchange"]
    x = torch.arange(2 * N, dtype=torch.float32).reshape(2, 1, N).share_memory_()
    y = torch.zeros_like(x).share_memory_()
    stats = torch.zeros((2, 1, 8), dtype=torch.int32).share_memory_()
    fault = int(mode == "transient")
    observed = gate_graph_builder(monkeypatch) if fault else []
    expected = torch.tensor([1, 1 + fault, 1, 1, 1, 1, 0, 0], dtype=torch.int32).repeat(2, 1)
    durations = []
    recovery = None if mode == "baseline" else OperatorRecoveryConfig()
    with compiled.prepare(operator_recovery=recovery) as runtime:
        for _ in range(1 if fault else 4):
            y.zero_()
            stats.zero_()
            started = time.monotonic_ns()
            runtime(x, y, stats, fault)
            durations.append(time.monotonic_ns() - started)
            torch.testing.assert_close(y, x.flip(0) + 4, rtol=0, atol=0)
            assert torch.equal(stats.reshape(2, 8), expected), stats.tolist()
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    with capfd.disabled():
        print(logs, end="", flush=True)
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_DISPATCH ", logs)) == fault
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_COMMIT ", logs)) == fault
    assert len(re.findall(r"STOP_DELIVERED[^\n]* active=1", logs)) == fault
    if fault:
        assert len(observed) == 1
        bindings = re.findall(r"SUBMISSION_BINDING_CURRENT[^\n]*worker=(\d+)[^\n]*generation=2\b", logs)
        assert sorted(bindings) == ["0", "1"], bindings
        outcomes = re.findall(r"AUTO_MEMBER_TERMINAL[^\n]*worker=(\d+)[^\n]*runtime_status=(-?\d+)", logs)
        assert sorted(outcomes) == [("0", "-101"), ("1", "-12")], outcomes
    assert "507014" not in logs and "AUTO_RECOVERY_FAILED" not in logs
    with capfd.disabled():
        if not fault:
            print(
                f"[PYPTO_RECOVERY_TEST] HEALTHY_DISPATCH_LATENCY mode={mode} "
                f"samples_ns={durations[1:]} median_ns={int(statistics.median(durations[1:]))}",
                flush=True,
            )
        print(
            f"[PYPTO_RECOVERY_TEST] PROGRAM_CONTINUED_PASS mode={mode} counts={stats.reshape(2, 8).tolist()}",
            flush=True,
        )


@pl.jit.incore
def urma_warm_kernel(
    x: pl.Tensor[[1, N], pl.FP32],
    source: pl.Out[pld.DistributedTensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> tuple[pld.DistributedTensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
    source = pl.store(pl.add(pl.load(x, [0, 0], [1, N]), 1.0), [0, 0], source)
    return source, stats


@pl.jit.incore
def urma_entry_kernel(
    source: pld.DistributedTensor[[1, N], pl.FP32],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, 8], pl.INT32]:
    return stats


@pl.jit.incore
def urma_get_kernel(
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, 8], pl.INT32]:
    return stats


@pl.jit.incore
def urma_finish_kernel(
    received: pld.DistributedTensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
    out = pl.store(pl.add(pl.load(received, [0, 0], [1, N]), 3.0), [0, 0], out)
    return out, stats


@pl.jit
def urma_warmup(
    x: pl.Tensor[[1, N], pl.FP32],
    source: pl.Out[pld.DistributedTensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> tuple[pld.DistributedTensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
    source, stats = urma_warm_kernel(x, source, flags, stats, mode)
    return source, stats


@pl.jit
def urma_exchange(
    source: pld.DistributedTensor[[1, N], pl.FP32],
    received: pl.Out[pld.DistributedTensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
    peer: pl.Scalar[pl.INT32],
) -> tuple[pld.DistributedTensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
    stats = urma_entry_kernel(source, flags, stats, mode)
    stats = urma_get_kernel(flags, stats, mode)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="urma_payload"):
        pld.tensor.get_async(received, peer=peer, src=source)
    return received, stats


@pl.jit
def urma_finish(
    received: pld.DistributedTensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[1, N], pl.FP32], pl.Tensor[[1, 8], pl.INT32]]:
    return urma_finish_kernel(received, out, flags, stats, mode)


@pl.jit.host
def urma_program(
    x: pl.Tensor[[2, 1, N], pl.FP32],
    y: pl.Out[pl.Tensor[[2, 1, N], pl.FP32]],
    stats: pl.InOut[pl.Tensor[[2, 1, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
) -> pl.Tensor[[2, 1, N], pl.FP32]:
    source_buf = pld.alloc_window_buffer([1, N], dtype=pl.FP32)
    recv_buf = pld.alloc_window_buffer([1, N], dtype=pl.FP32)
    flags_buf = pld.alloc_window_buffer([1024], dtype=pl.UINT8)
    for rank in pl.range(pld.world_size()):
        source = pld.window(source_buf, [1, N], dtype=pl.FP32)
        received = pld.window(recv_buf, [1, N], dtype=pl.FP32)
        flags = pld.window(flags_buf, [1024], dtype=pl.UINT8)
        local_stats = stats[rank]
        source, local_stats = urma_warmup(x[rank], source, flags, local_stats, mode, device=rank)
        source = pld.window(source_buf, [1, N], dtype=pl.FP32)
        received, local_stats = urma_exchange(
            source, received, flags, local_stats, mode, 1 - rank, device=rank
        )
        received = pld.window(recv_buf, [1, N], dtype=pl.FP32)
        out, local_stats = urma_finish(received, y[rank], flags, local_stats, mode, device=rank)
    return y


@pytest.mark.timeout(240)
@pytest.mark.skipif(not _urma_enabled(), reason="URMA recovery requires the URMA runtime build")
@pytest.mark.parametrize("mode", ["baseline", "healthy", "transient"])
def test_urma_program_continues(test_config, device_ids, mode, capfd, caplog, monkeypatch, tmp_path):
    if test_config.platform != "a5" or len(device_ids) != 2:
        pytest.skip("URMA recovery requires exactly two local A5 devices")
    caplog.set_level(logging.INFO, logger="simpler")
    instrument_wrappers(monkeypatch, backend="urma")
    x = torch.arange(2 * N, dtype=torch.float32).reshape(2, 1, N).share_memory_()
    y = torch.zeros_like(x).share_memory_()
    stats = torch.zeros((2, 1, 8), dtype=torch.int32).share_memory_()
    fault = int(mode == "transient")
    compiled = urma_program.compile(
        x,
        y,
        stats,
        fault,
        config=RunConfig(
            platform="a5",
            save_kernels_dir=str(tmp_path / "program"),
            distributed_config=DistributedConfig(device_ids=device_ids, aicpu_thread_num=2),
        ),
    )
    proofs = json.loads((compiled.output_dir / "distributed_meta.json").read_text())["recovery_effects"]
    assert proofs["urma_exchange"]["safe"], proofs["urma_exchange"]
    wrappers = list(compiled.output_dir.glob("next_levels/urma_exchange/kernels/**/*.cpp"))
    assert any("send_request_entry(ctx, UrmaTget(" in path.read_text() for path in wrappers)
    observed = gate_graph_builder(monkeypatch) if fault else []
    recovery = None if mode == "baseline" else OperatorRecoveryConfig()
    with compiled.prepare(operator_recovery=recovery) as runtime:
        runtime(x, y, stats, fault)
    torch.testing.assert_close(y, x.flip(0) + 4, rtol=0, atol=0)
    expected = torch.tensor([1, 1 + fault, 1, 1, 1, 1, 0, 0], dtype=torch.int32).repeat(2, 1)
    assert torch.equal(stats.reshape(2, 8), expected), stats.tolist()
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    with capfd.disabled():
        print(logs, end="", flush=True)
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_DISPATCH ", logs)) == fault
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_COMMIT ", logs)) == fault
    assert len(re.findall(r"STOP_DELIVERED[^\n]* active=1", logs)) == fault
    if fault:
        assert len(observed) == 1
        outcomes = re.findall(r"AUTO_MEMBER_TERMINAL[^\n]*worker=(\d+)[^\n]*runtime_status=(-?\d+)", logs)
        assert sorted(outcomes) == [("0", "-101"), ("1", "-12")], outcomes
        bindings = re.findall(r"SUBMISSION_BINDING_CURRENT[^\n]*worker=(\d+)[^\n]*generation=2\b", logs)
        assert sorted(bindings) == ["0", "1"], bindings
        drains = re.findall(
            r"URMA_DRAIN_RESULT[^\n]*state=(\d+)[^\n]*read_rc=(-?\d+)[^\n]*timed_out=(\d+)", logs
        )
        assert len(drains) == 2 and all(proof == ("0", "0", "0") for proof in drains), drains
    identities = {}
    for stage, rank, allocation, fields in re.findall(
        r"URMA_BINDING stage=(READY|COMMIT) rank=(\d+) allocation_id=(\d+)([^\n]*)", logs
    ):
        assert (stage, rank) not in identities
        addresses = dict(re.findall(r"(\w+)=(0x[0-9a-fA-F]+)", fields))
        assert set(addresses) == {
            "context_c",
            "hccl",
            "comm_stream",
            "workspace",
            "local_window",
            "device_ctx",
        }
        assert all(int(value, 16) != 0 for value in addresses.values())
        identities[stage, rank] = allocation, addresses
    assert len(identities) == 2 + 2 * fault
    for rank in ("0", "1"):
        assert ("READY", rank) in identities
        if fault:
            assert identities["READY", rank] == identities["COMMIT", rank]
    assert "507014" not in logs and "AUTO_RECOVERY_FAILED" not in logs
    assert "SDMA_STOP_RESULT" not in logs and "URMA_CQE_QUERY_FAILED" not in logs
    with capfd.disabled():
        print(
            f"[PYPTO_RECOVERY_TEST] URMA_PROGRAM_CONTINUED_PASS mode={mode} "
            f"counts={stats.reshape(2, 8).tolist()}",
            flush=True,
        )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mode", ["healthy", "transient"])
def test_recovery_dfx_labels(test_config, device_ids, mode, capfd, monkeypatch, tmp_path):
    """Check real SDMA capture labels, including replay."""
    if _urma_enabled():
        pytest.skip("SDMA-only DFX test: URMA two-pass domain registration is not yet fixed")

    from pypto.runtime.distributed_runner import _run_directory  # noqa: PLC0415
    from simpler_setup.tools.swimlane_converter import load_kernel_config  # noqa: PLC0415

    if test_config.platform != "a5" or len(device_ids) != 2:
        pytest.skip("recovery DFX requires exactly two local A5 devices")
    backend = "sdma"
    instrument_wrappers(monkeypatch, backend=backend)
    x = torch.arange(2 * N, dtype=torch.float32).reshape(2, 1, N).share_memory_()
    y = torch.zeros_like(x).share_memory_()
    stats = torch.zeros((2, 1, 8), dtype=torch.int32).share_memory_()
    fault = int(mode == "transient")
    dc = DistributedConfig(device_ids=device_ids, num_sub_workers=0, aicpu_thread_num=2)
    compiled = ir.compile(
        build_program(), platform="a5", output_dir=str(tmp_path / "program"), distributed_config=dc
    )
    with compiled.prepare(operator_recovery=OperatorRecoveryConfig()) as runtime:
        callables = runtime._states[compiled]["chip_cids"]
        programs = {handle.digest.hex(): name for name, handle in callables.items()}
        runtime(x, y, stats, fault, config=RunConfig(platform="a5", enable_chip_swimlane=1))
    torch.testing.assert_close(y, x.flip(0) + 4, rtol=0, atol=0)
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    with capfd.disabled():
        print(logs, end="", flush=True)
    dispatches = len(re.findall(r"\[PTO_RECOVERY\] REPLAY_DISPATCH ", logs))
    commits = len(re.findall(r"\[PTO_RECOVERY\] REPLAY_COMMIT ", logs))
    # Swimlane executes graph and timing passes. Their mutable state is not
    # restored by the existing DFX protocol, so injection may fire in either/both.
    assert dispatches == commits
    assert dispatches > 0 if fault else dispatches == 0
    assert "507014" not in logs and "AUTO_RECOVERY_FAILED" not in logs
    assert "No L2 program recorded" not in logs
    assert "Skipping L3 swimlane name_map" not in logs
    assert "Skipping L3 swimlane conversion" not in logs
    dfx = _run_directory(compiled) / "dfx_outputs"
    captures = []
    for rank in range(2):
        rank_programs = set()
        for records in sorted((dfx / f"rank{rank}").glob("d[0-9]*/chip_swimlane_records.json")):
            capture = records.parent
            identity = json.loads((capture / "dispatch_identity.json").read_text())
            name = programs[identity["callable_digest"]]
            name_map = json.loads((capture / "name_map.json").read_text())
            expected = load_kernel_config(
                str(compiled.output_dir / "next_levels" / name / "kernel_config.py")
            )
            assert expected and name_map["callable_id_to_name"] == expected
            assert list(capture.glob("merged_swimlane_*.json")), capture
            rank_programs.add(name)
            captures.append((rank, capture.name, name))
        assert rank_programs == set(programs.values()), rank_programs
    with capfd.disabled():
        print(
            f"[PYPTO_RECOVERY_TEST] RECOVERY_DFX_HARDWARE_PASS backend={backend} mode={mode} "
            f"replays={dispatches} captures={captures}",
            flush=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
