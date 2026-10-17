# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""A generated PyPTO graph survives a stalled-peer SDMA attempt in its middle.

Fault injection lives only in the external test kernel. STOP, resource recovery,
replay and continuation are all performed by the runtime, without test commands.
"""

import re
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import OperatorRecoveryConfig, RunConfig

N = 256
_ROOT = Path(__file__).resolve().parents[3]
_KERNELS = Path(__file__).with_name("kernels")
_INCLUDES = [
    str(_ROOT / "runtime/src/common/tensormap_and_ringbuffer"),
    str(_ROOT / "runtime/src/common"),
    str(_ROOT / "runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime"),
]


@pl.jit.extern(source=_KERNELS / "recovery_protocol.cpp", core_type="aiv", include_dirs=_INCLUDES)
def protocol(
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[8], pl.INT32]],
    source: pld.DistributedTensor[[1, N], pl.FP32],
    ctx: pld.CommCtx,
    phase: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
) -> pl.Tensor[[8], pl.INT32]: ...


@pl.jit.extern(source=_KERNELS / "recovery_transfer.cpp", core_type="aiv", include_dirs=_INCLUDES)
def transfer(
    source: pld.DistributedTensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    token: pl.Tensor[[8], pl.INT32],
    ctx: pld.CommCtx,
) -> pl.Tensor[[1, N], pl.FP32]: ...


@pl.jit.incore
def add_one(x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
    value = pl.load(x, [0, 0], [1, N])
    return pl.store(pl.add(value, 1.0), [0, 0], out)


@pl.jit.incore
def stage(x: pl.Tensor[[1, N], pl.FP32], window: pl.InOut[pld.DistributedTensor[[1, N], pl.FP32]]):
    value = pl.load(x, [0, 0], [1, N])
    return pl.store(value, [0, 0], window)


@pl.jit.incore
def add_three(x: pl.Tensor[[1, N], pl.FP32], out: pl.Out[pl.Tensor[[1, N], pl.FP32]]):
    value = pl.load(x, [0, 0], [1, N])
    return pl.store(pl.add(value, 3.0), [0, 0], out)


@pl.jit
def warmup(
    x: pl.Tensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[8], pl.INT32]],
    source: pld.DistributedTensor[[1, N], pl.FP32],
):
    out = add_one(x, out)
    stats = protocol(flags, stats, source, pld.get_comm_ctx(flags), 2, 0)
    return out, stats


@pl.jit
def exchange(
    x: pl.Tensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    source: pl.InOut[pld.DistributedTensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
):
    flags_ctx = pld.get_comm_ctx(flags)
    source_ctx = pld.get_comm_ctx(source)
    source = stage(x, source)
    stats = protocol(flags, stats, source, flags_ctx, 0, mode)
    stats = protocol(flags, stats, source, flags_ctx, 1, mode)
    out = transfer(source, out, stats, source_ctx)
    return out, source, stats


@pl.jit
def finish(
    x: pl.Tensor[[1, N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    flags: pld.DistributedTensor[[1024], pl.UINT8],
    stats: pl.InOut[pl.Tensor[[8], pl.INT32]],
    source: pld.DistributedTensor[[1, N], pl.FP32],
):
    out = add_three(x, out)
    stats = protocol(flags, stats, source, pld.get_comm_ctx(flags), 3, 0)
    return out, stats


@pl.jit.host
def recovery_program(
    x: pl.Tensor[[2, 1, N], pl.FP32],
    y: pl.Out[pl.Tensor[[2, 1, N], pl.FP32]],
    stats: pl.InOut[pl.Tensor[[2, 8], pl.INT32]],
    mode: pl.Scalar[pl.INT32],
):
    data_buf = pld.alloc_window_buffer([1, N], dtype=pl.FP32)
    flags_buf = pld.alloc_window_buffer([1024], dtype=pl.UINT8)
    warmed = pl.create_tensor([2, 1, N], dtype=pl.FP32)
    received = pl.create_tensor([2, 1, N], dtype=pl.FP32)
    for rank in pl.range(pld.world_size()):
        source = pld.window(data_buf, [1, N], dtype=pl.FP32)
        flags = pld.window(flags_buf, [1024], dtype=pl.UINT8)
        local_x = x[rank]
        w = warmed[rank]
        recv = received[rank]
        out = y[rank]
        s = stats[rank]
        w, s = warmup(local_x, w, flags, s, source, device=rank)
        recv, source, s = exchange(w, recv, source, flags, s, mode, device=rank)
        source = pld.window(data_buf, [1, N], dtype=pl.FP32)
        out, s = finish(recv, out, flags, s, source, device=rank)
    return y


@pytest.mark.timeout(240)
@pytest.mark.parametrize("mode", ["healthy", "transient"])
def test_program_continues(test_config, device_ids, mode, capfd):
    if test_config.platform != "a5" or len(device_ids) != 2:
        pytest.skip("operator recovery requires exactly two local A5 devices")
    config = RunConfig(
        platform="a5",
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0, aicpu_thread_num=2),
    )
    x = torch.arange(2 * N, dtype=torch.float32).reshape(2, 1, N).share_memory_()
    y = torch.zeros_like(x).share_memory_()
    stats = torch.zeros((2, 8), dtype=torch.int32).share_memory_()
    fault = int(mode == "transient")
    compiled = recovery_program.compile(x, y, stats, fault, config=config)
    with compiled.prepare(operator_recovery=OperatorRecoveryConfig(replay_safe=("exchange",))) as runtime:
        runtime(x, y, stats, fault)
    torch.testing.assert_close(y, x.flip(0) + 4, rtol=0, atol=0)
    expected = torch.tensor([1, 1 + fault, 1, 1, 1, 1, 0, 0], dtype=torch.int32).repeat(2, 1)
    assert torch.equal(stats, expected), f"unexpected execution counts/order: {stats.tolist()}"
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    with capfd.disabled():
        print(logs, end="", flush=True)
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_DISPATCH ", logs)) == fault
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_COMMIT ", logs)) == fault
    assert len(re.findall(r"STOP_DELIVERED[^\n]* active=1", logs)) == fault
    if fault:
        outcomes = re.findall(r"AUTO_MEMBER_TERMINAL[^\n]*worker=(\d+)[^\n]*runtime_status=(-?\d+)", logs)
        assert sorted(outcomes) == [("0", "-101"), ("1", "-12")], outcomes
    assert "507014" not in logs and "AUTOMATIC_RECOVERY_FAILED" not in logs
    with capfd.disabled():
        print(f"[PYPTO_RECOVERY_TEST] PROGRAM_CONTINUED_PASS mode={mode} counts={stats.tolist()}", flush=True)
