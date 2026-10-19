# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Automatic replay proofs must distinguish effects from argument directions."""

import json
import runpy
from pathlib import Path

import pypto.language as pl
import pytest
from pypto.ir.replay_analysis import analyze_replay_effects


def test_disjoint_overwrite_is_proved_from_operations():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[1, 16], pl.FP32], y: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
            value = pl.load(x, [0, 0], [1, 16])
            result = pl.add(value, 1.0)
            y = pl.store(result, [0, 0], y)
            return y

    proof = analyze_replay_effects(Program)["kernel"]
    assert proof == {"safe": True, "complete": True, "tensor_effects": [1, 2], "reasons": []}


def test_read_write_storage_requires_restoration():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.InOut[pl.Tensor[[1, 16], pl.FP32]]):
            value = pl.load(x, [0, 0], [1, 16])
            x = pl.store(pl.add(value, 1.0), [0, 0], x)
            return x

    proof = analyze_replay_effects(Program)["kernel"]
    assert proof["tensor_effects"] == [3]
    assert not proof["safe"]
    assert any("restoration" in reason for reason in proof["reasons"])


def test_callee_effects_reach_the_chip_task():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[1, 16], pl.FP32], y: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
            y = pl.store(pl.load(x, [0, 0], [1, 16]), [0, 0], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def task(self, x: pl.Tensor[[1, 16], pl.FP32], y: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
            y = self.kernel(x, y)
            return y

    proof = analyze_replay_effects(Program)["task"]
    assert proof["safe"] and proof["tensor_effects"] == [1, 2]


def test_opaque_source_is_never_approved_by_its_name(tmp_path):
    source = tmp_path / "opaque.cpp"
    source.write_text('extern "C" void kernel_entry(long*) {}')

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, external_source=str(source))
        def exchange(self, x: pl.Tensor[[1, 16], pl.FP32], y: pl.Out[pl.Tensor[[1, 16], pl.FP32]]): ...

    proof = analyze_replay_effects(Program)["exchange"]
    assert not proof["safe"]
    assert any("opaque" in reason for reason in proof["reasons"])


@pytest.mark.parametrize("origin", ["builtin-template", "runtime-binding"])
def test_implementation_outside_the_ir_is_opaque_even_with_an_empty_body(origin):
    from pypto._function_attrs import BUILTIN_TEMPLATE_DIR_ATTR  # noqa: PLC0415
    from pypto.pypto_core import ir  # noqa: PLC0415

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(self, out: pl.Out[pl.Tensor[[1, 16], pl.FP32]]) -> pl.Tensor[[1, 16], pl.FP32]:
            return out

    function = Program.get_function("kernel")
    hidden = ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        function.return_types,
        ir.SeqStmts([], function.span),
        function.span,
        type=function.func_type,
        level=function.level,
        role=function.role,
        attrs={BUILTIN_TEMPLATE_DIR_ATTR: ":opaque:test"} if origin == "builtin-template" else {},
        requires_runtime_binding=origin == "runtime-binding",
    )
    proof = analyze_replay_effects(ir.Program([hidden], "Hidden", function.span))["kernel"]
    assert not proof["safe"] and not proof["complete"]
    assert any("opaque" in reason for reason in proof["reasons"])


def test_opaque_multi_result_task_still_compiles_for_normal_execution(tmp_path):
    from pypto import ir  # noqa: PLC0415

    source = tmp_path / "external.cpp"
    source.write_text('extern "C" void kernel_entry(long*) {}')

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, external_source=str(source))
        def kernel(
            self,
            x: pl.Tensor[[1, 16], pl.FP32],
            a: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            b: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[1, 16], pl.FP32], pl.Tensor[[1, 16], pl.FP32]]: ...

        @pl.function(type=pl.FunctionType.Orchestration)
        def task(
            self,
            x: pl.Tensor[[1, 16], pl.FP32],
            a: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            b: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[1, 16], pl.FP32], pl.Tensor[[1, 16], pl.FP32]]:
            a, b = self.kernel(x, a, b)
            return a, b

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(
            self,
            x: pl.Tensor[[2, 1, 16], pl.FP32],
            a: pl.Out[pl.Tensor[[2, 1, 16], pl.FP32]],
            b: pl.Out[pl.Tensor[[2, 1, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[2, 1, 16], pl.FP32], pl.Tensor[[2, 1, 16], pl.FP32]]:
            for rank in pl.range(2):
                self.task(x[rank], a[rank], b[rank], device=rank)
            return a, b

    compiled = ir.compile(Program, platform="a5", skip_ptoas=True, output_dir=str(tmp_path / "program"))
    proof = json.loads((compiled.output_dir / "distributed_meta.json").read_text())["recovery_effects"]
    assert not proof["task"]["safe"] and not proof["task"]["complete"]
    assert (compiled.output_dir / "next_levels/task/kernel_config.py").is_file()


def test_full_distributed_compile_persists_generated_task_proofs(tmp_path, monkeypatch):
    from pypto import ir  # noqa: PLC0415

    test_source = Path(__file__).resolve().parents[3] / "st/distributed/test_l3_operator_recovery.py"
    definitions = runpy.run_path(str(test_source))
    # Distributed window back-references have no printer syntax. Keep the full
    # structural verifier, as in the existing distributed codegen tests.
    from pypto.pypto_core import passes  # noqa: PLC0415

    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        compiled = ir.compile(
            definitions["build_program"](),
            platform="a5",
            skip_ptoas=True,
            output_dir=str(tmp_path / "program"),
            distributed_config=ir.DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
        )
    proof = json.loads((compiled.output_dir / "distributed_meta.json").read_text())["recovery_effects"]
    assert proof["exchange"]["safe"], proof["exchange"]
    assert proof["exchange"]["tensor_effects"][:2] == [1, 2]
    assert proof["warmup"]["safe"], proof["warmup"]
    generated = (compiled.output_dir / "orchestration/host_orch.py").read_text()
    assert "_bind_recovery_domain" in generated and "'tensor'" in generated
    from pypto.backend import pto_backend  # noqa: PLC0415

    definitions["instrument_wrappers"](monkeypatch)
    kernels = [f for f in compiled.program.functions.values() if f.func_type == ir.FunctionType.AIV]
    assert len(kernels) == 4
    for function in kernels:
        wrapper = pto_backend._generate_kernel_wrapper(function, "// generated business body")
        assert "inject_recovery_fault(args," in wrapper
        assert "// generated business body" in wrapper
    print("[PYPTO_RECOVERY_TEST] AUTOMATIC_REPLAY_PROOF_PASS", flush=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
