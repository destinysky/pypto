# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Conservative replay proofs from lowered IR and operator memory semantics.

The proof covers explicit tensor effects, not merely parameter directions.
Reads overlapping a write require restoration. Unknown code is never promoted
by its function name, and concrete argument aliases are checked at dispatch.
"""

from dataclasses import dataclass

from pypto._function_attrs import BUILTIN_TEMPLATE_DIR_ATTR, EXTERNAL_SOURCE_ATTR
from pypto.pypto_core import ir


@dataclass
class _Summary:
    effects: list[int]
    returns: list[frozenset[int]]
    reasons: list[str]


class _FunctionEffects(ir.IRVisitor):
    def __init__(self, function, resolve):
        super().__init__()
        self.function = function
        self.resolve = resolve
        self.env = {p.unique_id: frozenset({i}) for i, p in enumerate(function.params)}
        self.bases = {}
        self.effects = [0] * len(function.params)
        self.returns = []
        self.reasons = []
        for i, parameter in enumerate(function.params):
            if isinstance(parameter.type, ir.TensorType):
                memory = parameter.type.memref
                if memory is not None:
                    key = memory.base_.unique_id
                    self.bases[key] = self.bases.get(key, frozenset()) | {i}

    def unknown(self, reason):
        if reason not in self.reasons:
            self.reasons.append(reason)

    def storage(self, value):
        if isinstance(value, ir.Var) and value.unique_id in self.env:
            return self.env[value.unique_id]
        if isinstance(value.type, ir.TensorType):
            memory = value.type.memref
            if memory is None:
                return frozenset({-1})
            if memory.base_.unique_id in self.bases:
                return self.bases[memory.base_.unique_id]
            if memory.is_pinned_:
                return frozenset({-1})
        return frozenset()

    def access(self, roots, effect):
        if not effect:
            return
        if not roots and effect & 1:
            self.unknown("private GM read requires an initialized-region proof")
        for root in roots:
            if root < 0:
                self.unknown("unresolved tensor storage or pointer alias")
            elif isinstance(self.function.params[root].type, ir.TensorType):
                self.effects[root] |= effect

    def evaluate(self, value):
        if isinstance(value, (ir.Call, ir.Submit)):
            return self.call(value)
        if isinstance(value, ir.MakeTuple):
            return tuple(self.evaluate(x) for x in value.elements)
        if isinstance(value, ir.TupleGetItemExpr):
            result = self.evaluate(value.tuple)
            if isinstance(result, tuple) and 0 <= value.index < len(result):
                return result[value.index]
            self.unknown("unresolved tuple return alias")
            return frozenset({-1}) if isinstance(value.type, ir.TensorType) else frozenset()
        if isinstance(value, ir.Var):
            return self.storage(value)
        if isinstance(value.type, ir.PtrType):
            self.unknown("unclassified pointer arithmetic")
        super().visit_expr(value)
        return self.storage(value)

    def call(self, call):
        name = call.op.name
        arguments = [self.evaluate(arg) for arg in call.args]
        summary = self.resolve(name)
        if summary is not None:
            return self.function_call(call, arguments, summary)
        try:
            properties = ir.get_op_replay_properties(name)
        except Exception as error:
            self.unknown(f"unregistered operation: {name} ({type(error).__name__})")
            return frozenset({-1})
        if not properties["deterministic"]:
            self.unknown(f"unclassified operation effects: {name}")
        if properties["accesses_memory"]:
            for index, (argument, roots) in enumerate(zip(call.args, arguments)):
                if isinstance(argument.type, ir.TensorType) and isinstance(roots, frozenset):
                    effect = ir.get_op_arg_effect(name, index, **dict(call.kwargs))
                    self.access(
                        roots, 1 if effect == ir.ArgEffect.Read else 2 if effect == ir.ArgEffect.Write else 3
                    )
                elif isinstance(argument.type, ir.PtrType):
                    self.unknown(f"raw pointer operand: {name}")
        alias = properties["result_alias"]
        if alias is not None and alias < len(arguments):
            return arguments[alias]
        return self.storage(call)

    def function_call(self, call, arguments, summary):
        if len(arguments) != len(summary.effects):
            self.unknown(f"incomplete call arguments: {call.op.name}")
            return self.unknown_result(call.type)
        self.reasons.extend(r for r in summary.reasons if r not in self.reasons)
        for roots, effect in zip(arguments, summary.effects):
            if isinstance(roots, frozenset):
                self.access(roots, effect)
        if not summary.returns and summary.reasons:
            return self.unknown_result(call.type)
        results = []
        for returned in summary.returns:
            roots = frozenset()
            for index in returned:
                if index < 0 or index >= len(arguments) or not isinstance(arguments[index], frozenset):
                    roots |= {-1}
                else:
                    roots |= arguments[index]
            results.append(roots)
        return tuple(results) if len(results) != 1 else results[0]

    def unknown_result(self, result_type):
        if isinstance(result_type, ir.TupleType):
            return tuple(self.unknown_result(element) for element in result_type.types)
        return frozenset({-1}) if isinstance(result_type, ir.TensorType) else frozenset()

    def visit_assign_stmt(self, statement):
        roots = self.evaluate(statement.value)
        if roots == frozenset({-1}) and isinstance(statement.value, (ir.Call, ir.Submit)):
            # Assignment types carry the post-planning allocation even when
            # the call expression itself retains its pre-planning type.
            roots = self.storage(statement.var)
        self.env[statement.var.unique_id] = roots
        if isinstance(statement.var.type, ir.TensorType) and isinstance(roots, frozenset):
            memory = statement.var.type.memref
            if memory is not None:
                key = memory.base_.unique_id
                self.bases[key] = self.bases.get(key, frozenset()) | roots

    def visit_call(self, call):
        self.call(call)

    def visit_submit(self, submit):
        self.call(submit)
        for dependency in submit.deps:
            self.evaluate(dependency)

    def visit_return_stmt(self, statement):
        returned = []
        for value in statement.value:
            roots = self.evaluate(value)
            if isinstance(roots, tuple):
                returned.extend(roots)
            else:
                returned.append(roots)
        if any(not isinstance(roots, frozenset) for roots in returned):
            self.unknown("nested tensor return alias requires a structural proof")
            returned = [frozenset({-1}) for _ in returned]
        if not self.returns:
            self.returns = returned
        else:
            self.returns = [a | b for a, b in zip(self.returns, returned)]

    def visit_inline_stmt(self, statement):
        self.unknown("inline source has opaque side effects")

    def visit_for_stmt(self, statement):
        for argument in statement.iter_args:
            self.env[argument.unique_id] = self.evaluate(argument.initValue)
        super().visit_for_stmt(statement)
        for result in statement.return_vars:
            if isinstance(result.type, ir.TensorType):
                self.env[result.unique_id] = frozenset({-1})

    def visit_while_stmt(self, statement):
        # Mutable loop-carried storage needs a version/initialization proof.
        self.unknown("while loop requires a bounded state proof")
        super().visit_while_stmt(statement)

    def visit_if_stmt(self, statement):
        super().visit_if_stmt(statement)
        for result in statement.return_vars:
            if isinstance(result.type, ir.TensorType):
                self.env[result.unique_id] = frozenset({-1})


def analyze_replay_effects(program: ir.Program) -> dict:
    """Produce compiler-owned per-function effects for artifact persistence.

    Unknown and read/write effects stay non-replayable. A write declaration
    promises neither complete output coverage nor absence of concrete aliases.
    """
    functions = {function.name: function for function in program.functions.values()}
    cache = {}
    active = set()

    def resolve(name):
        if name not in functions:
            return None
        if name in cache:
            return cache[name]
        function = functions[name]
        if (
            name in active
            or function.body is None
            or function.requires_runtime_binding
            or EXTERNAL_SOURCE_ATTR in function.attrs
            or BUILTIN_TEMPLATE_DIR_ATTR in function.attrs
        ):
            return _Summary([3] * len(function.params), [], [f"opaque or recursive function: {name}"])
        active.add(name)
        visitor = _FunctionEffects(function, resolve)
        visitor.visit_stmt(function.body)
        active.remove(name)
        cache[name] = _Summary(visitor.effects, visitor.returns, visitor.reasons)
        return cache[name]

    result = {}
    for name, function in functions.items():
        summary = resolve(name)
        effects = [
            effect
            for param, effect in zip(function.params, summary.effects)
            if isinstance(param.type, ir.TensorType)
        ]
        reasons = list(summary.reasons)
        if 3 in effects:
            reasons.append("task reads storage that it may modify; restoration is required")
        result[name] = {
            "safe": not reasons,
            "complete": not summary.reasons,
            "tensor_effects": effects,
            "reasons": reasons,
        }
    return result
