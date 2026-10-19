# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compiler-owned replay contracts and incremental full-rank L3 publication."""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

_RUN_IDS = itertools.count(1)


@dataclass(frozen=True)
class OperatorRecoveryConfig:
    """Enable automatic local A5 recovery; configure policy, never semantic safety."""

    max_retries: int = 1
    backoff_s: float = 0.0
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError(f"max_retries must be a nonnegative integer, got {self.max_retries!r}")
        if not math.isfinite(self.backoff_s) or self.backoff_s < 0:
            raise ValueError(f"backoff_s must be finite and nonnegative, got {self.backoff_s!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and positive, got {self.timeout_s!r}")

    def worker_options(self) -> dict[str, Any]:
        """Return Simpler's existing automatic recovery configuration."""
        return {
            "enable_operator_recovery": True,
            "operator_recovery_max_retries": self.max_retries,
            "operator_recovery_backoff_s": self.backoff_s,
            "operator_recovery_timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class _GroupTask:
    owner: Any
    sequence: int


@dataclass
class _Group:
    name: str
    callable_id: Any
    config: Any
    members: dict[int, Any] = field(default_factory=dict)


class _RecoveryGraph:
    """Publish each rank-complete task group without waiting for later groups.

    Rank-local dispatch ordinals must match. Native publication freezes during
    recovery and advances future argument bindings before admitting them.
    """

    def __init__(self, ranks: int, proofs: dict, orch: Any) -> None:
        self.proofs = proofs
        self.orch = orch
        self._handles = []
        self._writers: dict[Any, list[tuple[int, int, int]]] = {}
        self._readers: dict[Any, dict[int, list[tuple[int, int]]]] = {}
        self._barrier: int | None = None
        self._communication_tail: int | None = None
        self.ranks = ranks
        self.logical_run_id = next(_RUN_IDS)
        if self.logical_run_id >= 1 << 64:
            raise OverflowError("logical recovery run identity exhausted")
        self._counts = [0] * ranks
        self._groups: list[_Group] = []
        self._arguments: dict[int, list[tuple[str, int, Any, Any]]] = {}
        self._submitted: set[int] = set()
        self._dependencies: dict[int, list[_GroupTask]] = {}
        self._contexts: dict[int, Any] = {}
        self._domain_id: int | None = None

    def annotate(self, args: Any, entries: list[tuple[str, int, Any, Any]]) -> None:
        if id(args) in self._submitted:
            raise ValueError("cannot annotate an already submitted task member")
        self._arguments[id(args)] = entries
        for kind, _index, context, _buffer in entries:
            if kind == "tensor":
                continue
            if kind not in ("window", "context"):
                raise ValueError(f"invalid recovery argument kind: {kind!r}")
            if context is None:
                raise ValueError(
                    "recovery cannot bind an opaque CommCtx parameter; pass a window-derived context"
                )
            self._remember_context(context)

    def bind_domain(self, contexts) -> None:
        """Register the run-scoped domain even for tasks with no window arguments."""
        for rank in range(self.ranks):
            context = contexts[rank]
            self._remember_context(context)

    def _remember_context(self, context) -> None:
        binding = context.recovery_binding
        if binding is None or context.domain_size != self.ranks:
            raise ValueError("recovery requires a full-communicator domain with native binding metadata")
        if self._domain_id not in (None, binding.allocation_id):
            raise ValueError("operator recovery currently requires exactly one communication domain")
        rank = context.domain_rank
        if rank < 0 or rank >= self.ranks:
            raise ValueError(f"invalid domain rank {rank} for {self.ranks} participants")
        previous = self._contexts.get(rank)
        if previous is not None and previous is not context:
            raise ValueError("domain context changed during graph construction")
        self._domain_id = binding.allocation_id
        self._contexts[rank] = context

    def _accesses(self, group: _Group) -> list[tuple]:
        proof = self.proofs.get(group.name, {})
        effects = proof.get("tensor_effects")
        if not isinstance(effects, list) or any(type(e) is not int or e not in (0, 1, 2, 3) for e in effects):
            raise ValueError(f"missing compiler effects for {group.name!r}; recompile the program")
        accesses = []
        for rank, args in group.members.items():
            entries = {
                index: (kind, owner, value)
                for kind, index, owner, value in self._arguments.get(id(args), ())
                if kind != "context"
            }
            for index, effect in enumerate(effects):
                if not effect:
                    continue
                if index not in entries:
                    raise ValueError(
                        f"missing concrete tensor provenance for {group.name!r} argument {index}"
                    )
                kind, owner, value = entries[index]
                if kind == "window":
                    domain = owner.recovery_binding
                    key = ("window", domain.allocation_id)
                    begin = int(value.base) - domain.local_window_base
                    end = begin + int(value.nbytes)
                elif hasattr(value, "untyped_storage"):
                    key = ("host", int(value.untyped_storage().data_ptr()))
                    begin = int(value.storage_offset()) * value.element_size()
                    # Include gaps in strided views; never assume contiguous.
                    span = sum((n - 1) * abs(s) for n, s in zip(value.shape, value.stride())) + 1
                    end = begin + (span * value.element_size() if value.numel() else 0)
                else:
                    tensor = args.tensor(index)
                    identity = tensor.buffer.identity
                    key = ("wire", bytes(identity.owner_instance_id), identity.buffer_id, identity.generation)
                    begin, end = 0, tensor.buffer.nbytes
                accesses.append((key, begin, end, effect))
        return accesses

    def _effects(self, group: _Group) -> list[tuple] | None:
        proof = self.proofs.get(group.name, {})
        if proof.get("complete") is not True:
            return None
        effects = proof.get("tensor_effects", [])
        for args in group.members.values():
            entries = self._arguments.get(id(args), ())
            # Generated communication ordering tokens are host-only extra args.
            ordering_tokens = int(any(kind == "window" for kind, *_ in entries))
            if args.tensor_count() != len(effects) + ordering_tokens:
                return None
        return self._accesses(group)

    def _safe(self, group: _Group, accesses: list[tuple] | None) -> bool:
        proof = self.proofs.get(group.name, {})
        if proof.get("safe") is not True or accesses is None:
            return False
        for position, (key, begin, end, effect) in enumerate(accesses):
            if effect == 3:
                return False
            if effect & 2:
                for other_position, (other_key, other_begin, other_end, other_effect) in enumerate(accesses):
                    conflicts = other_effect & 1 or (key[0] != "window" and other_effect & 2)
                    if (
                        position != other_position
                        and conflicts
                        and key == other_key
                        and begin < other_end
                        and other_begin < end
                    ):
                        return False
        return True

    def _hazard_dependencies(self, accesses: list[tuple] | None, uses_communication: bool) -> set[int]:
        dependencies = set()
        if self._barrier is not None:
            dependencies.add(self._barrier)
        if uses_communication and self._communication_tail is not None:
            dependencies.add(self._communication_tail)
        if accesses is None:
            return dependencies | set(range(len(self._handles)))
        for key, begin, end, effect in accesses:
            for old_begin, old_end, sequence in self._writers.get(key, ()):
                if begin < old_end and old_begin < end:
                    dependencies.add(sequence)
            if effect & 2:
                for sequence, regions in self._readers.get(key, {}).items():
                    if any(begin < old_end and old_begin < end for old_begin, old_end in regions):
                        dependencies.add(sequence)
        return dependencies

    def _remember_effects(
        self, sequence: int, accesses: list[tuple] | None, uses_communication: bool
    ) -> None:
        if uses_communication:
            self._communication_tail = sequence
        if accesses is None:
            self._writers.clear()
            self._readers.clear()
            self._barrier = sequence
            return
        unique = set(accesses)
        for key, begin, end, effect in unique:
            if not effect & 2:
                continue
            self._writers[key] = [
                old for old in self._writers.get(key, ()) if not (begin <= old[0] and old[1] <= end)
            ]
            readers = self._readers.get(key, {})
            for reader in list(readers):
                readers[reader] = [(a, b) for a, b in readers[reader] if not (begin <= a and b <= end)]
                if not readers[reader]:
                    del readers[reader]
        for key, begin, end, effect in unique:
            if effect & 2:
                self._writers.setdefault(key, []).append((begin, end, sequence))
            if effect & 1:
                self._readers.setdefault(key, {}).setdefault(sequence, []).append((begin, end))

    def add_dependency(self, args: Any, dependency: Any) -> None:
        if id(args) in self._submitted:
            raise ValueError("cannot change dependencies of an already submitted task member")
        if not isinstance(dependency, _GroupTask) or dependency.owner is not self:
            raise ValueError("recovery dependency must name a task in the same graph")
        self._dependencies.setdefault(id(args), []).append(dependency)

    def submit(self, callable_id: Any, args: Any, config: Any, rank: int, name: str | None) -> _GroupTask:
        if id(args) in self._submitted:
            raise ValueError("TaskArgs already belongs to a submitted member")
        if rank < 0 or rank >= self.ranks or not name:
            raise ValueError("recovery dispatch requires an explicit rank and regenerated chip function name")
        sequence = self._counts[rank]
        if any(dep.sequence >= sequence for dep in self._dependencies.get(id(args), ())):
            raise ValueError("recovery dependencies must precede their consumer group")
        if sequence == len(self._groups):
            self._groups.append(_Group(name, callable_id, config))
        group = self._groups[sequence]
        if group.name != name or group.callable_id != callable_id or group.config is not config:
            raise ValueError(f"rank {rank} dispatch {sequence} differs from the other ranks: {name!r}")
        group.members[rank] = args
        self._submitted.add(id(args))
        self._counts[rank] += 1
        while len(self._handles) < len(self._groups):
            ready = self._groups[len(self._handles)]
            if len(ready.members) != self.ranks:
                break
            self._publish(len(self._handles), ready)
        return _GroupTask(self, sequence)

    def finish(self) -> None:
        if not self._groups or len(set(self._counts)) != 1 or len(self._handles) != len(self._groups):
            raise ValueError(
                f"recovery requires matching nonempty full-rank task sequences, got {self._counts}"
            )

    def _publish(self, sequence: int, group: _Group) -> None:
        from simpler.task_interface import (  # noqa: PLC0415
            RecoveryArgBinding,
            RecoveryArgKind,
            RecoveryMetadata,
            ReplaySafety,
            TaskDomainBinding,
        )

        participants = list(range(self.ranks))
        if sorted(self._contexts) != participants:
            raise ValueError("recovery requires one run-scoped full-communicator domain on every rank")
        accesses = self._effects(group)
        safe = self._safe(group, accesses)
        uses_communication = accesses is None or any(
            kind in ("window", "context")
            for args in group.members.values()
            for kind, *_ in self._arguments.get(id(args), ())
        )
        hazards = self._hazard_dependencies(accesses, uses_communication)
        for rank, args in group.members.items():
            context = self._contexts[rank]
            binding = TaskDomainBinding()
            binding.domain = context.recovery_binding
            arguments = []
            for kind, index, owner, buffer in self._arguments.get(id(args), ()):
                if kind == "tensor":
                    continue
                if owner.domain_rank != rank:
                    raise ValueError("task recovery binding belongs to a different rank")
                entry = RecoveryArgBinding()
                entry.index = index
                entry.kind = (
                    RecoveryArgKind.WINDOW_TENSOR if kind == "window" else RecoveryArgKind.COMM_CONTEXT
                )
                entry.offset = int(buffer.base) - binding.domain.local_window_base if kind == "window" else 0
                arguments.append(entry)
            binding.arguments = arguments
            if arguments:
                args.add_domain_binding(binding)
            metadata = RecoveryMetadata()
            metadata.logical_run_id = self.logical_run_id
            metadata.domain_id = self._domain_id
            metadata.op_seq = sequence + 1
            metadata.rank, metadata.participants = rank, participants
            metadata.replay_safety = ReplaySafety.REPLAY_SAFE if safe else ReplaySafety.UNKNOWN
            metadata.uses_communication = uses_communication
            args.set_recovery_metadata(metadata)
            deps = hazards | {dep.sequence for dep in self._dependencies.get(id(args), ())}
            for dependency in sorted(deps):
                args.add_dep_wait(self._handles[dependency])
        self._handles.append(
            self.orch.submit_next_level_group(
                group.callable_id,
                [group.members[rank] for rank in participants],
                group.config,
                workers=participants,
            )
        )
        self._remember_effects(sequence, accesses, uses_communication)


def _add_dispatch_dependency(orch: Any, args: Any, dependency: Any) -> None:
    capture = _get_recovery(orch)
    if capture is None:
        args.add_dep_wait(dependency)
    else:
        capture.add_dependency(args, dependency)


def _get_recovery(orch: Any) -> _RecoveryGraph | None:
    capture = getattr(orch, "_pypto_recovery", None)
    return capture if isinstance(capture, _RecoveryGraph) else None


def _bind_recovery_domain(orch: Any, contexts: Any) -> None:
    capture = _get_recovery(orch)
    if capture is not None:
        capture.bind_domain(contexts)
