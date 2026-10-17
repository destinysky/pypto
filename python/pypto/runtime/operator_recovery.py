# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Opt-in replay contracts and full-rank L3 task grouping."""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

_RUN_IDS = itertools.count(1)


@dataclass(frozen=True)
class OperatorRecoveryConfig:
    """Declare replay-safe chip functions for a local A5 SDMA program.

    ``replay_safe`` names chip orchestration functions, not in-core kernels.
    The caller guarantees valid replay inputs and repeatable/discardable effects
    for each named function. Other functions retain UNKNOWN replay safety.
    Graph, rank-stop, resource and retry checks are still enforced by Simpler.
    """

    replay_safe: tuple[str, ...]
    max_retries: int = 1
    backoff_s: float = 0.0
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.replay_safe, tuple)
            or not self.replay_safe
            or any(not isinstance(name, str) or not name for name in self.replay_safe)
        ):
            raise ValueError("replay_safe must be a nonempty tuple of chip function names")
        if len(set(self.replay_safe)) != len(self.replay_safe):
            raise ValueError(f"replay_safe contains duplicate names: {self.replay_safe!r}")
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
class _CapturedTask:
    owner: Any
    sequence: int


@dataclass
class _Group:
    name: str
    callable_id: Any
    config: Any
    members: dict[int, Any] = field(default_factory=dict)


class _RecoveryGraph:
    """Build one full-communicator group per rank-local dispatch ordinal.

    Capture completes before native task admission. All ranks must have the same
    chip-function sequence; explicit TaskId dependencies must point backwards.
    Domain Buffers remain owned by the submitted run after its allocation scope
    closes. Native TaskArgs copies and the original DAG survive replay.
    """

    def __init__(self, config: OperatorRecoveryConfig, ranks: int, callables: dict[str, Any]) -> None:
        unknown = set(config.replay_safe) - callables.keys()
        if unknown:
            raise ValueError(f"replay_safe names are not registered chip functions: {sorted(unknown)}")
        self.config = config
        self.ranks = ranks
        self.logical_run_id = next(_RUN_IDS)
        if self.logical_run_id >= 1 << 64:
            raise OverflowError("logical recovery run identity exhausted")
        self._counts = [0] * ranks
        self._groups: list[_Group] = []
        self._arguments: dict[int, list[tuple[str, int, Any, Any]]] = {}
        self._dependencies: dict[int, list[_CapturedTask]] = {}
        self._contexts: dict[int, Any] = {}
        self._domain_id: int | None = None

    def annotate(self, args: Any, entries: list[tuple[str, int, Any, Any]]) -> None:
        self._arguments[id(args)] = entries
        for kind, _index, context, _buffer in entries:
            if kind not in ("window", "context"):
                raise ValueError(f"invalid recovery argument kind: {kind!r}")
            if context is None:
                raise ValueError(
                    "recovery cannot bind an opaque CommCtx parameter; pass a window-derived context"
                )
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
                raise ValueError("domain context changed during graph capture")
            self._domain_id = binding.allocation_id
            self._contexts[rank] = context

    def add_dependency(self, args: Any, dependency: Any) -> None:
        if not isinstance(dependency, _CapturedTask) or dependency.owner is not self:
            raise ValueError("recovery dependency must name a task in the same captured graph")
        self._dependencies.setdefault(id(args), []).append(dependency)

    def submit(self, callable_id: Any, args: Any, config: Any, rank: int, name: str | None) -> _CapturedTask:
        if rank < 0 or rank >= self.ranks or not name:
            raise ValueError("recovery dispatch requires an explicit rank and regenerated chip function name")
        sequence = self._counts[rank]
        self._counts[rank] += 1
        if sequence == len(self._groups):
            self._groups.append(_Group(name, callable_id, config))
        group = self._groups[sequence]
        if group.name != name or group.callable_id != callable_id or group.config is not config:
            raise ValueError(f"rank {rank} dispatch {sequence} differs from the other ranks: {name!r}")
        group.members[rank] = args
        return _CapturedTask(self, sequence)

    def flush(self, orch: Any) -> None:
        from simpler.task_interface import (  # noqa: PLC0415
            RecoveryArgBinding,
            RecoveryArgKind,
            RecoveryMetadata,
            ReplaySafety,
            TaskDomainBinding,
        )

        participants = list(range(self.ranks))
        if not self._groups or len(set(self._counts)) != 1:
            raise ValueError(
                f"recovery requires matching nonempty full-rank task sequences, got {self._counts}"
            )
        if sorted(self._contexts) != participants:
            raise ValueError("recovery requires one run-scoped full-communicator domain on every rank")
        # Validate the entire capture before any group enters native execution.
        for sequence, group in enumerate(self._groups):
            if sorted(group.members) != participants:
                raise ValueError(f"recovery group {sequence} has incomplete participants")
            for rank, args in group.members.items():
                if any(dep.sequence >= sequence for dep in self._dependencies.get(id(args), ())):
                    raise ValueError("recovery dependencies must precede their consumer group")
                context = self._contexts[rank]
                binding = TaskDomainBinding()
                binding.domain = context.recovery_binding
                arguments = []
                for kind, index, owner, buffer in self._arguments.get(id(args), ()):
                    if owner.domain_rank != rank:
                        raise ValueError("task recovery binding belongs to a different rank")
                    entry = RecoveryArgBinding()
                    entry.index = index
                    entry.kind = (
                        RecoveryArgKind.WINDOW_TENSOR if kind == "window" else RecoveryArgKind.COMM_CONTEXT
                    )
                    entry.offset = (
                        int(buffer.base) - binding.domain.local_window_base if kind == "window" else 0
                    )
                    arguments.append(entry)
                binding.arguments = arguments
                args.add_domain_binding(binding)
                metadata = RecoveryMetadata()
                metadata.logical_run_id = self.logical_run_id
                metadata.domain_id = self._domain_id
                metadata.op_seq = sequence + 1
                metadata.rank, metadata.participants = rank, participants
                metadata.replay_safety = (
                    ReplaySafety.REPLAY_SAFE
                    if group.name in self.config.replay_safe
                    else ReplaySafety.UNKNOWN
                )
                args.set_recovery_metadata(metadata)

        handles = []
        for sequence, group in enumerate(self._groups):
            args_list = [group.members[rank] for rank in participants]
            for args in args_list:
                # Full-group order also fences unrelated rank-local successors.
                deps = {dep.sequence for dep in self._dependencies.get(id(args), ())}
                if sequence:
                    deps.add(sequence - 1)
                for dependency in sorted(deps):
                    args.add_dep_wait(handles[dependency])
            handles.append(
                orch.submit_next_level_group(group.callable_id, args_list, group.config, workers=participants)
            )


def _add_dispatch_dependency(orch: Any, args: Any, dependency: Any) -> None:
    capture = _get_recovery(orch)
    if capture is None:
        args.add_dep_wait(dependency)
    else:
        capture.add_dependency(args, dependency)


def _get_recovery(orch: Any) -> _RecoveryGraph | None:
    capture = getattr(orch, "_pypto_recovery", None)
    return capture if isinstance(capture, _RecoveryGraph) else None
