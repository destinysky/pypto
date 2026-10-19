# Local A5 operator recovery

Prepared distributed programs can opt into Simpler's automatic SDMA recovery:

```python
from pypto.runtime import OperatorRecoveryConfig

compiled = program.compile(x, y, config=config)
with compiled.prepare(
    operator_recovery=OperatorRecoveryConfig(
        max_retries=1, timeout_s=30.0
    )
) as runtime:
    runtime(x, y)
```

PyPTO analyzes each generated L3 task, including its in-core callees, from
operator effects and tensor alias provenance. Deterministic reads of preserved
inputs and blind overwrites can be replayed. Read/write storage requires a
restore protocol and is rejected; unknown operations, opaque external kernels,
raw pointers, unresolved aliases and private GM reads without an initialized-region
proof remain non-replayable. Concrete argument ranges are checked across ranks before admission, so passing an output as an
input under another parameter cannot bypass the proof. No user function-name
list grants safety. Unsupported semantics still run normally; a failure follows
the terminal path. Old artifacts without compiler effects must be recompiled.
The application configures retry policy only.

PyPTO derives run/operator identities, participants, and window/context argument
bindings from generated code. Each rank-complete ordinal is validated and
submitted immediately while the host callback continues building later groups.
Group ordering follows explicit task dependencies, overlapping RAW/WAR/WAW
accesses and communication-domain protocol order. Shared immutable reads and
disjoint regions do not create a blanket previous-group edge. Unknown effects or
unaccounted native output arguments remain conservative barriers. Existing native
tensor dependencies and per-domain ordering tokens are preserved. Proved compute
work carries execution-only scope; the legacy native metadata default includes
the communication domain.

Full-rank grouping and the current A5 single-frame endpoint constraints remain.
An advanced consumer or unproved/live endpoint work can still force GIVE_UP.
Removing an extra graph edge does not authorize recovery over live execution.

During recovery native publication is frozen: eligibility inspects a stable
published prefix, without requiring the callback to finish. New publications
wait for the recovery episode to resolve and apply its committed binding history
before admission. The retained graph must fit native task/heap capacity;
incremental construction does not imply unbounded graph capacity. Late callback
errors terminate the run; completed effects are not rolled back. Direct device
control must still precede the first task submission. Normal dispatch is
unchanged when recovery is omitted.

On a supported fault, Simpler fences successors, issues identity-checked STOP to
the current attempt's members, collects retirement proofs, validates replay
eligibility, recovers SDMA resources and rebinds arguments, then replays and
commits the failed group. Successful commit releases its original successors.
The application does not call STOP or redispatch. Rejection, failed recovery or
exhausted retry budget follows the existing terminal failure path.

The initial integration requires one local A5 `tensormap_and_ringbuffer` program,
one run-scoped full-communicator domain, explicit rank dispatch, matching chip
function sequences and a shared call configuration across ranks. Host callbacks,
sub-worker tasks, persistent domains, host-level builtin collective dispatches,
multi-program workers and cross-machine execution are rejected. A builtin
collective may be wrapped inside a chip function. Window-derived contexts carry
bindings; opaque context parameters cannot be recovered. This layer describes
bindings and replay semantics without choosing a transport from error codes;
URMA/CCU require backend support and validation before enabling recovery.

`tests/st/distributed/test_l3_operator_recovery.py` compiles a real PyPTO
program: warmup → exchange → finish. Its test-only generated wrapper raises an
asynchronous completion error after the peer enters a notification wait. Business arithmetic and SDMA GET operations
are generated from PyPTO IR and receive no manual safety override. The peer needs
framework STOP; the next attempt uses fresh notification epochs. The tests check
SDMA data, final numerical output, execution counts and order. They do not inject
a physical network failure. No injection hook is added to production code.

The transient case delays construction of the last successor until the framework
observes the fault. It reads recovery status only; STOP, resource recovery and
redispatch remain automatic. `OPEN_GRAPH_RECOVERY_SEEN` plus two
`SUBMISSION_BINDING_CURRENT ... generation=2` records verify open-graph recovery
and current bindings on later submissions. The original output/count assertions
remain required.

The healthy and baseline cases each dispatch four times on one prepared worker.
`baseline` omits recovery; `healthy` enables it. `HEALTHY_DISPATCH_LATENCY` reports
the last three host dispatch/wait durations after the first-call warmup, excluding
compilation and worker initialization. Domain lifecycle/copies are included.
These short samples are a smoke comparison, not a zero-overhead guarantee.
