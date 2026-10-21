# Local A5 operator recovery

Prepared distributed programs can opt into Simpler's automatic SDMA or URMA recovery:

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
eligibility, recovers the selected transport and rebinds arguments, then replays and
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
SDMA and URMA share this coordination contract; CCU recovery is not implemented.

On local A5 URMA, `pld.tensor.get_async(dst, peer=peer, src=src)` submits a real
URMA GET and registers its CQE with Simpler. Both tensors must be direct window
parameters in one domain, with equal positive static, contiguous full slices
of byte-addressable elements, at most 256 MiB. The peer must be a remote rank.
Use one GET per dedicated top-level `pl.at(CORE_GROUP)` scope: its TaskId completes only
after the CQE is successful. Payload consumers belong in subsequent dependent
tasks; an in-kernel read after submission is rejected. The compiler derives
read/write effects and replay safety just as for other operations. No safety
declaration is needed. Existing synchronous `get` remains unchanged and uses
MTE staging; selecting a URMA workspace does not rewrite it to URMA.

The A5 runtime currently selects one workspace backend per build. Build with
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON` for URMA and `OFF` for default SDMA.
Rebuild after switching; changing the execution environment alone cannot change
an existing binary. Generated URMA host orchestration checks every rank's actual
domain engine before the first task, including when recovery is disabled.
URMA recovery proves cooperative PTO retirement, drains the existing queues by
fresh successful CQEs, then commits and rearms in place. It preserves HCOMM/HCCL,
registered windows and queue identities; it does not destroy channels or reset
the device. Missing/error CQE evidence makes recovery fail and follows the terminal
path. This contract does not claim recovery from a broken physical link.

`tests/st/distributed/test_l3_operator_recovery.py` compiles a real PyPTO
program: warmup → exchange → finish. Its test-only generated wrapper raises an
asynchronous completion error after the peer enters a notification wait. Business arithmetic and synchronous GET operations
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

`test_urma_program_continues` additionally compiles an `@pl.jit.host` program
with generated URMA payload GET, using the same warmup → exchange → finish
contract. The transient case injects a null-address completion condition on rank
0 before its ready notification; rank 1 requires automatic coordinated STOP.
This is a software completion-contract error, not a fabricated hardware CQE.
Tests require correct payload, counts/order, original-graph continuation, one
replay/commit, successful CQE drain and preserved communication identities.
SDMA tests remain selected for the default build; URMA tests require the URMA build.

The SDMA healthy and baseline cases each dispatch four times on one prepared worker.
`baseline` omits recovery; `healthy` enables it. `HEALTHY_DISPATCH_LATENCY` reports
the last three host dispatch/wait durations after the first-call warmup, excluding
compilation and worker initialization. Domain lifecycle/copies are included.
These short samples are a smoke comparison, not a zero-overhead guarantee.
