# Local A5 operator recovery

Prepared distributed programs can opt into Simpler's automatic SDMA recovery:

```python
from pypto.runtime import OperatorRecoveryConfig

compiled = program.compile(x, y, config=config)
with compiled.prepare(
    operator_recovery=OperatorRecoveryConfig(
        replay_safe=("exchange",), max_retries=1, timeout_s=30.0
    )
) as runtime:
    runtime(x, y)
```

`exchange` is the registered CHIP orchestration function name. It is the whole
L3 task that is replayed, including its in-core kernels. The declaration promises
that its inputs survive failure and that partially written outputs and effects
can safely be repeated. For example, reading immutable inputs into private,
fully overwritten output buffers can be replay-safe; an in-place accumulation or
an externally visible send generally needs a restore/commit protocol first.
Functions absent from `replay_safe` retain `UNKNOWN` safety and cannot be replayed.
An unknown name is an error. Safety is never inferred from a fault code, tensor
direction, or the presence of a dependency edge.

PyPTO derives run/operator identities, participants, and window/context argument
bindings from generated code. It captures rank-local dispatches before admitting
tasks, validates matching function sequences, and submits each ordinal as one
full-rank group. Each group depends on the preceding whole group as well as its
explicit task dependencies. Existing tensor dependencies remain in the original
native graph. This deliberately serializes whole operators in the recovery mode.
Native admission stays closed until the complete graph is submitted, so an early
fault cannot block the graph builder behind its own recovery gate. The graph must
fit the configured native task/heap capacity. Normal dispatch is unchanged when
the option is omitted. Capturing the host
callback adds build latency and retains argument metadata until the run ends.

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

`tests/st/distributed/test_l3_operator_recovery.py` compiles a real `@pl.jit.host`
program: warmup → exchange → finish. Its test-only external AIV kernel raises an
asynchronous completion error after the peer enters a notification wait. The peer
needs framework STOP; the next attempt uses fresh notification epochs. The tests
check SDMA data, final numerical output, execution counts and order. They do not
inject a physical network failure. No injection hook is added to production code.
