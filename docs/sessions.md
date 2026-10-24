# Replanning with planner sessions

A `PlannerSession` plans repeatedly as the world changes. It keeps the stream results of earlier solves and reuses
those that a change leaves valid, so a robot that observes a change replans without sampling everything again.

```python
from pddlstream import PlannerSession, StreamInfo, WorldPatch

stream_info = {
    'sample-grasp': StreamInfo(reads=()),  # Depends only on its inputs
    'plan-motion': StreamInfo(reads=('obstacles',)),  # Also depends on obstacles that are not facts
}
with PlannerSession(problem, stream_info=stream_info, algorithm='adaptive') as session:
    result = session.solve()
    # The robot observes that a block moved and that an obstacle appeared
    session.apply(WorldPatch(add=[('AtPose', 'b', new_pose)], remove=[('AtPose', 'b', old_pose)]))
    session.apply(WorldPatch(changed=['obstacles']))
    result = session.solve()
    print(session.report)  # Per stream: results reused and lost, instances invalidated, calls replayed and fresh
```

`examples/continuous_tamp/replan.py` runs this end to end.

## What is reused

- **Results.** Each solve starts from the current initial facts plus the certified facts of earlier results whose
  domain facts still hold. A result whose domain no longer holds (for example, one that needed a removed fact) is
  *lost*, and so are the results that depended on it. A fact certified by two results survives the loss of either.
- **Histories.** Instances replay their earlier outputs before calling their procedures again, so a sampler continues
  where it left off instead of repeating itself.

## What invalidates results

A stream declares what its procedure depends on besides its inputs with `StreamInfo(reads=...)`:

| `reads` | Meaning | Invalidated by |
|---|---|---|
| `None` (the default) | Unknown | Any change |
| `()` | Only its inputs | Nothing; its results are lost only when their domain facts no longer hold |
| `('obstacles',)` | Its inputs and the named things | Changes that name them (`WorldPatch(changed=...)`), or that add or remove facts of those predicates |

`session.invalidate(stream)` drops a stream's results explicitly, for example after its procedure changed. An instance
whose procedure raised (with `stream_errors='fail'`) is retried by the next solve.

## Notes

- Values are matched by equality, and unhashable values (such as numpy arrays) by identity: a fact that a patch removes
  must use the same objects as the fact that was added.
- The session keeps its own objects, so sessions in the same process (or thread) do not interfere.
- Stream statistics are loaded when the session starts and saved by `close()`.
- Exogenous streams and optimizers are not supported in sessions yet.
