# Agent tools

`pddlstream.tools` gives agents (coding agents, LLM tool use, scripts) a small set of tools over problems that the host
program registers. Each tool takes and returns JSON-compatible data. No tool accepts facts, initial states or
certificates: an agent can propose a plan or a skeleton, and pddlstream checks it, but an agent never asserts what holds.

```python
from pddlstream.tools import Toolbox, Workspace

workspace = Workspace()
workspace.add_problem('kitchen', problem, stream_info, unit_costs=True)  # The host registers problems and procedures
toolbox = Toolbox(workspace)
toolbox.call('run_search', {'problem_id': 'kitchen', 'budget': {'max_time': 30}})
# {'ok': True, 'result': {'run_id': 'run0', 'status': 'succeeded', 'plan': [...], 'cost': ..., 'stats': {...}}}
```

| Tool | What it does |
|---|---|
| `describe_problem` | Predicates, actions, streams (with source lines), capabilities and lint issues |
| `list_stream_contracts` | Each stream's inputs, outputs, domain and certified facts, options and statistics |
| `check_stream` | Calls a procedure on input values outside any search and reports contract violations (wrong arity, exceptions, duplicate outputs) |
| `run_search` | Searches within a budget (`max_time`, `max_iterations`, `max_complexity`, `success_cost`, `algorithm`, `deterministic`) |
| `propose_hint` | Asks the next search to first try a plan skeleton; it only orders the search |
| `validate_plan` | Verifies a run's certificate, or replays a plan the agent wrote and reports the first failing step and missing facts |
| `explain_failure`, `inspect_trace`, `profile` | Why a run ended as it did, its events, and where its time went |
| `run_regressions` | Runs a search on several problems |

Each problem has a `PlannerSession`, so later searches reuse stream results. `examples/agent/scripted_agent.py` shows
an agent finding a broken sampler with `check_stream`, the host installing a fixed one, and the agent planning and
validating.

`pddlstream.tools.mcp_server.serve(workspace)` serves the tools over the Model Context Protocol; it needs the optional
`mcp` package (`pip install 'pddlstream[agent]'`).
