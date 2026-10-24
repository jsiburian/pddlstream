"""Options for solving a PDDLStream problem.

solve() accepts every SolverConfig field as a keyword argument, or a whole SolverConfig as config=...
"""

import dataclasses
import difflib
import math
from dataclasses import dataclass, field

INF = math.inf
FOCUSED_ALGORITHMS = ['focused', 'binding', 'adaptive']
ALGORITHMS = ['incremental'] + FOCUSED_ALGORITHMS
DEFAULT_ALGORITHM = 'adaptive'
STREAM_ERRORS = ['raise', 'fail']
# Fields that are passed on to FastDownward's search
SEARCH_FIELDS = ['planner', 'max_planner_time', 'hierarchy', 'temp_dir', 'clean', 'debug']


@dataclass(frozen=True)
class SolverConfig:
    """How to solve a problem. Change fields with dataclasses.replace(config, ...)."""

    algorithm: str = DEFAULT_ALGORITHM  # One of ALGORITHMS

    # Termination
    unit_costs: bool = False  # Use unit action costs rather than numeric costs
    success_cost: float = INF  # Stop once a plan costs strictly less than this
    max_time: float = INF  # Seconds
    max_iterations: float = INF
    max_memory: float = INF  # Kilobytes

    # Stream complexity
    initial_complexity: int = 0
    complexity_step: int = 1  # The increase of the complexity limit per iteration
    max_complexity: float = INF

    # Plan skeletons (adaptive) and stream failures (focused, binding)
    max_skeletons: float = INF  # None plans without skeletons
    search_sample_ratio: float = 1  # Desired ratio of sampling time to search time
    max_failures: float = 0  # Stream failures before switching phases (focused, binding)

    # Stream effort
    unit_efforts: bool = False  # Use unit stream efforts rather than estimated ones
    max_effort: float = INF
    effort_weight: float | None = None  # Multiplier for stream effort compared to action costs
    reorder: bool = True  # Reorder stream plans to reduce the expected sampling overhead

    # FastDownward search
    planner: str | None = None  # A key of SEARCH_OPTIONS; None uses DEFAULT_PLANNER
    max_planner_time: float | None = None  # Seconds per search; None uses DEFAULT_MAX_TIME
    hierarchy: list = field(default_factory=list)  # ABSTRIPS layers (incremental only)
    temp_dir: str | None = None  # Directory for FastDownward's files; None uses a private temporary directory
    clean: bool = False  # Delete temp_dir afterwards
    debug: bool = False  # Print FastDownward's output and keep its files in ./temp/

    # Output and reproducibility
    visualize: bool = False  # Draw the constraint network and stream plans with graphviz
    verbose: bool = True  # Print the result of each stream call
    deterministic: bool = False  # Count work units instead of seconds for time-based decisions (see README)

    # Failures (docs/v2/adr-001-stream-outcomes.md)
    # When a stream or predicate raises: 'raise' stops the solve, 'fail' treats the call as exhausted (a cost function
    # that raises always stops the solve, since planning needs its value)
    stream_errors: str = 'raise'

    # Stream statistics from earlier runs
    load_statistics: bool = True
    save_statistics: bool = True

    # Advanced
    fd_instantiate: bool = True  # Instantiate actions with FastDownward rather than pddlstream's own instantiator
    # The adaptive algorithm's skeleton queue: whether bindings are ordered by complexity, visits and remaining streams
    # (otherwise by visits), how many visits a binding is sampled greedily for, whether only a skeleton's most
    # advanced binding is, and whether a binding is sampled only while its downstream streams can still succeed
    skeleton_priorities: bool = True
    greedy_visits: int = 0
    greedy_best: bool = True
    require_downstream: bool = True
    exogenous_axioms: bool = True  # Compile exogenous streams into axioms rather than actions
    exogenous_future_values: bool = True  # Plan with placeholder values until an exogenous stream can be called

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f'Unknown algorithm {self.algorithm!r}; choose one of {ALGORITHMS}')
        if self.stream_errors not in STREAM_ERRORS:
            raise ValueError(f'Unknown stream_errors {self.stream_errors!r}; choose one of {STREAM_ERRORS}')

    @classmethod
    def from_kwargs(cls, base=None, caller='solve()', **kwargs):
        """base (by default SolverConfig()) with the given fields replaced; unknown names raise a TypeError."""
        check_keywords(kwargs, FIELD_NAMES, caller)
        return dataclasses.replace(base or cls(), **kwargs)

    def search_kwargs(self):
        """The search fields that differ from their defaults, as keyword arguments for the search subroutine."""
        return {name: getattr(self, name) for name in SEARCH_FIELDS if getattr(self, name) != DEFAULTS[name]}


FIELD_NAMES = [f.name for f in dataclasses.fields(SolverConfig)]
DEFAULTS = {name: getattr(SolverConfig(), name) for name in FIELD_NAMES}


def check_keywords(names, valid, caller):
    unknown = [name for name in names if name not in valid]
    if unknown:
        suggestions = {name: difflib.get_close_matches(name, valid, n=1) for name in unknown}
        hints = ''.join(
            f' (did you mean {match[0]!r} instead of {name!r}?)' for name, match in suggestions.items() if match
        )
        raise TypeError(f'{caller} got unexpected keyword arguments {unknown}{hints}')
