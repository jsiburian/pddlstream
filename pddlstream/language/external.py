import enum
import logging
from collections import Counter
from typing import NamedTuple

from pddlstream.core.context import CALL_WORK, get_clock, get_config
from pddlstream.core.conversion import substitute_fact, values_from_objects
from pddlstream.core.evaluations import compute_complexity
from pddlstream.core.objects import Object, OptimisticObject
from pddlstream.core.syntax import Fact, get_args, get_prefix, is_parameter
from pddlstream.core.utils import INF, CreationHash, Score, get_mapping, safe_apply_mapping, str_from_object
from pddlstream.language.executor import REPLAY, SerialExecutor
from pddlstream.language.statistics import DEFAULT_SEARCH_OVERHEAD, Performance, PerformanceInfo

logger = logging.getLogger(__name__)

# As a PDDLProblem's stream_map, these replace every stream's procedure with placeholder outputs, to test the domain
# before writing the samplers: DEBUG gives every call new placeholders, SHARED_DEBUG one set per stream
DEBUG = 'debug'
SHARED_DEBUG = 'shared_debug'
DEBUG_MODES = [DEBUG, SHARED_DEBUG]


# A StreamInfo's defer_fn decides whether a stream call can wait until the plan reaches the step that needs it


def never_defer(*args, **kwargs):
    """Calls the stream before executing the plan (the default)."""
    return False


def defer_unique(result, *args, **kwargs):
    """Defers the call if its optimistic outputs are unique to it."""
    return result.is_refined()


def defer_shared(*args, **kwargs):
    """Always defers the call."""
    return True


def select_inputs(instance, inputs):
    external = instance.external
    assert set(inputs) <= set(external.inputs)
    mapping = get_mapping(external.inputs, instance.input_objects)
    return safe_apply_mapping(inputs, mapping)


def get_defer_any_unbound(unique=False):
    """A defer_fn that defers a call while any of its inputs is still unbound (with unique, also defer_unique)."""

    def defer_any_unbound(result, bound_objects=set(), *args, **kwargs):
        # The set bound_objects may contain shared objects in which case replanning is required
        if unique and not defer_unique(result):
            return False
        return not all(isinstance(obj, Object) or (obj in bound_objects) for obj in result.input_objects)

    return defer_any_unbound


def get_defer_all_unbound(inputs='', unique=False):  # TODO: shortcut for all inputs
    """A defer_fn that defers a call while all of the given inputs (space-separated) are unbound (with unique, also
    defer_unique)."""
    inputs = tuple(inputs.split())
    # Empty implies defer_shared

    def defer_all_unbound(result, bound_objects=set(), *args, **kwargs):
        if unique and not defer_unique(result):
            return False
        return not any(
            isinstance(obj, Object) or (obj in bound_objects) for obj in select_inputs(result.instance, inputs)
        )

    return defer_all_unbound


def get_domain_predicates(streams):
    return {get_prefix(a) for s in streams for a in s.domain}


def convert_constants(fact):
    # TODO: take the constant map as an input
    # TODO: throw an error if undefined
    return Fact(get_prefix(fact), [p if is_parameter(p) else Object.from_name(p) for p in get_args(fact)])


##################################################


class ExternalInfo(PerformanceInfo):
    def __init__(
        self,
        eager=False,
        eager_skeleton=False,
        defer_fn=never_defer,
        complete=False,
        reads=None,
        resource=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Whether the procedure's outputs are exactly the values that satisfy the certified facts, so running out of
        # outputs proves that no others exist (used by strict=True; see docs/v2/adr-001-stream-outcomes.md)
        self.complete = complete
        # What the procedure's results depend on besides its inputs, for a PlannerSession to know which results a change
        # to the world invalidates: None (unknown: any change), () (nothing: only the inputs), or names of the things
        # it reads (e.g. ('walls',)), which may be predicates of facts that change
        self.reads = reads
        # Where a parallel executor runs the procedure: a named pool, 'main' (the solver's thread) or None (the default
        # pool); see language/executor.py
        self.resource = resource
        # TODO: enable eager=True for inexpensive test streams by default
        # TODO: change p_success and overhead if it's a function or test stream
        self.eager = eager
        self.eager_skeleton = eager_skeleton  # TODO: apply in binding and adaptive
        # TODO: automatically set tests and costs to be eager
        self.defer_fn = defer_fn  # Old syntax was defer=True


##################################################


class Result(CreationHash):
    def __init__(self, instance, opt_index, call_index, optimistic):
        self.instance = instance
        self.opt_index = opt_index
        self.call_index = call_index
        self.optimistic = optimistic

    @property
    def external(self):
        return self.instance.external

    @property
    def info(self):
        return self.external.info

    @property
    def name(self):
        return self.external.name

    @property
    def input_objects(self):
        return self.instance.input_objects

    @property
    def domain(self):
        return self.instance.domain

    def is_refined(self):
        # TODO: result.opt_index is None
        return self.opt_index == 0  # TODO: base on output objects instead

    def is_deferrable(self, *args, **kwargs):
        return self.info.defer_fn(self, *args, **kwargs)

    def get_domain(self):
        return self.instance.get_domain()

    def get_certified(self):
        raise NotImplementedError()

    def get_components(self):
        return [self]

    def get_unsatisfiable(self):
        return [self.get_components()]

    def get_action(self):
        raise NotImplementedError()

    def remap_inputs(self, bindings):
        raise NotImplementedError()

    def is_successful(self):
        raise NotImplementedError()

    def compute_complexity(self, evaluations, **kwargs):
        # Should be constant
        return compute_complexity(evaluations, self.get_domain(), **kwargs) + self.external.get_complexity(
            self.call_index
        )

    def get_effort(self, **kwargs):
        if not self.optimistic:
            return 0  # Unit efforts?
        if self.external.is_negated:
            return 0
        # TODO: this should be the min of all instances
        return self.instance.get_effort(**kwargs)

    def success_heuristic(self):  # High is likely to succeed
        # self.external.is_function
        num_free = sum(isinstance(obj, OptimisticObject) for obj in self.input_objects)
        return Score(num_free, -len(self.external.inputs))  # TODO: treat objects in the same domain as a unit

    def overhead_heuristic(self):  # Low is cheap
        return self.external.overhead_heuristic()

    def stats_heuristic(self):  # Low is cheap and unlikely to succeed
        return Score(self.overhead_heuristic(), self.success_heuristic())

    def effort_heuristic(self):  # Low is cheap and likely to succeed
        return Score(self.overhead_heuristic(), -self.success_heuristic())


##################################################


class Outcome(enum.Enum):
    """What one stream or function call did (docs/v2/adr-001-stream-outcomes.md)."""

    YIELDED = 'yielded'  # Produced at least one new output tuple or fact
    NO_SAMPLE_YET = 'no_sample_yet'  # Produced nothing new; later calls may
    EXHAUSTED = 'exhausted'  # Produced nothing new, and the procedure has no further outputs for these inputs
    ERROR = 'error'  # The procedure raised an exception (with stream_errors='fail')


class Call(NamedTuple):
    """One call of an instance during a solve."""

    instance: 'Instance'
    call_index: int  # The instance's number of calls before this one
    outcome: Outcome
    results: list  # The new results
    facts: list  # The new facts beyond the results' certified facts (wild streams)
    fresh: bool  # Whether the procedure ran, rather than the instance replaying its history
    seconds: float = 0.0  # How long the procedure ran


SERIAL_EXECUTOR = SerialExecutor()


def get_executor():
    """The solve's executor (SolverConfig.executor), by default serial."""
    return get_config().executor or SERIAL_EXECUTOR


##################################################


class Instance(CreationHash):
    _Result: type | None = None  # Set by subclasses

    def __init__(self, external, input_objects):
        self.external = external
        self.input_objects = tuple(input_objects)
        self.disabled = False  # TODO: perform disabled using complexity
        self.history = []  # TODO: facts history
        self.results_history = []
        self.error = None  # The exception the procedure raised, with stream_errors='fail'
        self._mapping = None
        self._domain = None
        self._prefetched = None  # The next call's procedure run, started ahead by a parallel executor
        self.reset()

    @property
    def info(self):
        return self.external.info

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = get_mapping(self.external.inputs, self.input_objects)
        return self._mapping

    @property
    def domain(self):
        if self._domain is None:
            self._domain = tuple(substitute_fact(atom, self.mapping) for atom in self.external.domain)
        return self._domain

    def get_iteration(self):
        return INF if self.enumerated else self.num_calls

    def get_domain(self):
        return self.domain

    def get_all_input_objects(self):
        return set(self.input_objects)

    def get_input_values(self):
        return values_from_objects(self.input_objects)

    def fail(self, error, recoverable=True):
        """Applies the solve's stream_errors policy to an exception that the procedure raised: re-raises it (naming
        the call), or records it and marks the instance exhausted."""
        error.add_note(f'Raised by {self.external.name}{str_from_object(self.get_input_values())}')
        if (get_config().stream_errors == 'raise') or not recoverable:
            raise error
        logger.warning(f'{self.external.name} raised {error!r}; treating it as exhausted')
        self.error = error
        self.enumerated = True

    def reset(self):
        self.disabled = False
        self.opt_index = self.external.num_opt_fns
        self.num_calls = 0
        self.enumerated = False
        self.successful = False

    def is_refined(self):
        return self.opt_index == 0

    def refine(self):
        # TODO: could instead create a new Instance per opt_index
        if not self.is_refined():
            self.opt_index -= 1
        return self.opt_index

    def will_replay(self):
        """Whether the next call replays the instance's history rather than running the procedure."""
        return self.num_calls < len(self.history)

    def run_procedure(self):
        """Runs the procedure for the next call and returns a PendingCall. It changes only the instance's own
        generator, so an executor may run it on another thread; finish_call commits its outcome."""
        raise NotImplementedError()

    def procedure_request(self, key):
        """What a ProcessExecutor's worker needs to run the procedure for the next call."""
        return key, self.external.name, self.get_input_values(), None, 'function'

    def batch_procedure(self):
        """The BatchProcedure that can run the next call together with other instances' calls, if any."""
        return None

    def runs_inline(self):
        """Whether the procedure must run on the solver's thread."""
        return (self.info.resource == 'main') or getattr(self.external, 'inline', False)

    def can_prefetch(self):
        """Whether the next call can start ahead: it runs the procedure, which is not running already."""
        return (self._prefetched is None) and not (self.enumerated or self.will_replay())

    def set_prefetched(self, handle):
        self._prefetched = handle

    def begin_call(self):
        """The PendingCall of the next call: the prefetched run, a new run by the solve's executor, or REPLAY."""
        assert not self.enumerated
        if self.will_replay():
            return REPLAY
        handle, self._prefetched = self._prefetched, None
        if handle is None:
            handle = get_executor().submit(self)
        return handle.result()

    def finish_call(self, pending, verbose=False):
        """Commits a call: its new results and new facts."""
        raise NotImplementedError()

    def next_results(self, verbose=False):
        return self.finish_call(self.begin_call(), verbose=verbose)

    def first_results(self, num=1, **kwargs):
        results = []
        index = 0
        while len(results) < num:
            while index >= len(self.results_history):
                if self.enumerated:
                    return results
                self.next_results(**kwargs)
            results.extend(self.results_history[index])
            index += 1
        return results

    def all_results(self, **kwargs):
        return self.first_results(num=INF, **kwargs)

    def get_results(self, start=0):
        results = []
        for index in range(start, self.num_calls):
            results.extend(self.results_history[index])
        return results

    def compute_complexity(self, evaluations, **kwargs):
        # Will change as self.num_calls increases
        return compute_complexity(evaluations, self.get_domain(), **kwargs) + self.external.get_complexity(
            self.num_calls
        )

    def get_effort(self, search_overhead=DEFAULT_SEARCH_OVERHEAD):
        # TODO: handle case where resampled several times before the next search (search every ith time)
        replan_effort = self.opt_index * search_overhead  # By linearity of expectation
        effort_fn = self.external.info.effort
        if callable(effort_fn):
            return replan_effort + effort_fn(*self.get_input_values())
        return replan_effort + self.external.get_effort(search_overhead=search_overhead)

    def update_statistics(self, start_time, results):
        # start_time is from get_clock().now(), taken before the call
        get_clock().charge(CALL_WORK)
        overhead = get_clock().now() - start_time
        successes = sum(r.is_successful() for r in results)
        self.external.update_statistics(overhead, bool(successes))
        self.results_history.append(results)

    def disable(self, evaluations, domain):
        self.disabled = True

    def enable(self, evaluations, domain):
        self.disabled = False


##################################################


class External(Performance):
    _Instance: type | None = None  # Set by subclasses

    def __init__(self, name, info, inputs, domain):
        super().__init__(name, info)
        self.inputs = tuple(inputs)
        self.domain = tuple(map(convert_constants, domain))
        for p, c in Counter(self.inputs).items():
            if not is_parameter(p):
                # AssertionError: Expected item to be a variable: q2 in (?q1 q2)
                raise ValueError(f'Input [{p}] for stream [{name}] is not a parameter')
            if c != 1:
                raise ValueError(f'Input [{p}] for stream [{name}] is not unique')
        parameters = {a for i in self.domain for a in get_args(i) if is_parameter(a)}
        for p in parameters - set(self.inputs):
            raise ValueError(f'Parameter [{p}] for stream [{name}] is not included within inputs')
        for p in set(self.inputs) - parameters:
            logger.warning(f'Input [{p}] for stream [{name}] is not covered by a domain condition')
        self.constants = {a for i in self.domain for a in get_args(i) if not is_parameter(a)}
        self.instances = {}

    def reset(self, *args, **kwargs):
        for instance in self.instances.values():
            instance.reset(*args, **kwargs)

    # TODO: naming convention for statics and fluents
    @property
    def has_outputs(self):
        raise NotImplementedError()

    @property
    def is_fluent(self):
        raise NotImplementedError()

    @property
    def is_negated(self):
        raise NotImplementedError()

    @property
    def is_special(self):
        return self.is_fluent or self.is_negated

    @property
    def is_complete(self):
        return self.info.complete

    @property
    def is_function(self):
        raise NotImplementedError()

    @property
    def is_cost(self):
        return False

    @property
    def zero_complexity(self):
        return self.is_special or not self.has_outputs

    def get_complexity(self, num_calls=0):
        if self.zero_complexity:
            return 0
        return num_calls + 1

    def get_instance(self, input_objects):
        input_objects = tuple(input_objects)
        assert len(input_objects) == len(self.inputs)
        if input_objects not in self.instances:
            self.instances[input_objects] = self._Instance(self, input_objects)
        return self.instances[input_objects]

    def overhead_heuristic(self):  # Low is little overhead
        # TODO: infer other properties from use in the context of a stream plan
        # TODO: use num_certified (only those that are in another stream) instead of num_outputs?
        return Score(
            self.is_fluent, not self.is_function, self.has_outputs, len(self.inputs)
        )  # structural/relational overhead


##################################################


def get_procedure_fn(stream_map, name):
    if not isinstance(stream_map, dict):  # DEBUG_MODES
        return stream_map
    if name not in stream_map:
        raise ValueError(f'Undefined external procedure: {name}')
    return stream_map[name]


def is_attribute(attribute):
    return isinstance(attribute, str) and attribute.startswith(':')


def parse_lisp_list(lisp_list):
    attributes = [lisp_list[i] for i in range(0, len(lisp_list), 2)]
    for attribute in attributes:
        if not is_attribute(attribute):
            raise ValueError(f'Expected an attribute but got: {attribute}')
    values = [lisp_list[i] for i in range(1, len(lisp_list), 2)]
    if len(lisp_list) % 2 != 0:
        raise ValueError(f'No value specified for attribute [{lisp_list[-1]}]')
    return get_mapping(attributes, values)
