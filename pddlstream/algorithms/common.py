import logging
import time
from collections import Counter, namedtuple

from pddlstream.algorithms.certificate import certify
from pddlstream.core.context import get_clock, get_config, get_record
from pddlstream.core.conversion import param_from_object, revert_solution, transform_plan_args

# Moved to pddlstream.core.evaluations; re-exported for existing code
from pddlstream.core.evaluations import COMPLEXITY_OP as COMPLEXITY_OP
from pddlstream.core.evaluations import INIT_EVALUATION as INIT_EVALUATION
from pddlstream.core.evaluations import INTERNAL_EVALUATION as INTERNAL_EVALUATION
from pddlstream.core.evaluations import UNKNOWN_EVALUATION as UNKNOWN_EVALUATION
from pddlstream.core.evaluations import EvaluationNode as EvaluationNode
from pddlstream.core.evaluations import add_certified as add_certified
from pddlstream.core.evaluations import add_fact as add_fact
from pddlstream.core.evaluations import add_facts as add_facts
from pddlstream.core.evaluations import compute_complexity as compute_complexity
from pddlstream.core.evaluations import evaluations_from_init as evaluations_from_init
from pddlstream.core.evaluations import is_instance_ready as is_instance_ready
from pddlstream.core.problem import FAILED, Incumbent, OptPlan, SolveResult, Status, get_length, is_plan
from pddlstream.core.trace import emit, event_to_dict
from pddlstream.core.utils import INF, check_memory, elapsed_time, str_from_object
from pddlstream.language.external import Call, Outcome
from pddlstream.language.statistics import write_stream_statistics

logger = logging.getLogger(__name__)

Solution = namedtuple('Solution', ['plan', 'cost', 'time'])

SOLUTIONS = []  # Deprecated: the last solve's solutions, now SolveResult.solutions


def get_action_plan(plan):
    return plan.action_plan if isinstance(plan, OptPlan) else plan


def to_incumbent(solution):
    """A solution (plan, cost, time) with values instead of objects."""
    return Incumbent(
        transform_plan_args(get_action_plan(solution.plan), param_from_object), solution.cost, solution.time
    )


class SolutionStore:
    def __init__(self, evaluations, max_time, success_cost, verbose, max_memory=INF):
        # TODO: store a map from head to value?
        # TODO: include other problem information here?
        # TODO: determine when the plan converges
        self.evaluations = evaluations
        self.start_time = time.time()
        self.clock_start = get_clock().now()  # Search and sample times are measured with the solve's clock
        self.max_time = max_time
        self.max_memory = max_memory
        self.success_cost = success_cost  # Inclusive
        self.verbose = verbose
        self.solutions = []
        self.sample_time = 0.0

    @property
    def search_time(self):
        return (get_clock().now() - self.clock_start) - self.sample_time

    @property
    def best_plan(self):
        return self.solutions[-1].plan if self.solutions else FAILED

    @property
    def best_cost(self):
        return self.solutions[-1].cost if self.solutions else INF

    def add_plan(self, plan, cost):
        # TODO: double-check that plan is a solution
        if is_plan(plan) and (cost < self.best_cost):
            self.solutions.append(Solution(plan, cost, elapsed_time(self.start_time)))
            emit(to_incumbent(self.solutions[-1]))

    def has_solution(self):
        return is_plan(self.best_plan)

    def is_solved(self):
        return self.has_solution() and (self.best_cost <= self.success_cost)

    def elapsed_time(self):
        return elapsed_time(self.start_time)

    def is_timeout(self):
        return (self.max_time <= self.elapsed_time()) or not check_memory(self.max_memory)

    def is_terminated(self):
        return self.is_solved() or self.is_timeout()

    def sample(self, instance, verbose=False):
        """The instance's next results and new facts, with the time charged to sampling and the call recorded."""
        start_time = get_clock().now()
        call_index, num_history, error = instance.num_calls, len(instance.history), instance.error
        new_results, new_facts = instance.next_results(verbose=verbose)
        self.sample_time += get_clock().now() - start_time
        if instance.error is not error:
            outcome = Outcome.ERROR
        elif new_results or new_facts:
            outcome = Outcome.YIELDED
        else:
            outcome = Outcome.EXHAUSTED if instance.enumerated else Outcome.NO_SAMPLE_YET
        fresh = (num_history < len(instance.history)) or (outcome is Outcome.ERROR)
        call = Call(instance, call_index, outcome, new_results, new_facts, fresh)
        get_record().calls.append(call)
        emit(call)
        return new_results, new_facts

    def finish(self, externals=(), infeasible=False, **stats):
        """Logs the summary (with stats added), saves the externals' statistics and returns the SolveResult.
        infeasible says whether the solver ran out of things to try; the solve is reported infeasible only if nothing
        was cut short (docs/v2/adr-001-stream-outcomes.md)."""
        record = get_record()
        gaps = set(record.gaps)
        if self.max_time <= self.elapsed_time():
            gaps.add('max_time')
        if not check_memory(self.max_memory):
            gaps.add('max_memory')
        if any(call.outcome is Outcome.ERROR for call in record.calls):
            gaps.add('stream_errors')
        if get_config().strict and any(
            call.instance.enumerated and not call.instance.external.is_complete for call in record.calls
        ):
            gaps.add('incomplete_streams')
        summary = self.export_summary()
        summary.update(stats)
        summary.update(
            outcomes=dict(Counter(call.outcome.value for call in record.calls)),
            termination=self.get_termination(infeasible, gaps, stats),
            gaps=sorted(gaps),
        )
        logger.info(f'Summary: {str_from_object(summary, ndigits=3)}')
        write_stream_statistics(externals, self.verbose)
        return self.extract_solution(summary, infeasible and not gaps)

    def get_termination(self, infeasible, gaps, stats):
        """Why the solver stopped."""
        config = get_config()
        if self.is_solved():
            return 'solved'
        for limit in ['max_time', 'max_memory']:  # Running out of time also cuts the search short
            if limit in gaps:
                return limit
        if infeasible:
            return 'exhausted'
        if stats.get('iterations', 0) >= config.max_iterations:
            return 'max_iterations'
        if stats.get('complexity', 0) > config.max_complexity:
            return 'max_complexity'
        return 'stopped'

    def extract_solution(self, stats=None, infeasible=False):
        """The best plan as a SolveResult; infeasible says whether the solver concluded that no plan exists."""
        SOLUTIONS[:] = self.solutions
        plan, cost, certificate = revert_solution(self.best_plan, self.best_cost, self.evaluations)
        if isinstance(self.best_plan, OptPlan):
            certificate = certify(certificate, self.evaluations, self.best_plan.preimage_facts)
        status = Status.SUCCEEDED if is_plan(plan) else Status.INFEASIBLE if infeasible else Status.FAILED
        solutions = list(map(to_incumbent, self.solutions))
        stats = self.export_summary() if stats is None else stats
        result = SolveResult(plan, cost, certificate, status, stats, solutions)
        if get_config().trace:
            result.trace = list(map(event_to_dict, get_record().events))
        return result

    def export_summary(self):
        return {
            'solved': self.is_solved(),
            'solutions': len(self.solutions),
            'cost': self.best_cost,
            'length': get_length(self.best_plan),
            'evaluations': len(self.evaluations),
            'search_time': self.search_time,
            'sample_time': self.sample_time,
            'run_time': self.elapsed_time(),
            'timeout': self.is_timeout(),
        }


##################################################


##################################################


def stream_plan_preimage(stream_plan):
    # Easy because monotonic
    preimage = set()
    achieved = set()
    for stream in stream_plan:
        preimage.update(set(stream.get_domain()) - achieved)
        achieved.update(stream.get_certified())
    return preimage
