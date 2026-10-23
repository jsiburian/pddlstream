import logging
import time
from collections import namedtuple

from pddlstream.core.context import get_clock
from pddlstream.core.conversion import revert_solution

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
from pddlstream.core.problem import FAILED, SolveResult, Status, get_length, is_plan
from pddlstream.core.utils import INF, check_memory, elapsed_time, str_from_object
from pddlstream.language.statistics import write_stream_statistics

logger = logging.getLogger(__name__)

Solution = namedtuple('Solution', ['plan', 'cost', 'time'])

SOLUTIONS = []  # TODO: remove global variable


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
        # TODO: return INFEASIBLE if can prove no solution
        return self.solutions[-1].plan if self.solutions else FAILED

    @property
    def best_cost(self):
        return self.solutions[-1].cost if self.solutions else INF

    def add_plan(self, plan, cost):
        # TODO: double-check that plan is a solution
        if is_plan(plan) and (cost < self.best_cost):
            self.solutions.append(Solution(plan, cost, elapsed_time(self.start_time)))

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
        """The instance's next results and new facts, with the time charged to sampling."""
        start_time = get_clock().now()
        new_results, new_facts = instance.next_results(verbose=verbose)
        self.sample_time += get_clock().now() - start_time
        return new_results, new_facts

    def finish(self, externals=(), infeasible=False, **stats):
        """Logs the summary (with stats added), saves the externals' statistics and returns the SolveResult."""
        summary = self.export_summary()
        summary.update(stats)
        logger.info(f'Summary: {str_from_object(summary, ndigits=3)}')
        write_stream_statistics(externals, self.verbose)
        return self.extract_solution(summary, infeasible)

    def extract_solution(self, stats=None, infeasible=False):
        """The best plan as a SolveResult; infeasible says whether the solver concluded that no plan exists."""
        SOLUTIONS[:] = self.solutions
        plan, cost, certificate = revert_solution(self.best_plan, self.best_cost, self.evaluations)
        status = Status.SUCCEEDED if is_plan(plan) else Status.INFEASIBLE if infeasible else Status.FAILED
        return SolveResult(plan, cost, certificate, status, self.export_summary() if stats is None else stats)

    def export_summary(self):  # TODO: log, etc...
        # TODO: SOLUTIONS
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
