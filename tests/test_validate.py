"""The plan validator must accept the solvers' plans and reject corrupted ones."""
import pytest

from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.validate import validate_solution
from pddlstream.language.constants import And, Certificate, PDDLProblem, Solution, StreamAction
from pddlstream.utils import INF

from examples.blocksworld.run import get_problem as get_blocksworld


@pytest.fixture(scope='module')
def blocksworld(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path_factory.mktemp('blocksworld'))
        problem = get_blocksworld()
        return problem, solve(problem, unit_costs=True)


@pytest.fixture(scope='module')
def continuous_tamp(tmp_path_factory):
    from examples.continuous_tamp.primitives import tight
    from examples.continuous_tamp.run import get_stream_info, pddlstream_from_tamp, set_deterministic
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path_factory.mktemp('continuous_tamp'))
        set_deterministic()
        problem = pddlstream_from_tamp(tight(n_blocks=2))
        return problem, solve(problem, stream_info=get_stream_info(), planner='max-astar', unit_costs=True,
                              max_time=120)


def replace_plan(solution, plan):
    return Solution(plan, solution.cost, solution.certificate)


def test_accepts_solution(blocksworld):
    problem, solution = blocksworld
    assert validate_solution(problem, solution, unit_costs=True) == (True, 4)


@pytest.mark.parametrize('corrupt', [
    lambda plan: plan[:-1],                             # Doesn't reach the goal
    lambda plan: [plan[1], plan[0]] + plan[2:],         # Puts down a block before unstacking it
    lambda plan: [plan[0]._replace(args=plan[0].args[::-1])] + plan[1:],  # Unstacks the wrong block
], ids=['truncated', 'swapped', 'wrong-args'])
def test_rejects_corrupted_plan(blocksworld, corrupt):
    problem, solution = blocksworld
    assert validate_solution(problem, replace_plan(solution, corrupt(list(solution.plan)))) == (False, INF)


def test_no_plan_is_invalid(blocksworld):
    problem, _ = blocksworld
    assert validate_solution(problem, Solution(None, INF, Certificate([], []))) == (False, INF)


def test_deferred_streams_cannot_be_replayed(blocksworld):
    problem, solution = blocksworld
    plan = [StreamAction('sample', ('a',), ('?x',))] + list(solution.plan)
    assert validate_solution(problem, replace_plan(solution, plan)) == (None, INF)


def test_accepts_continuous_solution(continuous_tamp):
    # Plan arguments are numpy arrays, which the skeleton constraints must treat as constants
    problem, solution = continuous_tamp
    valid, cost = validate_solution(problem, solution, unit_costs=True)
    assert valid and (cost == len(solution.plan))


def test_rejects_uncertified_facts(continuous_tamp):
    problem, solution = continuous_tamp
    all_facts = [fact for fact in solution.certificate.all_facts if fact[0].lower() != 'motion']
    uncertified = Solution(solution.plan, solution.cost, Certificate(all_facts, None))
    assert validate_solution(problem, uncertified, unit_costs=True) == (False, INF)


def test_constraints_avoid_domain_predicate_names():
    # The skeleton compilation adds `order`/`assigned`/... predicates, which must not clash with the domain's
    domain_pddl = """(define (domain clash)
      (:predicates (order ?x) (assigned ?x) (done))
      (:action finish :parameters (?x) :precondition (order ?x) :effect (and (done) (assigned ?x))))"""
    problem = PDDLProblem(domain_pddl, {}, None, {}, [('order', 'a'), ('order', 'b')], And(('done',)))
    plan, _, _ = solve(problem, algorithm='incremental', unit_costs=True,
                       constraints=PlanConstraints(skeletons=[[('finish', ['b'])]], exact=True))
    assert [(action.name, action.args) for action in plan] == [('finish', ('b',))]
