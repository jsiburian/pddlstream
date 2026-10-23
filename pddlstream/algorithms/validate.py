"""Independent checks of solutions returned by the solvers."""

from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.core.problem import Action, PDDLProblem, is_plan
from pddlstream.core.utils import INF


def validate_solution(problem, solution, unit_costs=False, max_time=60, **search_kwargs):
    """Checks that a solution's action plan is executable and achieves the goal.

    The plan is replayed by FastDownward on the problem without streams, from the initial state plus every fact
    the streams certified (certificate.all_facts), constrained to exactly the plan's action sequence. This uses the
    planner's own semantics for derived predicates and conditional effects.

    Returns (valid, cost): valid is True or False, or None when the solution cannot be replayed (its plan contains
    deferred stream evaluations, or the solver did not record the certified facts); cost is the replayed plan's
    cost (INF unless valid).
    """
    plan, _, certificate = solution
    if not is_plan(plan):
        return False, INF
    all_facts = certificate.all_facts
    if not all(isinstance(action, Action) for action in plan) or not all_facts:
        return None, INF
    domain_pddl, constant_map, _, _, init, goal = problem
    replay_problem = PDDLProblem(domain_pddl, constant_map, None, {}, list(init) + list(all_facts), goal)
    skeleton = [(action.name, list(action.args)) for action in plan]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    replay, cost, _ = solve_incremental(
        replay_problem,
        constraints=constraints,
        unit_costs=unit_costs,
        max_time=max_time,
        verbose=False,
        **search_kwargs,
    )
    return is_plan(replay), cost
