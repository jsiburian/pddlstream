"""Independent checks of solutions returned by the solvers."""

from pddlstream.algorithms.certificate import PlanCertificate
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.core.problem import Action, PDDLProblem, is_plan
from pddlstream.core.syntax import EQ, get_args, get_prefix
from pddlstream.core.utils import INF, value_or_id


def validate_solution(problem, solution, unit_costs=False, max_time=60, facts=None, **search_kwargs):
    """Checks that a solution's action plan is executable and achieves the goal.

    The plan is replayed by FastDownward on the problem without streams, from the initial state plus facts (by
    default every fact the streams certified, certificate.all_facts), constrained to exactly the plan's action
    sequence. This uses the planner's own semantics for derived predicates and conditional effects.

    Returns (valid, cost): valid is True or False, or None when the solution cannot be replayed (its plan contains
    deferred stream evaluations, or the solver did not record the certified facts); cost is the replayed plan's
    cost (INF unless valid).
    """
    plan, _, certificate = solution
    if not is_plan(plan):
        return False, INF
    facts = certificate.all_facts if facts is None else facts
    if not all(isinstance(action, Action) for action in plan) or not certificate.all_facts:
        return None, INF
    domain_pddl, constant_map, _, _, init, goal = problem
    replay_problem = PDDLProblem(domain_pddl, constant_map, None, {}, list(init) + list(facts), goal)
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


def fact_key(fact):
    """A hashable key for a fact with values: predicate names ignore case, and unhashable values compare by identity.
    A function value ('=', head, value) is keyed by its head's key."""
    if get_prefix(fact) == EQ:
        head, value = get_args(fact)
        return (EQ, fact_key(head), value_or_id(value))
    return (get_prefix(fact).lower(), *map(value_or_id, get_args(fact)))


def verify_certificate(problem, solution, unit_costs=False, max_time=60, **search_kwargs):
    """Checks a solution's certificate independently of the solver: each support's domain facts are initial or
    certified by an earlier support, every preimage fact is initial, internal or certified, and the plan reaches the
    goal from the initial facts plus only the certified and internal facts (and the cost function values its cost
    assumes, certificate.unevaluated, which concern only the cost).

    Returns (verified, issues): verified is True or False, or None when the solution has no PlanCertificate or its
    plan relies on facts that no stream certified yet (deferred streams); issues says what failed.
    """
    plan, _, certificate = solution
    if not is_plan(plan):
        return False, ['There is no plan']
    if not isinstance(certificate, PlanCertificate) or not certificate.complete:
        return None, ['The certificate does not support every preimage fact']
    known = {fact_key(fact) for fact in list(problem.init) + certificate.internal}
    issues, certified = [], []
    for support in certificate.supports:
        missing = [fact for fact in support.domain if fact_key(fact) not in known]
        if missing:
            issues.append(f'{support.stream}{support.inputs} needs {missing}, which no earlier support certifies')
        known.update(map(fact_key, support.certified))
        certified.extend(support.certified)
    unevaluated = {fact_key(fact) for fact in certificate.unevaluated}  # Cost function values the cost assumes
    for fact in certificate.preimage_facts:
        if (fact_key(fact) not in known) and (fact_key(fact) not in unevaluated):
            issues.append(f'Preimage fact {fact} is neither initial nor certified by a support')
    if issues:
        return False, issues
    facts = certified + certificate.internal + certificate.unevaluated
    valid, _ = validate_solution(problem, solution, unit_costs, max_time, facts=facts, **search_kwargs)
    return valid, [] if valid else ['The plan does not reach the goal from the initial and certified facts']
