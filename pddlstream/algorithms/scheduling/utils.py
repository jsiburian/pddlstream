from pddlstream.core.conversion import evaluation_from_fact
from pddlstream.core.syntax import And, Not
from pddlstream.core.utils import apply_mapping
from pddlstream.fastdownward.tasks import add_predicate, fact_from_fd, get_literals, make_predicate
from pddlstream.fastdownward.translator import pddl
from pddlstream.language.function import FunctionResult
from pddlstream.language.optimizer import UNSATISFIABLE

# Moved to pddlstream.language.plans; re-exported for existing code
from pddlstream.language.plans import partition_external_plan as partition_external_plan


def partition_results(evaluations, results, apply_now):
    applied_results = []
    deferred_results = []
    opt_evaluations = set(evaluations)
    for result in results:
        assert not result.instance.disabled
        assert not result.instance.enumerated
        domain = set(map(evaluation_from_fact, result.instance.get_domain()))
        if isinstance(result, FunctionResult) or (apply_now(result) and (domain <= opt_evaluations)):
            applied_results.append(result)
            opt_evaluations.update(map(evaluation_from_fact, result.get_certified()))
        else:
            deferred_results.append(result)
    return applied_results, deferred_results


def add_unsatisfiable_to_goal(domain, goal_expression):
    add_predicate(domain, make_predicate(UNSATISFIABLE, []))
    negated_atom = pddl.NegatedAtom(UNSATISFIABLE, tuple())
    for action in domain.actions:
        if negated_atom not in action.precondition.parts:
            action.precondition = pddl.Conjunction([action.precondition, negated_atom]).simplified()
    return And(goal_expression, Not((UNSATISFIABLE,)))


def get_instance_facts(instance, node_from_atom):
    # TODO: ignores conditional effect conditions
    facts = []
    for precondition in get_literals(instance.action.precondition):
        if precondition.negated:
            continue
        args = apply_mapping(precondition.args, instance.var_mapping)
        literal = precondition.__class__(precondition.predicate, args)
        fact = fact_from_fd(literal)
        if fact in node_from_atom:
            facts.append(fact)
    return facts
