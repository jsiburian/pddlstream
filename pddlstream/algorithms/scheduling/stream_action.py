from collections import OrderedDict

from pddlstream.core.conversion import pddl_from_object
from pddlstream.core.syntax import Not
from pddlstream.core.utils import INF
from pddlstream.fastdownward.tasks import make_action, make_domain, make_parameters
from pddlstream.language.function import FunctionResult
from pddlstream.language.statistics import check_effort
from pddlstream.language.stream import StreamResult

BOUND_PREDICATE = '_bound'


def enforce_single_binding(result, preconditions, effects):
    binding_facts = [(BOUND_PREDICATE, pddl_from_object(out)) for out in result.output_objects]
    preconditions.extend(Not(fact) for fact in binding_facts)
    effects.extend(fact for fact in binding_facts)


def get_stream_actions(results, unique_binding=False, effort_scale=1, max_effort=INF, **kwargs):
    result_from_name = OrderedDict()
    stream_actions = []
    for result in results:
        if type(result) is FunctionResult:
            continue
        effort = result.get_effort(**kwargs)
        if not check_effort(effort, max_effort):
            continue
        name = f'{result.external.name}-{len(result_from_name)}'
        assert name not in result_from_name
        result_from_name[name] = result

        preconditions = list(result.instance.get_domain())
        effects = list(result.get_certified()) + [result.stream_fact]
        if unique_binding:
            enforce_single_binding(result, preconditions, effects)
        cost = effort_scale * effort
        stream_actions.append(make_action(name, [], preconditions, effects, cost))
    return stream_actions, result_from_name


def add_stream_actions(domain, results, **kwargs):
    if not results:
        return domain, {}
    stream_actions, result_from_name = get_stream_actions(results, **kwargs)
    output_objects = []
    for result in result_from_name.values():
        if isinstance(result, StreamResult):
            output_objects.extend(map(pddl_from_object, result.output_objects))
    new_constants = list(make_parameters(set(output_objects) | set(domain.constants)))
    # to_untyped_strips, free_variables
    new_domain = make_domain(
        constants=new_constants,
        predicates=domain.predicates,
        actions=domain.actions[:] + stream_actions,
        axioms=domain.axioms,
    )
    return new_domain, result_from_name
