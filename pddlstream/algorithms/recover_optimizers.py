from pddlstream.algorithms.reorder import get_stream_plan_components
from pddlstream.algorithms.scheduling.utils import partition_external_plan
from pddlstream.language.constants import get_prefix, is_plan, get_args
from pddlstream.language.conversion import evaluation_from_fact
from pddlstream.language.function import FunctionResult
from pddlstream.language.optimizer import ComponentStream, OptimizerStream
from pddlstream.utils import get_mapping, safe_apply_mapping

CLUSTER = True


def get_optimizer(result):
    return result.external.optimizer if isinstance(result.external, ComponentStream) else None


##################################################


def combine_optimizer_plan(stream_plan, functions):
    if not stream_plan:
        return stream_plan
    optimizer = get_optimizer(stream_plan[-1])
    if optimizer is None:
        return stream_plan
    function_plan = list(filter(lambda r: get_prefix(r.instance.external.head) in optimizer.objectives, functions))
    external_plan = stream_plan + function_plan
    cluster_plans = get_stream_plan_components(external_plan) if CLUSTER else [external_plan]
    optimizer_plan = []
    for cluster_plan in cluster_plans:
        if all(isinstance(r, FunctionResult) for r in cluster_plan):
            continue
        stream = OptimizerStream(optimizer, cluster_plan)
        instance = stream.get_instance(stream.input_objects, fluent_facts=stream.fluent_facts)
        result = instance.get_result(stream.output_objects)
        optimizer_plan.append(result)
    return optimizer_plan


def combine_optimizers(evaluations, external_plan):
    if not is_plan(external_plan):
        return external_plan
    stream_plan, function_plan = partition_external_plan(external_plan)
    optimizers = {get_optimizer(r) for r in stream_plan}  # None is like a unique optimizer
    if len(optimizers - {None}) == 0:
        return external_plan

    print(f'Constraint plan: {external_plan}')
    combined_results = []
    for optimizer in optimizers:
        relevant_results = [r for r in stream_plan if get_optimizer(r) == optimizer]
        combined_results.extend(combine_optimizer_plan(relevant_results, function_plan))
    combined_results.extend(function_plan)

    current_facts = set()
    for result in combined_results:
        current_facts.update(filter(lambda f: evaluation_from_fact(f) in evaluations, result.get_domain()))
    combined_plan = []
    while combined_results:
        for result in combined_results:
            if set(result.get_domain()) <= current_facts:
                combined_plan.append(result)
                current_facts.update(result.get_certified())
                combined_results.remove(result)
                break
        else:  # TODO: can also just try one cluster and return
            raise RuntimeError()
    return combined_plan


##################################################


def retrace_instantiation(fact, streams, evaluations, free_parameters, visited_facts, planned_results):
    # Makes two assumptions:
    # 1) Each stream achieves a "primary" fact that uses all of its inputs + outputs
    # 2) Outputs are only free parameters (no constants)
    if (evaluation_from_fact(fact) in evaluations) or (fact in visited_facts):
        return
    visited_facts.add(fact)
    for stream in streams:
        for cert in stream.certified:
            if get_prefix(fact) == get_prefix(cert):
                mapping = get_mapping(get_args(cert), get_args(fact))  # Should be same anyways
                if not all(p in mapping for p in (stream.inputs + stream.outputs)):
                    # TODO: assumes another effect is sufficient for binding
                    # Create arbitrary objects for inputs/outputs that aren't mentioned
                    # Can lead to incorrect ordering
                    continue

                input_objects = safe_apply_mapping(stream.inputs, mapping)
                output_objects = safe_apply_mapping(stream.outputs, mapping)
                if not all(out in free_parameters for out in output_objects):
                    # Can only bind if free
                    continue
                instance = stream.get_instance(input_objects)
                for new_fact in instance.get_domain():
                    retrace_instantiation(
                        new_fact, streams, evaluations, free_parameters, visited_facts, planned_results
                    )
                planned_results.append(instance.get_result(output_objects))
