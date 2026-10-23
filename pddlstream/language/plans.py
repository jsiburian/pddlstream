"""Stream plans (sequences of stream and function results): their partial orders, components and partitions."""

from pddlstream.core.utils import get_connected_components, is_acyclic
from pddlstream.language.function import FunctionResult
from pddlstream.language.stream import StreamResult


def get_output_objects(result):
    if isinstance(result, StreamResult):
        return result.output_objects
    return tuple()


def get_object_orders(stream_plan):
    # TODO: check that only one result per output object
    partial_orders = set()
    for i, stream1 in enumerate(stream_plan):
        for stream2 in stream_plan[i + 1 :]:
            if set(get_output_objects(stream1)) & stream2.instance.get_all_input_objects():
                partial_orders.add((stream1, stream2))
    return partial_orders


def get_fact_orders(stream_plan, init_facts=set()):
    # TODO: explicitly recover this from plan_streams
    # TODO: init_facts isn't used in practice
    achieved_facts = set(init_facts)
    partial_orders = set()
    for i, stream1 in enumerate(stream_plan):
        new_facts = set(stream1.get_certified()) - achieved_facts
        for stream2 in stream_plan[i + 1 :]:  # Prevents circular
            if new_facts & set(stream2.get_domain()):
                partial_orders.add((stream1, stream2))
        achieved_facts.update(new_facts)
    return partial_orders


def get_partial_orders(stream_plan, use_facts=True, **kwargs):
    partial_orders = get_object_orders(stream_plan)
    if use_facts:
        partial_orders.update(get_fact_orders(stream_plan, **kwargs))
    assert is_acyclic(stream_plan, partial_orders)
    return partial_orders


def get_stream_plan_components(external_plan, **kwargs):
    partial_orders = get_partial_orders(external_plan, **kwargs)
    return get_connected_components(external_plan, partial_orders)


def partition_external_plan(external_plan):
    function_plan = list(filter(lambda r: isinstance(r, FunctionResult), external_plan))
    stream_plan = list(filter(lambda r: r not in function_plan, external_plan))
    return stream_plan, function_plan
