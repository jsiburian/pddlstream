"""Certificates: the stream calls behind a plan, back to the initial facts.

Each fact in the solver's evaluations records what supports it (core/evaluations.py). Following those records from a
plan's preimage facts through each result's domain facts gives the plan's supports, a DAG whose leaves are initial
facts and facts that the solver adds itself.
"""

from typing import NamedTuple

from pddlstream.core.conversion import evaluation_from_fact, param_from_object, replace_expression
from pddlstream.core.evaluations import INTERNAL_EVALUATION
from pddlstream.core.problem import Certificate
from pddlstream.language.external import Instance, Result
from pddlstream.language.stream import StreamInstance, StreamResult


class Support(NamedTuple):
    """A stream call that certified facts a plan relies on (all as values; placeholders as in plans, e.g. '#o1')."""

    stream: str
    inputs: tuple
    outputs: tuple
    domain: list  # The facts the call required
    fluents: list  # For fluent streams, the state facts the call was made with
    certified: list  # The facts it certified


class PlanCertificate(Certificate):
    """A Certificate (all_facts, preimage_facts), which it unpacks and compares like, with the plan's supports.

    supports: the stream calls behind the preimage facts, each after those that support its domain facts
    internal: preimage facts that the solver added itself (e.g. compiled plan constraints)
    unsupported: preimage facts no stream call certified yet (facts about the outputs of deferred streams)
    """

    supports: list
    internal: list
    unsupported: list

    def __new__(cls, all_facts, preimage_facts, supports=(), internal=(), unsupported=()):
        self = super().__new__(cls, all_facts, preimage_facts)
        self.supports = list(supports)
        self.internal = list(internal)
        self.unsupported = list(unsupported)
        return self

    @property
    def complete(self):
        """Whether every preimage fact is initial, internal or certified by a support."""
        return not self.unsupported


def get_fluent_facts(instance):
    return list(instance.fluent_facts) if isinstance(instance, StreamInstance) else []


def extract_supports(evaluations, facts):
    """The results (or, for wild facts, instances) that support facts, transitively, each after those that support
    its domain facts; the internal facts among them; and the facts without a recorded support."""
    supports, internal, unsupported = [], [], []
    expanded, emitted = set(), set()
    stack = [(fact, False) for fact in reversed(list(facts))]
    while stack:
        fact, done = stack.pop()
        evaluation = evaluation_from_fact(fact)
        node = evaluations.get(evaluation)
        if done:
            if node.result not in emitted:
                emitted.add(node.result)
                supports.append(node.result)
            continue
        if evaluation in expanded:
            continue
        expanded.add(evaluation)
        if node is None:
            unsupported.append(fact)
            continue
        if node.result is INTERNAL_EVALUATION:
            internal.append(fact)
            continue
        if isinstance(node.result, Result):
            instance = node.result.instance
        elif isinstance(node.result, Instance):  # A wild fact
            instance = node.result
        else:  # An initial fact
            continue
        stack.append((fact, True))
        stack.extend((domain_fact, False) for domain_fact in reversed(instance.get_domain()))
    return supports, internal, unsupported


def value_fact(fact):
    """A fact with values instead of objects; optimistic objects stay placeholders, as in plans."""
    return replace_expression(fact, param_from_object)


def support_from_result(support):
    """The Support of a result, or of an instance that produced wild facts."""
    if isinstance(support, Result):
        instance, certified = support.instance, list(map(value_fact, support.get_certified()))
    else:
        instance, certified = support, [tuple(fact) for output in support.history for fact in output.facts]
    outputs = support.output_objects if isinstance(support, StreamResult) else ()
    return Support(
        stream=instance.external.name,
        inputs=tuple(map(param_from_object, instance.input_objects)),
        outputs=tuple(map(param_from_object, outputs)),
        domain=list(map(value_fact, instance.get_domain())),
        fluents=list(map(value_fact, get_fluent_facts(instance))),
        certified=certified,
    )


def certify(certificate, evaluations, preimage_facts):
    """certificate (with value-level facts) extended with the supports of the object-level preimage_facts."""
    supports, internal, unsupported = extract_supports(evaluations, preimage_facts)
    return PlanCertificate(
        *certificate, map(support_from_result, supports), map(value_fact, internal), map(value_fact, unsupported)
    )
