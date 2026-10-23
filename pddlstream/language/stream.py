import logging
from collections import Counter
from collections.abc import Sequence

from pddlstream.algorithms.common import INTERNAL_EVALUATION, add_fact
from pddlstream.algorithms.downward import make_axiom
from pddlstream.core.context import get_clock
from pddlstream.language.constants import AND, Fact, StreamAction, concatenate, get_args, get_prefix, is_parameter
from pddlstream.language.conversion import (
    evaluation_from_fact,
    get_formula_operators,
    list_from_conjunction,
    obj_from_value_expression,
    objects_from_values,
    substitute_expression,
    substitute_fact,
    values_from_objects,
)
from pddlstream.language.external import (
    DEBUG,
    SHARED_DEBUG,
    External,
    ExternalInfo,
    Instance,
    Result,
    convert_constants,
    get_procedure_fn,
    parse_lisp_list,
    select_inputs,
)
from pddlstream.language.generator import BoundedGenerator, from_fn, get_next, universe_test
from pddlstream.language.object import (
    DebugValue,
    Object,
    OptimisticObject,
    SharedDebugValue,
    SharedOptValue,
    UniqueOptValue,
)
from pddlstream.utils import apply_mapping, get_mapping, irange, str_from_object

logger = logging.getLogger(__name__)

VERBOSE_FAILURES = True
VERBOSE_WILD = False
DEFAULT_UNIQUE = False
NEGATIVE_BLOCKED = True
NEGATIVE_SUFFIX = '-negative'
# TODO: could also make only wild facts and automatically identify output tuples satisfying certified
# TODO: default effort cost of streams with more inputs to be higher (but negated are free)
# TODO: automatically convert to test streams on inputs

##################################################


class PartialInputs:
    def __init__(self, inputs='', unique=DEFAULT_UNIQUE, test=universe_test):
        self.inputs = tuple(inputs.split())
        self.unique = unique  # TODO: refactor this
        self.test = test
        self.stream = None

    def get_opt_gen_fn(self, instance):
        # TODO: just condition on the external
        external = instance.external
        inputs = external.inputs if self.unique else self.inputs
        assert set(inputs) <= set(external.inputs)
        unique = set(inputs) == set(external.inputs)

        # TODO: ensure no scoping errors with inputs
        def gen_fn(*input_values):
            if not self.test(*input_values):
                return
            # TODO: recover input_objects from input_values
            selected_objects = select_inputs(instance, inputs)
            for idx in irange(instance.num_optimistic):  # self.num
                if unique:
                    outputs = tuple(UniqueOptValue(instance, idx, out) for out in external.outputs)
                else:
                    outputs = tuple(
                        SharedOptValue(external.name, inputs, selected_objects, out) for out in external.outputs
                    )
                yield [outputs]

        return gen_fn

    def __repr__(self):
        return repr(self.__dict__)


def get_debug_gen_fn(stream, shared=True):
    if shared:
        return from_fn(lambda *args, **kwargs: tuple(SharedDebugValue(stream.name, o) for o in stream.outputs))
    return from_fn(lambda *args, **kwargs: tuple(DebugValue(stream.name, args, o) for o in stream.outputs))


##################################################


class WildOutput:
    def __init__(self, values=[], facts=[], actions=[], enumerated=False, replan=False):
        self.values = values
        self.facts = facts
        self.actions = actions
        if self.actions:
            raise NotImplementedError()
        self.enumerated = enumerated
        self.replan = replan  # Reports back whether the problem has changed substantially

    def __iter__(self):
        return iter([self.values, self.facts])


class StreamInfo(ExternalInfo):
    def __init__(
        self, opt_gen_fn=None, negate=False, simultaneous=False, verbose=True, **kwargs
    ):  # TODO: set negate to None to express no user preference
        # TODO: could change frequency/priority for the incremental algorithm
        # TODO: maximum number of evaluations per iteration of adaptive
        super().__init__(**kwargs)
        # TODO: call this an abstraction instead
        self.opt_gen_fn = PartialInputs() if opt_gen_fn is None else opt_gen_fn
        self.negate = negate
        self.simultaneous = simultaneous
        self.verbose = verbose
        # TODO: make this false by default for negated test streams


##################################################


class StreamResult(Result):
    def __init__(self, instance, output_objects, opt_index=None, call_index=None, list_index=None, optimistic=True):
        super().__init__(instance, opt_index, call_index, optimistic)
        self.output_objects = tuple(output_objects)
        assert len(self.output_objects) == len(self.external.outputs)
        self.list_index = list_index
        self._mapping = None
        self._certified = None
        self._stream_fact = None

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = get_mapping(self.external.outputs, self.output_objects)
            self._mapping.update(self.instance.mapping)
        return self._mapping

    @property
    def stream_fact(self):
        if self._stream_fact is None:
            self._stream_fact = substitute_expression(self.external.stream_fact, self.mapping)
        return self._stream_fact

    @property
    def certified(self):
        if self._certified is None:
            self._certified = substitute_expression(self.external.certified, self.mapping)
        return self._certified

    def get_certified(self):
        return self.certified

    def get_action(self):
        return StreamAction(self.name, self.input_objects, self.output_objects)

    def get_optimistic(self):
        raise NotImplementedError()
        index = 0
        return self.instance.opt_results[index]

    def remap_inputs(self, bindings):
        new_instance = self.instance.remap_inputs(bindings)
        return self.__class__(
            new_instance, self.output_objects, self.opt_index, self.call_index, self.list_index, self.optimistic
        )

    def is_successful(self):
        return True

    def __repr__(self):
        inputs, outputs = str_from_object(self.instance.input_objects), str_from_object(self.output_objects)
        return f'{self.external.name}:{inputs}->{outputs}'


##################################################


class StreamInstance(Instance):
    _Result = StreamResult

    def __init__(self, stream, input_objects, fluent_facts):
        super().__init__(stream, input_objects)
        self._generator = None
        self.fluent_facts = frozenset(fluent_facts)
        self.opt_gen_fns = [
            opt_gen_fn.get_opt_gen_fn(self) if isinstance(opt_gen_fn, PartialInputs) else opt_gen_fn
            for opt_gen_fn in self.external.opt_gen_fns
        ]
        self.opt_gens = len(self.opt_gen_fns) * [None]
        self._axiom_predicate = None
        self._disabled_axiom = None
        # TODO: keep track of unique outputs to prune repeated ones

    def _check_output_values(self, new_values):
        if not isinstance(new_values, Sequence):
            raise ValueError(f'An output list for stream [{self.external.name}] is not a sequence: {new_values}')
        for output_values in new_values:
            if not isinstance(output_values, Sequence):
                raise ValueError(
                    f'An output tuple for stream [{self.external.name}] is not a sequence: {output_values}'
                )
            if len(output_values) != len(self.external.outputs):
                raise ValueError(
                    f'An output tuple for stream [{self.external.name}] has length {len(output_values)} '
                    f'instead of {len(self.external.outputs)}: {output_values}'
                )

    def _check_wild_facts(self, new_facts):
        if not isinstance(new_facts, Sequence):
            raise ValueError(f'Output wild facts for wild stream [{self.external.name}] is not a sequence: {new_facts}')

    def reset(self):
        super().reset()
        self.previous_outputs = set()
        self.num_optimistic = 1

    #########################

    def get_result(self, output_objects, opt_index=None, list_index=None, optimistic=True):
        # TODO: rename to create_result because not unique
        # TODO: ideally would increment a flag per stream for each failure
        call_index = self.num_calls
        return self._Result(
            instance=self,
            output_objects=tuple(output_objects),
            opt_index=opt_index,
            call_index=call_index,
            list_index=list_index,
            optimistic=optimistic,
        )

    def get_all_input_objects(self):  # TODO: lazily compute
        return set(self.input_objects) | {o for f in self.fluent_facts for o in get_args(f)}

    def get_fluent_values(self):
        return [Fact(get_prefix(f), values_from_objects(get_args(f))) for f in self.fluent_facts]

    def _create_generator(self):
        if self._generator is None:
            input_values = self.get_input_values()
            if self.external.is_fluent:  # self.fluent_facts
                self._generator = self.external.gen_fn(*input_values, fluents=self.get_fluent_values())
            else:
                self._generator = self.external.gen_fn(*input_values)
        return self._generator

    def _next_wild(self):
        output, self.enumerated = get_next(self._generator, default=[])
        if not isinstance(output, WildOutput):
            output = WildOutput(values=output)
        return output

    def _next_outputs(self):
        # TODO: deprecate
        self._create_generator()
        # TODO: shuffle history
        # TODO: return all test stream outputs at once
        if self.num_calls == len(self.history):
            self.history.append(self._next_wild())
        return self.history[self.num_calls]

    def dump_new_values(self, new_values=[]):
        if (not new_values and VERBOSE_FAILURES) or (new_values and self.info.verbose):
            logger.info(
                f'iter={self.get_iteration()}, outs={len(new_values)}) {self.external.name}:'
                f'{str_from_object(self.get_input_values())}->{str_from_object(new_values)}'
            )

    def dump_new_facts(self, new_facts=[]):
        if VERBOSE_WILD and new_facts:
            # TODO: format all_new_facts
            logger.info(
                f'iter={self.get_iteration()}, facts={self.external.name}) '
                f'{str_from_object(self.get_input_values())}:{new_facts}->{len(new_facts)}'
            )

    def next_results(self, verbose=False):
        assert not self.enumerated
        start_time = get_clock().now()
        start_history = len(self.history)
        new_values, new_facts = self._next_outputs()
        self._check_output_values(new_values)
        self._check_wild_facts(new_facts)
        if verbose:
            self.dump_new_values(new_values)
            self.dump_new_facts(new_facts)

        objects = [objects_from_values(output_values) for output_values in new_values]
        new_objects = list(filter(lambda o: o not in self.previous_outputs, objects))
        self.previous_outputs.update(new_objects)  # Only counting new outputs as successes
        new_results = [
            self.get_result(output_objects, list_index=list_index, optimistic=False)
            for list_index, output_objects in enumerate(new_objects)
        ]
        if start_history <= len(self.history) - 1:
            self.update_statistics(start_time, new_results)
        new_facts = list(map(obj_from_value_expression, new_facts))
        self.successful |= any(r.is_successful() for r in new_results)
        self.num_calls += 1  # Must be after get_result
        return new_results, new_facts

    #########################

    def get_representative_optimistic(self):
        # Prefer the most abstract optimistic generator (a user-provided opt_gen_fn when there is one) so that
        # refined objects keep its values, which downstream opt_gen_fns and function opt_fns expect.
        # The unique level (index 0) only yields UniqueOptValue placeholders and serves as the fallback.
        for opt_index in reversed(range(len(self.opt_gens))):
            opt_gen = self._create_opt_generator(opt_index)
            if opt_gen.history and opt_gen.history[0]:
                return opt_gen.history[0][0]
        return None

    def wrap_optimistic(self, output_values, call_index):
        output_objects = []
        representative_outputs = self.get_representative_optimistic()
        assert representative_outputs is not None
        for name, value, rep in zip(self.external.outputs, output_values, representative_outputs):
            # TODO: retain the value from a custom opt_gen_fn but use unique
            param = value
            value = rep
            output_objects.append(OptimisticObject.from_opt(value, param))
        return tuple(output_objects)

    def _create_opt_generator(self, opt_index=None):
        # TODO: automatically refine opt_index based on self.opt_gens
        if opt_index is None:
            opt_index = self.opt_index
        if self.opt_gens[opt_index] is None:
            self.opt_gens[opt_index] = BoundedGenerator(self.opt_gen_fns[opt_index](*self.get_input_values()))
            opt_gen = self.opt_gens[opt_index]
            try:
                next(opt_gen)  # next | list
            except StopIteration:
                pass
        return self.opt_gens[opt_index]

    def next_optimistic(self):
        if self.enumerated or self.disabled:
            return []
        opt_gen = self._create_opt_generator(self.opt_index)
        # TODO: how do I distinguish between real and not real verifications of things?
        output_set = set()
        opt_results = []
        for output_list in opt_gen.history:
            self._check_output_values(output_list)
            for output_values in output_list:
                call_index = len(opt_results)
                output_objects = self.wrap_optimistic(output_values, call_index)
                if output_objects not in output_set:
                    output_set.add(output_objects)  # No point returning the exact thing here...
                    opt_results.append(
                        self._Result(
                            instance=self,
                            output_objects=output_objects,
                            opt_index=self.opt_index,
                            call_index=call_index,
                            list_index=0,
                        )
                    )
        return opt_results

    def get_blocked_fact(self):
        if self.external.is_fluent:
            assert self._axiom_predicate is not None
            return Fact(self._axiom_predicate, self.input_objects)
        return Fact(self.external.blocked_predicate, self.input_objects)

    def _disable_fluent(self, evaluations, domain):
        assert self.external.is_fluent
        if self.successful or (self._axiom_predicate is not None):
            return
        self.disabled = True
        index = len(self.external.disabled_instances)
        self.external.disabled_instances.append(self)
        self._axiom_predicate = f'_ax{self.external.blocked_predicate}-{index}'
        add_fact(
            evaluations,
            self.get_blocked_fact(),
            result=INTERNAL_EVALUATION,
            complexity=self.compute_complexity(evaluations),
        )
        # TODO: allow reporting back minimum unsatisfiable subset

        static_fact = Fact(self._axiom_predicate, self.external.inputs)
        preconditions = [static_fact] + list(self.fluent_facts)
        derived_fact = Fact(self.external.blocked_predicate, self.external.inputs)
        self._disabled_axiom = make_axiom(
            parameters=self.external.inputs, preconditions=preconditions, derived=derived_fact
        )
        domain.axioms.append(self._disabled_axiom)

    def _disable_negated(self, evaluations):
        assert self.external.is_negated
        if self.successful:
            return
        self.disabled = True
        add_fact(
            evaluations,
            self.get_blocked_fact(),
            result=INTERNAL_EVALUATION,
            complexity=self.compute_complexity(evaluations),
        )

    def disable(self, evaluations, domain):
        if self.external.is_fluent:
            self._disable_fluent(evaluations, domain)
        elif self.external.is_negated:
            self._disable_negated(evaluations)
        else:
            self.disabled = True

    def enable(self, evaluations, domain):
        if not self.disabled:
            return
        evaluations.pop(evaluation_from_fact(self.get_blocked_fact()), None)

    def remap_inputs(self, bindings):
        # TODO: speed this procedure up
        input_objects = apply_mapping(self.input_objects, bindings)
        fluent_facts = [substitute_fact(f, bindings) for f in self.fluent_facts]
        new_instance = self.external.get_instance(input_objects, fluent_facts=fluent_facts)
        new_instance.opt_index = self.opt_index
        return new_instance

    def __repr__(self):
        return f'{self.external.name}:{self.input_objects}->{self.external.outputs}'


##################################################


class Stream(External):
    _Instance = StreamInstance

    def __init__(self, name, gen_fn, inputs, domain, outputs, certified, info=StreamInfo(), fluents=[]):
        super().__init__(name, info, inputs, domain)
        self.outputs = tuple(outputs)
        self.certified = tuple(map(convert_constants, certified))
        self.constants.update(a for i in certified for a in get_args(i) if not is_parameter(a))
        self.fluents = fluents

        for p, c in Counter(self.outputs).items():
            if not is_parameter(p):
                raise ValueError(f'Output [{p}] for stream [{name}] is not a parameter')
            if c != 1:
                raise ValueError(f'Output [{p}] for stream [{name}] is not unique')
        for p in set(self.inputs) & set(self.outputs):
            raise ValueError(f'Parameter [{p}] for stream [{name}] is both an input and output')
        certified_parameters = {a for i in certified for a in get_args(i) if is_parameter(a)}
        for p in certified_parameters - set(self.inputs + self.outputs):
            raise ValueError(f'Parameter [{p}] for stream [{name}] is not included within outputs')
        for p in set(self.outputs) - certified_parameters:
            logger.warning(f'Output [{p}] for stream [{name}] is not covered by a certified condition')

        # TODO: automatically switch to unique if only used once
        self.gen_fn = gen_fn  # DEBUG_MODES
        if gen_fn == DEBUG:
            self.gen_fn = get_debug_gen_fn(self, shared=False)  # TODO: list of abstractions that is considered in turn
        elif gen_fn == SHARED_DEBUG:
            self.gen_fn = get_debug_gen_fn(self, shared=True)
        assert callable(self.gen_fn)

        self.opt_gen_fns = [PartialInputs(unique=True)]
        if (
            not self.is_test
            and not self.is_special
            and not (isinstance(self.info.opt_gen_fn, PartialInputs) and self.info.opt_gen_fn.unique)
        ):
            self.opt_gen_fns.append(self.info.opt_gen_fn)

        if NEGATIVE_BLOCKED:
            self.blocked_predicate = f'~{self.name}{NEGATIVE_SUFFIX}'  # Args are self.inputs
        else:
            self.blocked_predicate = f'~{self.name}'
        self.disabled_instances = []  # For tracking disabled axioms
        self.stream_fact = Fact(f'_{name}', concatenate(inputs, outputs))  # TODO: just add to certified?

        if self.is_negated:
            if self.outputs:
                raise ValueError(f'Negated streams cannot have outputs: {self.outputs}')
            for certified in self.certified:
                if not (set(self.inputs) <= set(get_args(certified))):
                    raise ValueError('Negated streams must have certified facts including all input parameters')

    @property
    def num_opt_fns(self):
        return len(self.opt_gen_fns) - 1

    @property
    def has_outputs(self):
        return bool(self.outputs)

    @property
    def is_test(self):
        return not self.has_outputs

    @property
    def is_fluent(self):
        return bool(self.fluents)

    @property
    def is_negated(self):
        return self.info.negate

    @property
    def is_function(self):
        return False

    def get_instance(self, input_objects, fluent_facts=frozenset()):
        input_objects = tuple(input_objects)
        fluent_facts = frozenset(fluent_facts)
        assert all(isinstance(obj, Object) or isinstance(obj, OptimisticObject) for obj in input_objects)
        key = (input_objects, fluent_facts)
        if key not in self.instances:
            self.instances[key] = self._Instance(self, input_objects, fluent_facts)
        return self.instances[key]

    def as_test_stream(self):
        # TODO: method that converts a stream into a test stream (possibly from ss)
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.name}:{self.inputs}->{self.outputs}'


##################################################


def parse_stream(lisp_list, stream_map, stream_info):
    value_from_attribute = parse_lisp_list(lisp_list)
    assert set(value_from_attribute) <= {':stream', ':inputs', ':domain', ':fluents', ':outputs', ':certified'}
    name = value_from_attribute[':stream']
    domain = value_from_attribute.get(':domain', None)
    # TODO: dnf_from_positive_formula(value_from_attribute.get(':domain', []))
    if not (get_formula_operators(domain) <= {AND}):
        # TODO: allow positive DNF
        raise ValueError(f'Stream [{name}] domain must be a conjunction')
    certified = value_from_attribute.get(':certified', None)
    if not (get_formula_operators(certified) <= {AND}):
        raise ValueError(f'Stream [{name}] certified must be a conjunction')
    return Stream(
        name,
        get_procedure_fn(stream_map, name),
        value_from_attribute.get(':inputs', []),
        list_from_conjunction(domain),
        value_from_attribute.get(':outputs', []),
        list_from_conjunction(certified),
        stream_info.get(name, StreamInfo()),
        fluents=value_from_attribute.get(':fluents', []),
    )
