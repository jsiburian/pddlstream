"""Converting between pddlstream's facts and the translator's PDDL objects: parsing domains, building tasks,
domains, actions and axioms, and applying plans to states."""

from collections import defaultdict, namedtuple

from pddlstream.core.conversion import (
    is_atom,
    is_negated_atom,
    obj_from_pddl,
    objects_from_evaluations,
    pddl_from_object,
    pddl_list_from_expression,
)
from pddlstream.core.problem import Action
from pddlstream.core.syntax import EQ, NOT, OBJECT, TOTAL_COST, Evaluation, Head, Not, get_args, get_prefix
from pddlstream.core.utils import MockSet, find_unique, safe_zip
from pddlstream.fastdownward.translator import get_cost_scale, instantiate, normalize, pddl, pddl_parser

##################################################

GOAL_NAME = '@goal'  # @goal-reachable

INTERNAL_AXIOM = 'new-axiom'  # @0
IDENTICAL = 'identical'  # lowercase is critical (!= instead?)
##################################################


def parse_lisp(lisp):
    return pddl_parser.lisp_parser.parse_nested_list(lisp.splitlines())


# TODO: dynamically generate type_dict and predicate_dict
Domain = namedtuple(
    'Domain',
    [
        'name',
        'requirements',
        'types',
        'type_dict',
        'constants',
        'predicates',
        'predicate_dict',
        'functions',
        'actions',
        'axioms',
        'pddl',
    ],
)


def parse_sequential_domain(domain_pddl):
    if isinstance(domain_pddl, Domain):
        return domain_pddl
    args = list(pddl_parser.parsing_functions.parse_domain_pddl(parse_lisp(domain_pddl))) + [domain_pddl]
    domain = Domain(*args)
    return domain


Problem = namedtuple(
    'Problem', ['task_name', 'task_domain_name', 'task_requirements', 'objects', 'init', 'goal', 'use_metric', 'pddl']
)


def parse_problem(domain, problem_pddl):
    if isinstance(problem_pddl, Problem):
        return problem_pddl
    args = list(
        pddl_parser.parsing_functions.parse_task_pddl(parse_lisp(problem_pddl), domain.type_dict, domain.predicate_dict)
    ) + [problem_pddl]
    return Problem(*args)


##################################################

# fact -> evaluation -> fd


def fd_from_fact(fact):
    # TODO: convert to evaluation?
    prefix = get_prefix(fact)
    if prefix == NOT:
        return fd_from_fact(fact[1]).negate()
    args = list(map(pddl_from_object, get_args(fact)))
    return pddl.Atom(prefix, args)


def fact_from_fd(fd):
    assert isinstance(fd, pddl.Literal)
    atom = (fd.predicate,) + tuple(map(obj_from_pddl, fd.args))
    return Not(atom) if fd.negated else atom


def evaluation_from_fd(fd):
    if isinstance(fd, pddl.Literal):
        head = Head(fd.predicate, tuple(map(obj_from_pddl, fd.args)))
        return Evaluation(head, not fd.negated)
    if isinstance(fd, pddl.f_expression.Assign):
        head = Head(fd.fluent.symbol, tuple(map(obj_from_pddl, fd.fluent.args)))
        return Evaluation(head, float(fd.expression.value) / get_cost_scale())  # Need to be careful due to rounding
    raise ValueError(fd)


def fd_from_evaluation(evaluation):
    name = evaluation.head.function
    args = tuple(map(pddl_from_object, evaluation.head.args))
    if is_atom(evaluation):
        return pddl.Atom(name, args)
    elif is_negated_atom(evaluation):
        return pddl.NegatedAtom(name, args)
    fluent = pddl.f_expression.PrimitiveNumericExpression(symbol=name, args=args)
    expression = pddl.f_expression.NumericConstant(evaluation.value)
    return pddl.f_expression.Assign(fluent, expression)


def fd_from_evaluations(evaluations):
    return [fd_from_evaluation(e) for e in evaluations if not is_negated_atom(e)]


##################################################


def parse_goal(goal_exp, domain):
    condition = pddl_parser.parsing_functions.parse_condition(
        pddl_list_from_expression(goal_exp), domain.type_dict, domain.predicate_dict
    )
    return condition.simplified()


def get_problem(evaluations, goal_exp, domain, unit_costs=False):
    objects = objects_from_evaluations(evaluations)
    typed_objects = list({make_object(pddl_from_object(obj)) for obj in objects} - set(domain.constants))
    # TODO: this doesn't include =
    init = fd_from_evaluations(evaluations)
    goal = pddl.Truth() if goal_exp is None else parse_goal(goal_exp, domain)
    problem_pddl = None
    return Problem(
        task_name=domain.name,
        task_domain_name=domain.name,
        objects=sorted(typed_objects, key=lambda o: o.name),
        task_requirements=pddl.tasks.Requirements([]),
        init=init,
        goal=goal,
        use_metric=not unit_costs,
        pddl=problem_pddl,
    )


def get_identical_atoms(objects):
    # TODO: optimistically evaluate (not (= ?o1 ?o2))
    init = []
    for fd_obj in objects:
        obj = obj_from_pddl(fd_obj.name)
        if obj.is_unique():
            init.append(pddl.Atom(IDENTICAL, (fd_obj.name, fd_obj.name)))
        else:
            assert obj.is_shared()
    return init


def task_from_domain_problem(domain, problem, add_identical=True):
    # TODO: prune evaluation that aren't needed in actions
    task_name, task_domain_name, task_requirements, objects, init, goal, use_metric, problem_pddl = problem

    assert domain.name == task_domain_name
    requirements = pddl.Requirements(sorted(set(domain.requirements.requirements + task_requirements.requirements)))
    objects = domain.constants + objects
    pddl_parser.parsing_functions.check_for_duplicates(
        [o.name for o in objects],
        errmsg='error: duplicate object %r',
        finalmsg='please check :constants and :objects definitions',
    )
    init.extend(pddl.Atom(EQ, (obj.name, obj.name)) for obj in objects)
    if add_identical:
        init.extend(get_identical_atoms(objects))

    task = pddl.Task(
        domain.name,
        task_name,
        requirements,
        domain.types,
        objects,
        domain.predicates,
        domain.functions,
        init,
        goal,
        domain.actions,
        domain.axioms,
        use_metric,
    )
    normalize.normalize(task)
    # task.add_axiom
    return task


##################################################


def get_derived_predicates(axioms):
    axioms_from_name = defaultdict(list)
    for axiom in axioms:
        axioms_from_name[axiom.name].append(axiom)
    return axioms_from_name


def get_fluents(domain):
    fluent_predicates = set(get_derived_predicates(domain.axioms))
    for action in domain.actions:
        for effect in action.effects:
            fluent_predicates.add(effect.literal.predicate)
    return fluent_predicates


def is_literal(condition):
    return isinstance(condition, pddl.Literal)


def get_literals(condition):
    if is_literal(condition):
        return [condition]
    if isinstance(condition, pddl.Truth):
        return []
    if isinstance(condition, pddl.Conjunction):
        literals = []
        for c in condition.parts:
            literals.extend(get_literals(c))
        return literals
    raise ValueError(condition)


def get_conjunctive_parts(condition):
    # TODO: apply recursively
    return condition.parts if isinstance(condition, pddl.Conjunction) else [condition]


def get_disjunctive_parts(condition):
    return condition.parts if isinstance(condition, pddl.Disjunction) else [condition]


##################################################


def normalize_domain_goal(domain, goal_exp):
    evaluations = []
    problem = get_problem(evaluations, goal_exp, domain, unit_costs=False)
    task = task_from_domain_problem(domain, problem)
    normalize.normalize(task)
    return task


##################################################


def literal_holds(state, literal):
    return (literal.positive() in state) != literal.negated


def conditions_hold(state, conditions):
    return all(literal_holds(state, cond) for cond in conditions)


def get_precondition(operator):
    if isinstance(operator, pddl.Action) or isinstance(operator, pddl.PropositionalAction):
        return operator.precondition
    elif isinstance(operator, pddl.Axiom) or isinstance(operator, pddl.PropositionalAxiom):
        return operator.condition
    raise ValueError(operator)


def get_conditional_effects(operator):
    if isinstance(operator, pddl.PropositionalAction):
        return [(cond, effect.negate()) for cond, effect in operator.del_effects] + [
            (cond, effect) for cond, effect in operator.add_effects
        ]
    elif isinstance(operator, pddl.PropositionalAxiom):
        return [([], operator.effect)]
    raise ValueError(operator)


def apply_action(state, action):
    assert isinstance(action, pddl.PropositionalAction)
    # TODO: signed literals
    # TODO: relaxed_apply_action
    for conditions, effect in action.del_effects:
        if conditions_hold(state, conditions):
            state.discard(effect)
    for conditions, effect in action.add_effects:
        if conditions_hold(state, conditions):
            state.add(effect)


##################################################


def get_function_assignments(task):
    return {f.fluent: f.expression for f in task.init if isinstance(f, pddl.f_expression.FunctionAssignment)}


def get_action_instances(task, action_plan):
    type_to_objects = instantiate.get_objects_by_type(task.objects, task.types)
    function_assignments = get_function_assignments(task)
    predicate_to_atoms = instantiate.get_atoms_by_predicate(task.init)
    fluent_facts = MockSet()
    init_facts = set()
    action_instances = []
    for name, objects in action_plan:
        # TODO: what if more than one action of the same name due to normalization?
        # Normalized actions have same effects, so I just have to pick one
        # TODO: conditional effects and internal parameters
        action = find_unique(lambda a: a.name == name, task.actions)  # noqa: B023 (called right away)
        args = list(map(pddl_from_object, objects))
        variable_mapping = {p.name: a for p, a in safe_zip(action.parameters, args)}
        instance = action.instantiate(
            variable_mapping,
            init_facts,
            function_assignments,
            fluent_facts,
            type_to_objects,
            task.use_min_cost_metric,
            predicate_to_atoms,
        )
        assert instance is not None
        action_instances.append(instance)
    return action_instances


##################################################


def add_preimage_condition(condition, preimage, i):
    for literal in condition:
        preimage.setdefault(literal, set()).add(i)


def add_preimage_effect(effect, preimage):
    preimage.pop(effect, None)


def action_preimage(action, preimage, i):
    for conditions, effect in action.add_effects + action.del_effects:
        assert not conditions
        # TODO: can later select which conditional effects are used
        # TODO: might need to truely decide whether one should hold or not for a preimage
        # Maybe I should do that here
        add_preimage_effect(effect, preimage)
    add_preimage_condition(action.precondition, preimage, i)


def axiom_preimage(axiom, preimage, i):
    add_preimage_effect(axiom.effect, preimage)
    add_preimage_condition(axiom.condition, preimage, i)


def plan_preimage(combined_plan, goal=[]):
    action_plan = [action for action in combined_plan if isinstance(action, pddl.PropositionalAction)]
    step = len(action_plan)
    preimage = {condition: {step} for condition in goal}
    for operator in reversed(combined_plan):
        if isinstance(operator, pddl.PropositionalAction):
            step -= 1
            action_preimage(operator, preimage, step)
        elif isinstance(operator, pddl.PropositionalAxiom):
            axiom_preimage(operator, preimage, step)
        else:
            raise ValueError(operator)
    return preimage


##################################################


def get_supertypes(domain):
    """The supertypes (other than object) of each declared type."""
    parent = {pddl_type.name: pddl_type.basetype_name for pddl_type in domain.types}
    supertypes = {}
    for name in parent:
        chain, current = [], parent[name]
        while current not in (None, 'object'):
            chain.append(current)
            current = parent.get(current)
        supertypes[name] = chain
    return {name: chain for name, chain in supertypes.items() if name != 'object'}


def compile_types(domain):
    """Compiles a typed domain into an untyped one in which types are unary predicates: each typed parameter (of an
    action, axiom, effect or quantifier) becomes untyped with a condition on its type. Returns the supertypes of each
    type and the type of each typed constant, so that type facts can be added for the closure of the hierarchy."""
    if len(domain.types) == 1:
        return {}, {}
    supertypes = get_supertypes(domain)
    conflicts = set(supertypes) & set(domain.predicate_dict)
    if conflicts:
        raise ValueError(f'Types and predicates share the names {sorted(conflicts)}')
    for name in supertypes:
        add_predicate(domain, make_predicate(name, ['?x']))

    def untyped(parameters):
        return [pddl.TypedObject(parameter.name, 'object') for parameter in parameters]

    def type_atoms(parameters):
        for parameter in parameters:
            if not isinstance(parameter.type_name, str):
                raise NotImplementedError(f'Either types are not supported: {parameter}')
        return [pddl.Atom(p.type_name, [p.name]) for p in parameters if p.type_name != 'object']

    def conjoin(atoms, condition):
        return pddl.Conjunction(atoms + [condition]).simplified() if atoms else condition

    def compile_condition(condition):
        parts = [compile_condition(part) for part in condition.parts]
        if isinstance(condition, pddl.ExistentialCondition):
            parameters = condition.parameters
            return pddl.ExistentialCondition(untyped(parameters), [pddl.Conjunction(type_atoms(parameters) + parts)])
        if isinstance(condition, pddl.UniversalCondition):
            parameters = condition.parameters
            negated_types = [atom.negate() for atom in type_atoms(parameters)]
            return pddl.UniversalCondition(untyped(parameters), [pddl.Disjunction(negated_types + parts)])
        return condition.change_parts(parts) if parts else condition

    for action in domain.actions:
        action.precondition = conjoin(type_atoms(action.parameters), compile_condition(action.precondition))
        action.parameters = untyped(action.parameters)
        for effect in action.effects:
            effect.condition = conjoin(type_atoms(effect.parameters), compile_condition(effect.condition))
            effect.parameters = untyped(effect.parameters)
    for axiom in domain.axioms:
        axiom.condition = conjoin(type_atoms(axiom.parameters), compile_condition(axiom.condition))
        axiom.parameters = untyped(axiom.parameters)
    for predicate in domain.predicates + list(domain.functions):
        predicate.arguments = untyped(predicate.arguments)
    constant_types = {constant.name: constant.type_name for constant in domain.constants}
    domain.constants[:] = untyped(domain.constants)
    domain.types[:] = [pddl_type for pddl_type in domain.types if pddl_type.name == 'object']
    domain.type_dict.clear()
    domain.type_dict.update({pddl_type.name: pddl_type for pddl_type in domain.types})
    return supertypes, {name: type_name for name, type_name in constant_types.items() if type_name != 'object'}


def add_predicate(domain, predicate):
    if predicate.name in domain.predicate_dict:
        return False
    domain.predicates.append(predicate)
    domain.predicate_dict[predicate.name] = predicate
    return True


def make_object(obj, type=OBJECT):
    return pddl.TypedObject(obj, type)


def make_parameters(parameters, **kwargs):
    return tuple(make_object(p, **kwargs) for p in parameters)


def make_predicate(name, parameters):
    return pddl.Predicate(name, make_parameters(parameters))


def make_preconditions(preconditions):
    return pddl.Conjunction(list(map(fd_from_fact, preconditions)))


def make_effects(effects):
    return [pddl.Effect(parameters=[], condition=pddl.Truth(), literal=fd_from_fact(fact)) for fact in effects]


def make_cost(cost):
    if cost is None:
        return cost
    fluent = pddl.PrimitiveNumericExpression(symbol=TOTAL_COST, args=[])
    try:
        expression = pddl.NumericConstant(cost)
    except TypeError:
        expression = pddl.PrimitiveNumericExpression(
            symbol=get_prefix(cost), args=list(map(pddl_from_object, get_args(cost)))
        )
    return pddl.Increase(fluent=fluent, expression=expression)


def has_costs(domain):
    for action in domain.actions:
        if (action.cost is not None) or (action.cost == 0):
            return True
    return False


def set_unit_costs(domain):
    # Cost of None becomes zero if metric = True
    for action in domain.actions:
        action.cost = make_cost(1)


def make_action(name, parameters, preconditions, effects, cost=None):
    # Usually all parameters are external
    return pddl.Action(
        name=name,
        parameters=make_parameters(parameters),
        num_external_parameters=len(parameters),
        precondition=make_preconditions(preconditions),
        effects=make_effects(effects),
        cost=make_cost(cost),
    )


def make_axiom(parameters, preconditions, derived):
    predicate = get_prefix(derived)
    external_parameters = list(get_args(derived))
    internal_parameters = [p for p in parameters if p not in external_parameters]
    parameters = external_parameters + internal_parameters
    return pddl.Axiom(
        name=predicate,
        parameters=make_parameters(parameters),
        num_external_parameters=len(external_parameters),
        condition=make_preconditions(preconditions),
    )


def make_domain(constants=[], predicates=[], functions=[], actions=[], axioms=[]):
    types = [pddl.Type(OBJECT)]
    pddl_parser.parsing_functions.set_supertypes(types)
    return Domain(
        name='',
        requirements=pddl.Requirements([]),
        types=types,
        type_dict={ty.name: ty for ty in types},
        constants=constants,
        predicates=predicates,
        predicate_dict={p.name: p for p in predicates},
        functions=functions,
        actions=actions,
        axioms=axioms,
        pddl=None,
    )


def pddl_from_instance(instance):
    action = instance.action
    args = [instance.var_mapping[p.name] for p in action.parameters[: action.num_external_parameters]]
    return Action(action.name, args)
