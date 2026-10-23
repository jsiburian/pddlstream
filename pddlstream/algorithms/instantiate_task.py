import logging
from collections import Counter, defaultdict, deque, namedtuple
from time import time

from pddlstream.algorithms.downward import (
    GOAL_NAME,
    get_conditional_effects,
    get_conjunctive_parts,
    get_fluents,
    get_function_assignments,
    get_literals,
    get_precondition,
    literal_holds,
    parse_problem,
    parse_sequential_domain,
    task_from_domain_problem,
)
from pddlstream.algorithms.relation import Relation, compute_order, solve_satisfaction
from pddlstream.core.context import get_config
from pddlstream.fastdownward.planner import write_sas_task
from pddlstream.fastdownward.translator import (
    axiom_rules,
    fact_groups,
    instantiate,
    normalize,
    options,
    pddl,
    simplify,
    timers,
    translate,
    variable_order,
)
from pddlstream.language.constants import is_parameter
from pddlstream.utils import (
    MockSet,
    Verbose,
    apply_mapping,
    elapsed_time,
    flatten,
    str_from_object,
)

logger = logging.getLogger(__name__)

# Deprecated: pass fd_instantiate=False to solve() instead of assigning this (it was previously ignored)
FD_INSTANTIATE = True

InstantiatedTask = namedtuple(
    'InstantiatedTask', ['task', 'atoms', 'actions', 'axioms', 'reachable_action_params', 'goal_list']
)


def instantiate_goal(goal):
    goal_list = get_conjunctive_parts(goal)
    assert all(isinstance(item, pddl.Literal) for item in goal_list)
    return goal_list


def get_goal_instance(goal):
    return pddl.PropositionalAction(GOAL_NAME, instantiate_goal(goal), [], None)


##################################################


def get_constants(atom):
    return tuple((i, a) for i, a in enumerate(atom.args) if not is_parameter(a))


def instantiate_condition(action, is_static, args_from_predicate):
    parameters = {p.name for p in action.parameters}
    static_conditions = list(filter(is_static, get_literals(get_precondition(action))))
    static_parameters = set(filter(is_parameter, flatten(atom.args for atom in static_conditions)))
    if not (parameters <= static_parameters):
        raise NotImplementedError(
            f'Could not instantiate action {action.name} due to parameters: '
            f'{str_from_object(parameters - static_parameters)}'
        )
    atoms_from_cond = {
        condition: args_from_predicate[condition.predicate, get_constants(condition)] for condition in static_conditions
    }
    conditions, atoms = zip(*atoms_from_cond.items())
    relations = [Relation(conditions[index].args, atoms[index]) for index in compute_order(conditions, atoms)]
    solution = solve_satisfaction(relations)
    for element in solution.body:
        yield solution.get_mapping(element)


def get_reachable_action_params(instantiated_actions):
    # TODO: use pddl_from_instance
    reachable_action_params = defaultdict(list)
    for inst_action in instantiated_actions:
        action = inst_action.action
        parameters = [p.name for p in action.parameters]
        args = apply_mapping(parameters, inst_action.var_mapping)
        reachable_action_params[action].append(args)  # TODO: does this actually do anything
    return reachable_action_params


##################################################


def filter_negated(conditions, negated_from_name):
    return list(filter(lambda a: a.predicate not in negated_from_name, conditions))


def get_achieving_axioms(state, operators, negated_from_name={}):
    # TODO: order by stream effort
    # marking algorithm for propositional Horn logic
    unprocessed_from_literal = defaultdict(list)
    operator_from_literal = {}
    remaining_from_stream = {}
    reachable_operators = set()  # TODO: only keep facts

    queue = deque()

    def process_axiom(op, effect):
        reachable_operators.add(id(op))
        if effect not in operator_from_literal:
            operator_from_literal[effect] = op
            queue.append(effect)

    # TODO: could produce a list of all derived conditions
    for op in operators:
        preconditions = get_precondition(op)
        for cond, effect in get_conditional_effects(op):
            conditions = cond + preconditions
            remaining_from_stream[id(op), effect] = 0
            for literal in filter_negated(conditions, negated_from_name):
                if literal_holds(state, literal):
                    operator_from_literal[literal] = None
                else:
                    remaining_from_stream[id(op), effect] += 1
                    unprocessed_from_literal[literal].append((op, effect))
            if remaining_from_stream[id(op), effect] == 0:
                process_axiom(op, effect)

    while queue:
        literal = queue.popleft()
        for op, effect in unprocessed_from_literal[literal]:
            remaining_from_stream[id(op), effect] -= 1
            if remaining_from_stream[id(op), effect] == 0:
                process_axiom(op, effect)
    return operator_from_literal, [op for op in operators if id(op) in reachable_operators]


##################################################


def instantiate_domain(task, prune_static=True):
    fluent_predicates = get_fluents(task)
    is_static = lambda a: isinstance(a, pddl.Atom) and (a.predicate not in fluent_predicates)

    fluent_facts = MockSet(lambda a: not prune_static or not is_static(a))
    init_facts = set(task.init)
    function_assignments = get_function_assignments(task)
    type_to_objects = instantiate.get_objects_by_type(task.objects, task.types)

    constants_from_predicate = defaultdict(set)
    for action in task.actions + task.axioms:
        for atom in filter(is_static, get_literals(get_precondition(action))):
            constants = tuple((i, a) for i, a in enumerate(atom.args) if not is_parameter(a))
            constants_from_predicate[atom.predicate].add(constants)

    predicate_to_atoms = defaultdict(set)
    args_from_predicate = defaultdict(set)
    for atom in filter(is_static, task.init):  # TODO: compute which predicates might involve constants
        predicate_to_atoms[atom.predicate].add(atom)
        args_from_predicate[atom.predicate].add(atom.args)
        for constants in constants_from_predicate[atom.predicate]:
            if all(atom.args[i] == o for i, o in constants):
                args_from_predicate[atom.predicate, constants].add(atom.args)

    instantiated_actions = []
    for action in task.actions:
        for variable_mapping in instantiate_condition(action, is_static, args_from_predicate):
            inst_action = action.instantiate(
                variable_mapping,
                init_facts,
                function_assignments,
                fluent_facts,
                type_to_objects,
                task.use_min_cost_metric,
                predicate_to_atoms,
            )
            if inst_action:
                instantiated_actions.append(inst_action)
    instantiated_axioms = []
    for axiom in task.axioms:
        for variable_mapping in instantiate_condition(axiom, is_static, args_from_predicate):
            inst_axiom = axiom.instantiate(variable_mapping, init_facts, fluent_facts)
            if inst_axiom:
                instantiated_axioms.append(inst_axiom)

    reachable_facts, reachable_operators = get_achieving_axioms(init_facts, instantiated_actions + instantiated_axioms)
    atoms = {atom.positive() for atom in (init_facts | set(reachable_facts)) if isinstance(atom, pddl.Literal)}
    relaxed_reachable = all(
        literal_holds(init_facts, goal) or goal in reachable_facts for goal in instantiate_goal(task.goal)
    )
    reachable_actions = [action for action in reachable_operators if isinstance(action, pddl.PropositionalAction)]
    reachable_axioms = [axiom for axiom in reachable_operators if isinstance(axiom, pddl.PropositionalAxiom)]
    return relaxed_reachable, atoms, reachable_actions, reachable_axioms


##################################################


def dump_instantiated(instantiated):
    logger.info(
        'Instantiated frequencies:\n'
        f'Atoms: {str_from_object(Counter(atom.predicate for atom in instantiated.atoms))}\n'
        f'Actions: {str_from_object(Counter(action.action.name for action in instantiated.actions))}\n'
        f'Axioms: {str_from_object(Counter(axiom.axiom.name for axiom in instantiated.axioms))}'
    )


def instantiate_task(task, check_infeasible=True, use_fd=None, **kwargs):
    """use_fd selects FastDownward's instantiator rather than pddlstream's (by default config.fd_instantiate)."""
    if use_fd is None:
        use_fd = get_config().fd_instantiate
    start_time = time()
    logger.info('')
    normalize.normalize(task)
    if use_fd:
        # TODO: recover relaxed reachability (from model)
        relaxed_reachable, atoms, actions, axioms, reachable_action_params = instantiate.explore(task)
    else:
        relaxed_reachable, atoms, actions, axioms = instantiate_domain(task, **kwargs)
        reachable_action_params = get_reachable_action_params(actions)
    logger.info(f'Infeasible: {not relaxed_reachable}')
    logger.info(f'Instantiation time: {elapsed_time(start_time):.3f}s')
    if check_infeasible and not relaxed_reachable:
        return None
    goal_list = instantiate_goal(task.goal)
    instantiated = InstantiatedTask(task, atoms, actions, axioms, reachable_action_params, goal_list)
    dump_instantiated(instantiated)
    return instantiated


##################################################


def sas_from_instantiated(instantiated_task):
    start_time = time()
    logger.info('')

    if not instantiated_task:
        return translate.unsolvable_sas_task('No relaxed solution')
    task, atoms, actions, axioms, reachable_action_params, goal_list = instantiated_task

    # TODO: option to skip and just use binary variables
    with timers.timing('Computing fact groups', block=True):
        groups, mutex_groups, translation_key = fact_groups.compute_groups(task, atoms, reachable_action_params)

    with timers.timing('Building STRIPS to SAS dictionary'):
        ranges, strips_to_sas = translate.strips_to_sas_dictionary(groups, assert_partial=options.use_partial_encoding)

    with timers.timing('Building dictionary for full mutex groups'):
        mutex_ranges, mutex_dict = translate.strips_to_sas_dictionary(mutex_groups, assert_partial=False)

    if options.add_implied_preconditions:
        with timers.timing('Building implied facts dictionary...'):
            implied_facts = translate.build_implied_facts(strips_to_sas, groups, mutex_groups)
    else:
        implied_facts = {}

    with timers.timing('Building mutex information', block=True):
        mutex_key = translate.build_mutex_key(strips_to_sas, mutex_groups)

    with timers.timing('Translating task', block=True):
        sas_task = translate.translate_task(
            strips_to_sas,
            ranges,
            translation_key,
            mutex_dict,
            mutex_ranges,
            mutex_key,
            task.init,
            goal_list,
            actions,
            axioms,
            task.use_min_cost_metric,
            implied_facts,
        )

    if options.filter_unreachable_facts:
        with timers.timing('Detecting unreachable propositions', block=True):
            try:
                simplify.filter_unreachable_propositions(sas_task)
            except simplify.Impossible:
                return translate.unsolvable_sas_task('Simplified to trivially false goal')
            except simplify.TriviallySolvable:
                return translate.solvable_sas_task('Simplified to empty goal')

    if options.reorder_variables or options.filter_unimportant_vars:
        with timers.timing('Reordering and filtering variables', block=True):
            variable_order.find_and_apply_variable_order(
                sas_task, options.reorder_variables, options.filter_unimportant_vars
            )

    translate.dump_statistics(sas_task)
    logger.info(f'Translation time: {elapsed_time(start_time):.3f}s')
    return sas_task


##################################################


def sas_from_pddl(task, debug=False):
    with Verbose(debug):
        instantiated = instantiate_task(task)
        sas_task = sas_from_instantiated(instantiated)
        sas_task.metric = task.use_min_cost_metric  # TODO: are these sometimes not equal?
    return sas_task


def translate_and_write_pddl(domain_pddl, problem_pddl, temp_dir, verbose):
    domain = parse_sequential_domain(domain_pddl)
    problem = parse_problem(domain, problem_pddl)
    task = task_from_domain_problem(domain, problem, add_identical=False)
    sas_task = sas_from_pddl(task)
    write_sas_task(sas_task, temp_dir)
    return task


def convert_instantiated(instantiated_task, verbose=False):
    task, atoms, actions, axioms, reachable_action_params, goal_list = instantiated_task
    normalize.normalize(task)

    with Verbose(verbose):
        axioms, axiom_layers = axiom_rules.handle_axioms(actions, axioms, goal_list, options.layer_strategy)
    init = task.init
    return InstantiatedTask(task, init, actions, axioms, reachable_action_params, goal_list)  # init instead of atoms
