import os
import re
import sys
import subprocess
from collections import namedtuple, defaultdict
from time import time

from pddlstream.language.constants import EQ, NOT, Head, Evaluation, get_prefix, get_args, OBJECT, TOTAL_COST, Action, Not
from pddlstream.language.conversion import is_atom, is_negated_atom, objects_from_evaluations, pddl_from_object, \
    pddl_list_from_expression, obj_from_pddl
from pddlstream.utils import read, write, INF, clear_dir, get_file_path, MockSet, find_unique, int_ceil, \
    safe_remove, safe_zip, elapsed_time, get_clock, SEARCH_WORK

##################################################

filepath = os.path.abspath(__file__)
if ' ' in filepath:
    raise RuntimeError('The path to pddlstream cannot include spaces')

def find_build(fd_path):
    for release in ['release', 'release64', 'release32']:  # TODO: list the directory
        path = os.path.join(fd_path, f'builds/{release}/')
        if os.path.exists(path):
            return path
    # TODO: could also just automatically compile
    raise RuntimeError(f'FastDownward is not compiled in {fd_path}. Run [.../pddlstream$ pixi run setup] '
                       '(pddlstream must be installed in editable mode)')

# TODO: check at runtime so users can use utils without FD
FD_PATH = get_file_path(__file__, '../../downward/')
TRANSLATE_PATH = os.path.join(find_build(FD_PATH), 'bin/translate')
FD_BIN = os.path.join(find_build(FD_PATH), 'bin')

DOMAIN_INPUT = 'domain.pddl'
PROBLEM_INPUT = 'problem.pddl'
TRANSLATE_FLAGS = []
original_argv = sys.argv[:]
sys.argv = sys.argv[:1] + TRANSLATE_FLAGS + [DOMAIN_INPUT, PROBLEM_INPUT]
sys.path.append(TRANSLATE_PATH)
# TODO: max translate time

import pddl.f_expression
import pddl
import instantiate
import pddl_parser.lisp_parser
import normalize
import pddl_parser
from pddl_parser.parsing_functions import parse_domain_pddl, parse_task_pddl, \
    parse_condition, check_for_duplicates
sys.argv = original_argv

def hash_conditions_by_class_name():
    # The translator hashes conditions as hash((self.__class__, ...)), and classes hash by memory address, so sets of
    # literals (which both the translator and pddlstream iterate over) were ordered differently on every run.
    # These are the translator's constructors, except that they hash the class name.
    def condition_init(self, parts):
        unique_parts = []
        for part in parts:
            if part not in unique_parts:
                unique_parts.append(part)
        self.parts = tuple(unique_parts)
        self.hash = hash((self.__class__.__name__, self.parts))

    def constant_condition_init(self):
        self.hash = hash(self.__class__.__name__)

    def quantified_condition_init(self, parameters, parts):
        self.parameters = tuple(parameters)
        self.parts = tuple(parts)
        self.hash = hash((self.__class__.__name__, self.parameters, self.parts))

    def literal_init(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
        self.hash = hash((self.__class__.__name__, self.predicate, self.args))

    def numeric_expression_init(self, symbol, args):
        self.symbol = symbol
        self.args = tuple(args)
        self.hash = hash((self.__class__.__name__, self.symbol, self.args))

    pddl.conditions.Condition.__init__ = condition_init
    pddl.conditions.ConstantCondition.__init__ = constant_condition_init
    pddl.conditions.QuantifiedCondition.__init__ = quantified_condition_init
    pddl.conditions.Literal.__init__ = literal_init
    pddl.f_expression.PrimitiveNumericExpression.__init__ = numeric_expression_init

hash_conditions_by_class_name()

TEMP_DIR = 'temp/'
TRANSLATE_OUTPUT = 'output.sas'
SEARCH_OUTPUT = 'sas_plan'
SEARCH_COMMAND = 'downward --internal-plan-file {} {} < {}'
INFINITY = 'infinity'
GOAL_NAME = '@goal' # @goal-reachable

INTERNAL_AXIOM = 'new-axiom' # @0
IDENTICAL = "identical" # lowercase is critical (!= instead?)
##################################################

# TODO: cost_type=PLUSONE can lead to suboptimality but often doesn't in practice due to COST_SCALE
# TODO: modify parsing_functions to support multiple costs

# bound (int): exclusive depth bound on g-values. Cutoffs are always performed according to the real cost.
# (i.e. solutions must be strictly better than the bound)

# TODO: move toward using this style

SEARCH_OPTIONS = {
    # See FastDownward's documentation for more configurations
    # http://www.fast-downward.org/Doc/Evaluator
    # http://www.fast-downward.org/Doc/SearchEngine

    # Optimal (when cost_type=NORMAL)
    'dijkstra': '--heuristic "h=blind(transform=adapt_costs(cost_type=PLUSONE))" '
                '--search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'max-astar': '--heuristic "h=hmax(transform=adapt_costs(cost_type=PLUSONE))"'
                 ' --search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"', # cost_type=PLUSONE
    'lmcut-astar': '--heuristic "h=lmcut(transform=adapt_costs(cost_type=PLUSONE))"'
                 ' --search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"',

    # Suboptimal
    'ff-astar': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                '--search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'ff-eager': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                '--search "eager_greedy([h],max_time=%s,bound=%s)"',
    'ff-eager-pref': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                     '--search "eager_greedy([h],preferred=[h],max_time=%s,bound=%s)"',
    'ff-lazy': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
               '--search "lazy_greedy([h],preferred=[h],max_time=%s,bound=%s)"',
    'goal-lazy': '--heuristic "h=goalcount(transform=no_transform())" '
                 '--search "lazy_greedy([h],randomize_successors=True,max_time=%s,bound=%s)"',
    'add-random-lazy': '--heuristic "h=add(transform=adapt_costs(cost_type=PLUSONE))" '
                       '--search "lazy_greedy([h],randomize_successors=True,max_time=%s,bound=%s)"',

    'ff-eager-tiebreak': '--heuristic "h=ff(transform=no_transform())" '
                         '--search "eager(tiebreaking([h, g()]),reopen_closed=false,'
                         'cost_type=PLUSONE,max_time=%s,bound=%s, f_eval=sum([g(), h]))"', # preferred=[h],
    'ff-lazy-tiebreak': '--heuristic "h=ff(transform=no_transform())" '
                        '--search "lazy(tiebreaking([h, g()]),reopen_closed=false,'
                        'randomize_successors=True,cost_type=PLUSONE,max_time=%s,bound=%s)"',  # preferred=[h],
    # TODO: eagerly evaluate goal count but lazily compute relaxed plan

    'ff-ehc': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
              '--search "ehc(h,preferred=[h],preferred_usage=RANK_PREFERRED_FIRST,'
              'cost_type=PLUSONE,max_time=%s,bound=%s)"',
    # The key difference is that ehc resets the open list upon finding an improvement

    # TODO: iterated search
}
# TODO: do I want to sort operators in FD hill-climbing search?
# TODO: greedily prioritize operators with less cost. Useful when prioritizing actions that have no stream cost

for w in range(1, 1+5):
    SEARCH_OPTIONS.update({
        # TODO: specify whether lazy or eager
        f'ff-wastar{w}': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                  f'--search "lazy_wastar([h],preferred=[h],reopen_closed=true,boost=100,w={w},'
                  'randomize_successors=false,preferred_successors_first=true,random_seed=-1,'
                  'cost_type=PLUSONE,max_time=%s,bound=%s)"',

        f'cea-wastar{w}': '--heuristic "h=cea(transform=adapt_costs(cost_type=PLUSONE))" '
                   f'--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=1000,w={w},'
                   'randomize_successors=false,preferred_successors_first=true,random_seed=-1,'
                   'cost_type=PLUSONE,max_time=%s,bound=%s)"',

        # TODO: eager_wastar
        # http://www.fast-downward.org/Doc/SearchEngine#Eager_weighted_A.2A_search
        f'ff-astar{w}': '--evaluator "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                                   f'--search "eager(alt([single(sum([g(), weight(h,{w})])),'
                                   f'single(sum([g(),weight(h,{w})]),pref_only=true)]),'
                                   'preferred=[h],cost_type=PLUSONE,max_time=%s,bound=%s)"',
    })

# TODO: throw a warning if max_planner_time is met
DEFAULT_MAX_TIME = 30 # INF
DEFAULT_GREEDY_PLANNER = 'ff-astar2'
DEFAULT_PLANNER = DEFAULT_GREEDY_PLANNER

##################################################

# WARNING: overflow on h^add! Costs clamped to 100000000
MAX_FD_COST = 1e8

def get_cost_scale():
    return pddl.f_expression.COST_SCALE

def set_cost_scale(cost_scale):
    pddl.f_expression.COST_SCALE = cost_scale

def convert_value(value):
    if value == INF:
        return INFINITY
    return int_ceil(value)

def scale_cost(cost):
    if cost == INF:
        return INF
    return int_ceil(get_cost_scale() * float(cost))

set_cost_scale(cost_scale=1e3) # TODO: make unit costs be equivalent to cost scale = 0

##################################################

def parse_lisp(lisp):
    return pddl_parser.lisp_parser.parse_nested_list(lisp.splitlines())

# TODO: dynamically generate type_dict and predicate_dict
Domain = namedtuple('Domain', ['name', 'requirements', 'types', 'type_dict', 'constants',
                               'predicates', 'predicate_dict', 'functions', 'actions', 'axioms', 'pddl'])

def parse_sequential_domain(domain_pddl):
    if isinstance(domain_pddl, Domain):
        return domain_pddl
    args = list(parse_domain_pddl(parse_lisp(domain_pddl))) + [domain_pddl]
    domain = Domain(*args)
    return domain

Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements',
                                 'objects', 'init', 'goal', 'use_metric', 'pddl'])

def parse_problem(domain, problem_pddl):
    if isinstance(problem_pddl, Problem):
        return problem_pddl
    args = list(parse_task_pddl(parse_lisp(problem_pddl), domain.type_dict, domain.predicate_dict)) + [problem_pddl]
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
    assert(isinstance(fd, pddl.Literal))
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
    return parse_condition(pddl_list_from_expression(goal_exp),
                           domain.type_dict, domain.predicate_dict).simplified()

def get_problem(evaluations, goal_exp, domain, unit_costs=False):
    objects = objects_from_evaluations(evaluations)
    typed_objects = list({make_object(pddl_from_object(obj)) for obj in objects} - set(domain.constants))
    # TODO: this doesn't include =
    init = fd_from_evaluations(evaluations)
    goal = pddl.Truth() if goal_exp is None else parse_goal(goal_exp, domain)
    problem_pddl = None
    write_pddl(domain.pddl, problem_pddl, temp_dir=TEMP_DIR)
    return Problem(task_name=domain.name, task_domain_name=domain.name,
                   objects=sorted(typed_objects, key=lambda o: o.name),
                   task_requirements=pddl.tasks.Requirements([]), init=init, goal=goal,
                   use_metric=not unit_costs, pddl=problem_pddl)

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
    requirements = pddl.Requirements(sorted(set(domain.requirements.requirements +
                                                task_requirements.requirements)))
    objects = domain.constants + objects
    check_for_duplicates([o.name for o in objects],
        errmsg="error: duplicate object %r",
        finalmsg="please check :constants and :objects definitions")
    init.extend(pddl.Atom(EQ, (obj.name, obj.name)) for obj in objects)
    if add_identical:
        init.extend(get_identical_atoms(objects))

    task = pddl.Task(domain.name, task_name, requirements, domain.types, objects,
                     domain.predicates, domain.functions, init, goal,
                     domain.actions, domain.axioms, use_metric)
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

def run_search(temp_dir, planner=DEFAULT_PLANNER, max_planner_time=DEFAULT_MAX_TIME,
               max_cost=INF, debug=False):
    """
    Runs FastDownward's search phase on translated SAS+ problem TRANSLATE_OUTPUT
    :param temp_dir: the directory for temporary FastDownward input and output files
    :param planner: a keyword for the FastDownward search configuration in SEARCH_OPTIONS
    :param max_planner_time: the maximum runtime of FastDownward
    :param max_cost: the maximum FastDownward plan cost
    :param debug: If True, print the FastDownward search output
    :return: a tuple (plan, cost) where plan is a sequence of PDDL actions
        (or None) and cost is the cost of the plan (INF if no plan)
    """
    max_time = convert_value(max_planner_time)
    max_cost = convert_value(scale_cost(max_cost))
    start_time = time()
    get_clock().charge(SEARCH_WORK)
    search = os.path.abspath(os.path.join(FD_BIN, SEARCH_COMMAND))
    planner_config = SEARCH_OPTIONS[planner] % (max_time, max_cost)
    temp_dir = os.path.abspath(temp_dir)
    command = search.format(os.path.join(temp_dir, SEARCH_OUTPUT), planner_config,
                            os.path.join(temp_dir, TRANSLATE_OUTPUT))
    if debug:
        print('Search command:', command)


    temp_path = os.path.join(os.getcwd(), TEMP_DIR) # TODO: temp dir?
    for filename in os.listdir(temp_path):
        if filename.startswith(SEARCH_OUTPUT):
            safe_remove(os.path.join(temp_path, filename))

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, cwd=None, close_fds=True)
    output, error = proc.communicate()

    if debug:
        print(output.decode(encoding='UTF-8')[:-1])
        print(f'Search runtime: {elapsed_time(start_time):.3f}')
    plan_files = sorted(f for f in os.listdir(temp_path) if f.startswith(SEARCH_OUTPUT))
    print('Plans:', plan_files)
    return parse_solutions(temp_path, plan_files)

##################################################

def parse_action(line):
    entries = line.strip('( )').split(' ')
    name = entries[0]
    args = tuple(entries[1:])
    return Action(name, args)

def parse_solution(solution):
    cost = INF
    if solution is None:
        return None, cost
    cost_regex = r'cost\s*=\s*(\d+)'
    matches = re.findall(cost_regex, solution)
    if matches:
        cost = float(matches[0]) / get_cost_scale()
    # TODO: recover the actual cost of the plan from the evaluations
    lines = solution.split('\n')[:-2]  # Last line is newline, second to last is cost
    plan = list(map(parse_action, lines))
    return plan, cost

def parse_solutions(temp_path, plan_files):
    # TODO: return multiple solutions for focused
    best_plan, best_cost = None, INF
    for plan_file in plan_files:
        solution = read(os.path.join(temp_path, plan_file))
        plan, cost = parse_solution(solution)
        if cost < best_cost:
            best_plan, best_cost = plan, cost
    return best_plan, best_cost

def write_pddl(domain_pddl=None, problem_pddl=None, temp_dir=TEMP_DIR):
    clear_dir(temp_dir)
    domain_path = os.path.join(temp_dir, DOMAIN_INPUT)
    if domain_pddl is not None:
        write(domain_path, domain_pddl)
    problem_path = os.path.join(temp_dir, PROBLEM_INPUT)
    if problem_pddl is not None:
        write(problem_path, problem_pddl)
    return domain_path, problem_path

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
        return [(cond, effect.negate()) for cond, effect in operator.del_effects] + \
               [(cond, effect) for cond, effect in operator.add_effects]
    elif isinstance(operator, pddl.PropositionalAxiom):
        return [([], operator.effect)]
    raise ValueError(operator)

def apply_action(state, action):
    assert(isinstance(action, pddl.PropositionalAction))
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
    return {f.fluent: f.expression for f in task.init
            if isinstance(f, pddl.f_expression.FunctionAssignment)}

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
        action = find_unique(lambda a: a.name == name, task.actions)
        args = list(map(pddl_from_object, objects))
        variable_mapping = {p.name: a for p, a in safe_zip(action.parameters, args)}
        instance = action.instantiate(variable_mapping, init_facts, function_assignments, fluent_facts,
                                      type_to_objects, task.use_min_cost_metric, predicate_to_atoms)
        assert (instance is not None)
        action_instances.append(instance)
    return action_instances

##################################################

def add_preimage_condition(condition, preimage, i):
    for literal in condition:
        preimage.setdefault(literal, set()).add(i)


def add_preimage_effect(effect, preimage):
    preimage.pop(effect, None)


def action_preimage(action, preimage, i):
    for conditions, effect in (action.add_effects + action.del_effects):
        assert(not conditions)
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
    return [pddl.Effect(parameters=[], condition=pddl.Truth(),
                        literal=fd_from_fact(fact)) for fact in effects]

def make_cost(cost):
    if cost is None:
        return cost
    fluent = pddl.PrimitiveNumericExpression(symbol=TOTAL_COST, args=[])
    try:
        expression = pddl.NumericConstant(cost)
    except TypeError:
        expression = pddl.PrimitiveNumericExpression(
            symbol=get_prefix(cost), args=list(map(pddl_from_object, get_args(cost))))
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
    return pddl.Action(name=name,
                       parameters=make_parameters(parameters),
                       num_external_parameters=len(parameters),
                       precondition=make_preconditions(preconditions),
                       effects=make_effects(effects),
                       cost=make_cost(cost))


def make_axiom(parameters, preconditions, derived):
    predicate = get_prefix(derived)
    external_parameters = list(get_args(derived))
    internal_parameters = [p for p in parameters if p not in external_parameters]
    parameters = external_parameters + internal_parameters
    return pddl.Axiom(name=predicate,
                      parameters=make_parameters(parameters),
                      num_external_parameters=len(external_parameters),
                      condition=make_preconditions(preconditions))


def make_domain(constants=[], predicates=[], functions=[], actions=[], axioms=[]):
    types = [pddl.Type(OBJECT)]
    pddl_parser.parsing_functions.set_supertypes(types)
    return Domain(name='', requirements=pddl.Requirements([]),
             types=types, type_dict={ty.name: ty for ty in types}, constants=constants,
             predicates=predicates, predicate_dict={p.name: p for p in predicates},
             functions=functions, actions=actions, axioms=axioms, pddl=None)


def pddl_from_instance(instance):
    action = instance.action
    args = [instance.var_mapping[p.name]
            for p in action.parameters[:action.num_external_parameters]]
    return Action(action.name, args)
