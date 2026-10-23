from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.common import evaluations_from_init
from pddlstream.algorithms.downward import (
    apply_action,
    evaluation_from_fd,
    get_action_instances,
    get_fluents,
    get_problem,
    task_from_domain_problem,
)
from pddlstream.algorithms.focused import solve_focused
from pddlstream.fastdownward.translator import pddl
from pddlstream.language.constants import (
    AND,
    And,
    PDDLProblem,
    SolveResult,
    Status,
    get_args,
    get_function,
    get_prefix,
    print_solution,
)
from pddlstream.language.conversion import Certificate, Object, transform_plan_args, value_from_evaluation
from pddlstream.utils import INF, SEPARATOR, Verbose, str_from_object


def serialize_goal(goal):
    if get_prefix(goal) == AND:
        return get_args(goal)
    return [goal]


def partition_facts(domain, facts):
    fluents = get_fluents(domain)
    static_facts = []
    fluent_facts = []
    for fact in facts:
        if get_prefix(get_function(fact)).lower() in fluents:
            fluent_facts.append(fact)
        else:
            static_facts.append(fact)
    return static_facts, fluent_facts


def apply_actions(domain, state, plan, unit_costs=False):

    # Goal serialization just assumes the tail of the plan includes an abstract action to achieve each condition
    static_state, _ = partition_facts(domain, state)
    print('Static:', static_state)
    # TODO: might need properties that involve an object that aren't useful yet
    evaluations = evaluations_from_init(state)
    goal_exp = None
    problem = get_problem(evaluations, goal_exp, domain, unit_costs)
    task = task_from_domain_problem(domain, problem)
    task.init = set(task.init)
    for instance in get_action_instances(task, transform_plan_args(plan, Object.from_value)):
        apply_action(task.init, instance)
    fluents = get_fluents(domain)
    fluent_state = [
        value_from_evaluation(evaluation_from_fd(atom))
        for atom in task.init
        if isinstance(atom, pddl.Atom) and (atom.predicate in fluents)
    ]
    print('Fluent:', fluent_state)
    state = static_state + fluent_state
    return state


##################################################


def solve_serialized(
    initial_problem, stream_info={}, unit_costs=False, unit_efforts=False, verbose=True, retain_facts=True, **kwargs
):
    # TODO: be careful of CanMove deadends
    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = initial_problem
    _, _, domain, streams = parse_problem(
        initial_problem, stream_info, constraints=None, unit_costs=unit_costs, unit_efforts=unit_efforts
    )
    static_init, _ = partition_facts(domain, init)  # might not be able to reprove static_int
    global_plan = []
    global_cost = 0
    state = list(init)
    goals = serialize_goal(goal)
    # TODO: instead just track how the true init updates
    for i in range(len(goals)):
        # TODO: option in algorithms to pass in existing facts
        for stream in streams:
            stream.reset()
        goal = And(*goals[: i + 1])
        print('Goal:', str_from_object(goal))
        # No strict need to reuse streams because generator functions
        local_problem = PDDLProblem(domain_pddl, constant_map, streams, None, state, goal)
        with Verbose(verbose):
            solution = solve_focused(
                local_problem,
                stream_info=stream_info,
                unit_costs=unit_costs,
                unit_efforts=unit_efforts,
                verbose=True,
                **kwargs,
            )
        print_solution(solution)
        local_plan, local_cost, local_certificate = solution
        if local_plan is None:
            # TODO: replan upon failure
            global_certificate = Certificate(all_facts={}, preimage_facts=None)
            return SolveResult(None, INF, global_certificate, status=Status.FAILED)

        if retain_facts:
            state = local_certificate.all_facts
        else:
            _, fluent_facts = partition_facts(domain, state)
            state = static_init + fluent_facts + local_certificate.preimage_facts  # TODO: include functions
        # TODO: indicate when each fact is used
        # TODO: record failed facts
        global_plan.extend(local_plan)  # TODO: compute preimage of the executed plan
        global_cost += local_cost

        static_state, _ = partition_facts(domain, state)
        print('Static:', static_state)
        state = apply_actions(domain, state, local_plan, unit_costs=unit_costs)
        print(SEPARATOR)
        # TODO: could also just test the goal here
        # TODO: constrain future plan skeletons

    global_certificate = Certificate(all_facts={}, preimage_facts=None)
    return SolveResult(global_plan, global_cost, global_certificate, status=Status.SUCCEEDED)
