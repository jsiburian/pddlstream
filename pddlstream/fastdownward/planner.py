"""Runs FastDownward's search on a translated (SAS+) task and reads back its plans."""

import contextlib
import logging
import os
import re
import shlex
import subprocess
import tempfile
from time import time

from pddlstream.core.context import SEARCH_WORK, get_clock
from pddlstream.core.problem import Action
from pddlstream.core.utils import INF, elapsed_time, ensure_dir, read, safe_remove, safe_rm_dir, write
from pddlstream.fastdownward.translator import convert_value, find_build, get_cost_scale, scale_cost

logger = logging.getLogger(__name__)

# Debugging runs keep FastDownward's input and output files here for inspection
TEMP_DIR = 'temp/'
DOMAIN_INPUT = 'domain.pddl'
PROBLEM_INPUT = 'problem.pddl'
TRANSLATE_OUTPUT = 'output.sas'
SEARCH_OUTPUT = 'sas_plan'

# TODO: cost_type=PLUSONE can lead to suboptimality but often doesn't in practice due to COST_SCALE
# TODO: modify parsing_functions to support multiple costs

# bound (int): exclusive depth bound on g-values. Cutoffs are always performed according to the real cost.
# (i.e. solutions must be strictly better than the bound)

SEARCH_OPTIONS = {
    # See FastDownward's documentation for more configurations
    # http://www.fast-downward.org/Doc/Evaluator
    # http://www.fast-downward.org/Doc/SearchEngine
    # Optimal (when cost_type=NORMAL)
    'dijkstra': '--heuristic "h=blind(transform=adapt_costs(cost_type=PLUSONE))" '
    '--search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'max-astar': '--heuristic "h=hmax(transform=adapt_costs(cost_type=PLUSONE))"'
    ' --search "astar(h,cost_type=PLUSONE,max_time=%s,bound=%s)"',  # cost_type=PLUSONE
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
    'cost_type=PLUSONE,max_time=%s,bound=%s, f_eval=sum([g(), h]))"',  # preferred=[h],
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

for w in range(1, 1 + 5):
    SEARCH_OPTIONS.update(
        {
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
        }
    )

# TODO: throw a warning if max_planner_time is met
DEFAULT_MAX_TIME = 30  # INF
DEFAULT_GREEDY_PLANNER = 'ff-astar2'
DEFAULT_PLANNER = DEFAULT_GREEDY_PLANNER

##################################################


@contextlib.contextmanager
def search_workspace(temp_dir=None, clean=False, debug=False):
    """Yields the directory for FastDownward's files: temp_dir if given, ./temp/ when debugging, and otherwise a
    private directory that is deleted afterwards, so that concurrent solves never share files.
    clean=True also deletes a given or debugging directory afterwards."""
    if (temp_dir is None) and not debug:
        with tempfile.TemporaryDirectory(prefix='pddlstream-') as directory:
            yield directory
        return
    directory = TEMP_DIR if temp_dir is None else temp_dir
    ensure_dir(os.path.join(directory, ''))
    try:
        yield directory
    finally:
        if clean:
            safe_rm_dir(directory)


def write_pddl(domain_pddl=None, problem_pddl=None, temp_dir=TEMP_DIR):
    ensure_dir(os.path.join(temp_dir, ''))
    domain_path = os.path.join(temp_dir, DOMAIN_INPUT)
    if domain_pddl is not None:
        write(domain_path, domain_pddl)
    problem_path = os.path.join(temp_dir, PROBLEM_INPUT)
    if problem_pddl is not None:
        write(problem_path, problem_pddl)
    return domain_path, problem_path


def write_sas_task(sas_task, temp_dir):
    translate_path = os.path.join(temp_dir, TRANSLATE_OUTPUT)
    safe_remove(translate_path)
    ensure_dir(translate_path)
    with open(translate_path, 'w') as output_file:
        sas_task.output(output_file)
    return translate_path


def run_search(temp_dir, planner=DEFAULT_PLANNER, max_planner_time=DEFAULT_MAX_TIME, max_cost=INF, debug=False):
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
    temp_dir = os.path.abspath(temp_dir)
    for filename in os.listdir(temp_dir):  # Plans left by an earlier search in this directory
        if filename.startswith(SEARCH_OUTPUT):
            safe_remove(os.path.join(temp_dir, filename))
    planner_config = SEARCH_OPTIONS[planner] % (max_time, max_cost)
    command = [
        os.path.join(find_build(), 'bin', 'downward'),
        '--internal-plan-file',
        os.path.join(temp_dir, SEARCH_OUTPUT),
        *shlex.split(planner_config),
    ]
    if debug:
        logger.info(f'Search command: {shlex.join(command)}')

    with open(os.path.join(temp_dir, TRANSLATE_OUTPUT)) as sas_file:
        output = subprocess.run(command, stdin=sas_file, stdout=subprocess.PIPE).stdout

    if debug:
        logger.info(output.decode(encoding='UTF-8')[:-1])
        logger.info(f'Search runtime: {elapsed_time(start_time):.3f}')
    plan_files = sorted(f for f in os.listdir(temp_dir) if f.startswith(SEARCH_OUTPUT))
    logger.info(f'Plans: {plan_files}')
    return parse_solutions(temp_dir, plan_files)


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
