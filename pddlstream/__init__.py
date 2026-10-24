"""PDDLStream: planning with symbolic planners and blackbox samplers (https://github.com/caelan/pddlstream).

The names that most programs need are available from the package itself, for example
`from pddlstream import solve, PDDLProblem, And, StreamInfo, from_gen_fn`. Each is imported when first used, so
`import pddlstream` stays cheap.
"""

import importlib

_MODULES = {
    'pddlstream.algorithms.meta': ['solve', 'create_parser'],
    'pddlstream.algorithms.focused': ['solve_adaptive', 'solve_binding', 'solve_focused'],
    'pddlstream.algorithms.incremental': ['solve_incremental'],
    'pddlstream.algorithms.constraints': ['PlanConstraints', 'OrderedSkeleton', 'WILD'],
    'pddlstream.algorithms.explain': ['explain'],
    'pddlstream.algorithms.model': ['describe'],
    'pddlstream.algorithms.session': ['PlannerSession', 'WorldPatch'],
    'pddlstream.algorithms.validate': ['validate_solution', 'verify_certificate'],
    'pddlstream.core.config': ['SolverConfig'],
    'pddlstream.core.problem': [
        'PDDLProblem',
        'Solution',
        'Certificate',
        'SolveResult',
        'Status',
        'Incumbent',
        'Action',
        'is_plan',
        'print_solution',
    ],
    'pddlstream.core.syntax': ['And', 'Or', 'Not', 'Equal', 'Exists', 'ForAll', 'Imply', 'Minimize', 'Output'],
    'pddlstream.core.utils': ['INF', 'read', 'get_file_path'],
    'pddlstream.language.executor': ['ThreadExecutor', 'ProcessExecutor'],
    'pddlstream.language.external': ['DEBUG', 'SHARED_DEBUG'],
    'pddlstream.language.function': ['FunctionInfo', 'PredicateInfo'],
    'pddlstream.language.generator': [
        'from_fn',
        'from_gen',
        'from_gen_fn',
        'from_batch_fn',
        'from_list_fn',
        'from_list_gen_fn',
        'from_sampler',
        'from_test',
        'empty_gen',
        'fn_from_constant',
        'negate_test',
    ],
    'pddlstream.language.stream': ['StreamInfo', 'PartialInputs', 'WildOutput'],
}
_MODULE_FROM_NAME = {name: module for module, names in _MODULES.items() for name in names}
__all__ = sorted(_MODULE_FROM_NAME)


def __getattr__(name):
    if name not in _MODULE_FROM_NAME:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(_MODULE_FROM_NAME[name]), name)
    globals()[name] = value  # Later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
