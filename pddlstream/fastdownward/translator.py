"""Loads FastDownward's Python translator on first use.

The translator is a directory of top-level modules (pddl, normalize, instantiate, ...) that is importable only
once it is on sys.path, and its options module parses sys.argv when imported. Other modules therefore never
import it directly: `from pddlstream.fastdownward.translator import pddl` gives a stand-in that loads the
translator the first time one of its attributes is used, and from then on behaves exactly like the module.
"""

import functools
import importlib
import os
import sys

from pddlstream.core.utils import INF, int_ceil

FD_PATH_VARIABLE = 'PDDLSTREAM_FD_PATH'
DEFAULT_FD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'downward')
# Imported together while sys.argv holds the arguments the translator's options module expects
TRANSLATOR_MODULES = [
    'pddl',
    'pddl_parser',
    'pddl_parser.lisp_parser',
    'pddl_parser.parsing_functions',
    'axiom_rules',
    'build_model',
    'fact_groups',
    'instantiate',
    'normalize',
    'options',
    'pddl_to_prolog',
    'simplify',
    'timers',
    'translate',
    'variable_order',
]
TRANSLATOR_ARGV = ['domain.pddl', 'problem.pddl']
# The fork scales action costs by this factor and rounds them to integers
DEFAULT_COST_SCALE = 1e3
# WARNING: overflow on h^add! Costs clamped to 100000000
MAX_FD_COST = 1e8
INFINITY = 'infinity'


def get_fd_path():
    """The FastDownward checkout: $PDDLSTREAM_FD_PATH, or the downward submodule next to the pddlstream package."""
    return os.path.normpath(os.environ.get(FD_PATH_VARIABLE) or DEFAULT_FD_PATH)


@functools.cache
def find_build():
    fd_path = get_fd_path()
    for release in ['release', 'release64', 'release32']:
        path = os.path.join(fd_path, 'builds', release)
        if os.path.exists(path):
            return path
    raise RuntimeError(
        f'FastDownward is not compiled in {fd_path}. Run [.../pddlstream$ pixi run setup], '
        f'or point {FD_PATH_VARIABLE} to a compiled FastDownward checkout'
    )


@functools.cache
def load_translator():
    translate_path = os.path.join(find_build(), 'bin', 'translate')
    sys.path.append(translate_path)
    argv = sys.argv
    sys.argv = argv[:1] + TRANSLATOR_ARGV
    try:
        for name in TRANSLATOR_MODULES:
            importlib.import_module(name)
    finally:
        sys.argv = argv
    hash_conditions_by_class_name(sys.modules['pddl'])
    sys.modules['pddl'].f_expression.COST_SCALE = DEFAULT_COST_SCALE
    return translate_path


def hash_conditions_by_class_name(pddl):
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


class LazyModule:
    """Stands in for a translator module until one of its attributes is first used.

    That first use loads the translator, and the stand-in then shares the module's namespace, so later reads and
    writes go straight to the module at the speed of a normal attribute lookup.
    """

    def __init__(self, name):
        self._lazy_name = name

    def __getattr__(self, attr):
        name = vars(self).get('_lazy_name')
        if name is None or attr.startswith('__'):
            raise AttributeError(attr)
        load_translator()
        self.__dict__ = importlib.import_module(name).__dict__
        return getattr(self, attr)


axiom_rules = LazyModule('axiom_rules')
build_model = LazyModule('build_model')
fact_groups = LazyModule('fact_groups')
instantiate = LazyModule('instantiate')
normalize = LazyModule('normalize')
options = LazyModule('options')
pddl = LazyModule('pddl')
pddl_parser = LazyModule('pddl_parser')
pddl_to_prolog = LazyModule('pddl_to_prolog')
simplify = LazyModule('simplify')
timers = LazyModule('timers')
translate = LazyModule('translate')
variable_order = LazyModule('variable_order')

##################################################


def get_cost_scale():
    return pddl.f_expression.COST_SCALE


def set_cost_scale(cost_scale):
    pddl.f_expression.COST_SCALE = cost_scale


def scale_cost(cost):
    if cost == INF:
        return INF
    return int_ceil(get_cost_scale() * float(cost))


def convert_value(value):
    """Formats a time or cost limit for a FastDownward search option."""
    if value == INF:
        return INFINITY
    return int_ceil(value)
