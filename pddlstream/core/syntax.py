"""PDDL expressions as tuples: operators, constructors such as And and Exists, and accessors."""

from collections.abc import Sequence
from typing import Any, NamedTuple

# A fact, i.e. a predicate and its arguments such as ('On', 'b1', 'table'), or an operator and its operands
Formula = tuple[Any, ...]

EQ = '='  # xnor
AND = 'and'
OR = 'or'
NOT = 'not'
EXISTS = 'exists'
FORALL = 'forall'
WHEN = 'when'
IMPLY = 'imply'
MINIMIZE = 'minimize'
MAXIMIZE = 'maximize'
INCREASE = 'increase'
PARAMETER = '?'
TYPE = '-'
OBJECT = 'object'
TOTAL_COST = 'total-cost'  # TotalCost
CONNECTIVES = (AND, OR, NOT, IMPLY)
QUANTIFIERS = (FORALL, EXISTS)
OBJECTIVES = (MINIMIZE, MAXIMIZE, INCREASE)
OPERATORS = CONNECTIVES + QUANTIFIERS + (WHEN,)  # + OBJECTIVES


class Head(NamedTuple):
    """A predicate or function applied to arguments."""

    function: str
    args: tuple


class Evaluation(NamedTuple):
    """A head and its value: True or False for predicates, a number for functions."""

    head: Head
    value: Any


Atom = lambda head: Evaluation(head, True)
NegatedAtom = lambda head: Evaluation(head, False)

##################################################


def Output(*args: Any) -> tuple:
    """A stream's output tuple."""
    return tuple(args)


def And(*expressions: Formula) -> Formula:
    """The conjunction of the formulas (a single formula is returned as is)."""
    if len(expressions) == 1:
        return expressions[0]
    return (AND,) + tuple(expressions)


def Or(*expressions: Formula) -> Formula:
    """The disjunction of the formulas (a single formula is returned as is)."""
    if len(expressions) == 1:
        return expressions[0]
    return (OR,) + tuple(expressions)


def Not(expression: Formula) -> Formula:
    """The negation of a formula."""
    return (NOT, expression)


def Imply(expression1: Formula, expression2: Formula) -> Formula:
    """The implication expression1 -> expression2."""
    return (IMPLY, expression1, expression2)


def Equal(expression1: Any, expression2: Any) -> Formula:
    """Equality, e.g. Equal(('Cost', q1, q2), 1) gives a function its value."""
    return (EQ, expression1, expression2)


def Minimize(expression: Formula) -> Formula:
    """A cost term to minimize, e.g. in an optimizer's constraints."""
    return (MINIMIZE, expression)


def Type(param: str, ty: str) -> tuple:
    return (param, TYPE, ty)


def Exists(args: Sequence[str], expression: Formula) -> Formula:
    """Existential quantification over parameters, e.g. Exists(['?b'], ('On', '?b', 'table'))."""
    return (EXISTS, args, expression)


def ForAll(args: Sequence[str], expression: Formula) -> Formula:
    """Universal quantification over parameters."""
    return (FORALL, args, expression)


##################################################


def get_prefix(expression):
    return expression[0]


def get_args(head):
    return head[1:]


def concatenate(*args):
    output = []
    for arg in args:
        output.extend(arg)
    return tuple(output)


def Fact(predicate, args):
    return (predicate,) + tuple(args)


def is_parameter(expression):
    return isinstance(expression, str) and expression.startswith(PARAMETER)


def get_parameter_name(expression):
    if is_parameter(expression):
        return expression[len(PARAMETER) :]
    return expression


def is_head(expression):
    return get_prefix(expression) not in OPERATORS


##################################################


def get_function(term):
    if get_prefix(term) in (EQ, MINIMIZE, NOT):
        return term[1]
    return term


def partition_facts(facts):
    functions = []
    negated = []
    positive = []
    for fact in facts:
        prefix = get_prefix(fact)
        func = get_function(fact)
        if prefix in (EQ, MINIMIZE):
            functions.append(func)
        elif prefix == NOT:
            negated.append(func)
        else:
            positive.append(func)
    return positive, negated, functions


def is_cost(o):
    return get_prefix(o) == MINIMIZE


def get_costs(objectives):
    return [o for o in objectives if is_cost(o)]


def get_constraints(objectives):
    return [o for o in objectives if not is_cost(o)]


##################################################
