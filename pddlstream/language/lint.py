"""Lint: problems with a domain and its stream declarations, with where they are declared when known."""

import re
from collections import Counter
from dataclasses import dataclass

from pddlstream.core.syntax import get_args, get_prefix, is_parameter

ERROR, WARNING = 'error', 'warning'
# Top-level declarations whose line lint reports; each pattern's group is the declared name
DECLARATIONS = [
    r'\(:action\s+([^\s()]+)',
    r'\(:derived\s+\(\s*([^\s()]+)',
    r'\(:stream\s+([^\s()]+)',
    r'\(:function\s+\(\s*([^\s()]+)',
    r'\(:predicate\s+\(\s*([^\s()]+)',
    r'\(:optimizer\s+([^\s()]+)',
]


class CapabilityError(NotImplementedError):
    """A problem uses a feature that the chosen algorithm does not support."""


@dataclass(frozen=True)
class Source:
    """Where a declaration is: the domain or a stream file (by index), and the line (from 1)."""

    file: str
    line: int


@dataclass(frozen=True)
class Issue:
    """A problem with a declaration that lint found."""

    code: str
    severity: str  # 'error' or 'warning'
    message: str
    subject: str  # The action, stream or predicate concerned
    source: Source | None = None


def find_sources(texts):
    """The Source of each declaration in the PDDL texts (by name), found by scanning for top-level forms."""
    sources = {}
    for file, text in texts.items():
        if not isinstance(text, str):
            continue
        for pattern in DECLARATIONS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count('\n', 0, match.start()) + 1
                sources.setdefault(match.group(1).lower(), Source(file, line))
    return sources


def lint(domain, externals, sources={}):
    """The Issues of a parsed domain and its streams."""
    issues = []
    for action in domain.actions + domain.axioms:
        for parameter, num in Counter(action.parameters).items():
            if num != 1:
                message = f'Parameter [{parameter.name}] for action [{action.name}] is not unique'
                issues.append(Issue('duplicate-parameter', ERROR, message, action.name, sources.get(action.name)))
    undeclared = set()
    for external in externals:
        facts = list(external.domain) + list(getattr(external, 'certified', []))
        for fact in facts:
            name = get_prefix(fact)
            if name not in domain.predicate_dict:
                undeclared.add(name)
            elif len(get_args(fact)) != domain.predicate_dict[name].get_arity():
                message = f'predicate used with wrong arity in stream [{external.name}]: {fact}'
                issues.append(Issue('wrong-arity', WARNING, message, external.name, sources.get(external.name)))
        covered = {arg for fact in external.domain for arg in get_args(fact) if is_parameter(arg)}
        for parameter in set(external.inputs) - covered:
            message = f'Input [{parameter}] for stream [{external.name}] is not covered by a domain condition'
            issues.append(Issue('uncovered-input', WARNING, message, external.name, sources.get(external.name)))
        certified = {arg for fact in getattr(external, 'certified', []) for arg in get_args(fact)}
        for parameter in set(getattr(external, 'outputs', [])) - certified:
            message = f'Output [{parameter}] for stream [{external.name}] is not covered by a certified condition'
            issues.append(Issue('uncovered-output', WARNING, message, external.name, sources.get(external.name)))
    if undeclared:
        message = f'Undeclared predicates: {sorted(undeclared)}'
        issues.append(Issue('undeclared-predicate', WARNING, message, ', '.join(sorted(undeclared))))
    return issues
