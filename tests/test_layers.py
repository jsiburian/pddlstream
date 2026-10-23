"""The packages form layers, each importing only those before it: core, fastdownward, language, algorithms."""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'pddlstream'
LAYERS = ['core', 'fastdownward', 'language', 'algorithms']


def layer(module):
    """The layer of a pddlstream module (pddlstream.<layer>...), or None for top-level modules."""
    parts = module.split('.')
    return parts[1] if (len(parts) > 2) and (parts[1] in LAYERS) else None


def imports(path):
    """The pddlstream modules that a file imports, anywhere in the file."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('pddlstream'):
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names if alias.name.startswith('pddlstream'))


MODULES = {
    str(path.relative_to(PACKAGE.parent).with_suffix('')).replace('/', '.').removesuffix('.__init__'): path
    for path in sorted(PACKAGE.rglob('*.py'))
}


@pytest.mark.parametrize('module', [module for module in MODULES if layer(module) is not None])
def test_layers(module):
    allowed = LAYERS[: LAYERS.index(layer(module)) + 1]
    violations = [target for target in imports(MODULES[module]) if layer(target) not in allowed]
    assert violations == [], f'{module} may only import from {allowed}'
