"""Run on the GPU machine with RoboLab installed: dumps every RoboLab task's instruction, success conditionals and
objects as JSON fixtures, and reports which tasks the planner's goals support.

Usage: python -m benchmarks.robolab.probe --output fixtures/robolab_tasks.json
The fixtures make the goal mapping testable without RoboLab (tests/test_goals.py reads them when present).
"""

import argparse
import importlib
import inspect
import json
import pkgutil

from benchmarks.common.goals import coverage
from benchmarks.common.protocol import to_json
from benchmarks.robolab.client import task_spec_from_robolab


def find_tasks(package_name='robolab.tasks'):
    """Every RoboLab Task subclass defined in the package (its modules import Isaac Lab)."""
    from robolab.core.task.task import Task

    package = importlib.import_module(package_name)
    tasks = {}
    for module_info in pkgutil.walk_packages(package.__path__, package_name + '.'):
        try:
            module = importlib.import_module(module_info.name)
        except Exception as error:  # Some task modules need assets that are not installed
            print(f'Skipping {module_info.name}: {error!r}')
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Task) and (cls is not Task) and (cls.__module__ == module.__name__):
                tasks[cls.__name__] = cls()
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', default='robolab.tasks', help='The package that defines the tasks')
    parser.add_argument('--output', default='robolab_tasks.json')
    args = parser.parse_args()
    specs = [task_spec_from_robolab(task) for task in find_tasks(args.package).values()]
    with open(args.output, 'w') as file:
        json.dump([to_json(spec) for spec in specs], file, indent=1)
    supported, skipped = coverage(specs)
    print(f'{len(specs)} tasks; supported: {len(supported)}')
    for _, reason in sorted(skipped.items()):
        print(f'  skipped {reason}')


if __name__ == '__main__':
    main()
