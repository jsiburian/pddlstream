"""Example runs checked by tests/test_properties.py and timed by tests/benchmark.py.

Each case is an example command (a module under examples/ and its arguments, as for `python -m examples.<module>`)
plus what its solution must satisfy. Costs are exact for examples whose solutions are reproducible and upper
bounds for those that sample differently between runs.
"""

from typing import NamedTuple

from pddlstream.core.utils import INF

FOCUSED_TAMP_BUG = 'Pre-existing (also on the original code): reports infeasible / never binds on continuous_tamp'
TURTLEBOT_REPLAY_BUG = 'Pre-existing: replay crashes in pybullet-planning visual_shape_from_data (bytes/str)'


FEATURES = {
    'pddl': 'Plain PDDL, no streams',
    'adl': 'Disjunctive goals, quantifiers or conditional effects',
    'derived': 'Derived predicates (axioms)',
    'costs': 'Action costs',
    'streams': 'Streams (samplers and tests)',
    'functions': 'External functions (:function), e.g. costs computed by procedures',
    'predicates': 'External predicates (:predicate)',
    'negated': 'Test streams used negated',
    'fluent': 'Fluent streams (:fluents)',
    'wild': 'Wild streams (WildOutput facts)',
    'exogenous': 'Exogenous streams',
    'deferred': 'Deferred streams (StreamAction steps in plans)',
    'rules': 'Rules (:rule)',
    'optimizer': 'Optimizers (:optimizer)',
    'constraints': 'Plan constraints (skeletons or a cost bound)',
    'satisfaction': 'Constraint satisfaction instead of planning',
    'belief': 'Belief-space planning',
    'debug': 'Placeholder procedures (stream_map=DEBUG)',
}


class Case(NamedTuple):
    example: str
    cost: float = None  # Exact cost of the last solve
    max_cost: float = INF  # Upper bound on the cost of the last solve
    length: int = None  # Exact plan length of the last solve
    valid: bool = True  # Validator verdict for PDDLStream plans (None: plans with deferred streams)
    timeout: float = 120  # Seconds for the whole run, including start-up and display
    marks: tuple = ()  # gui (needs a display), slow, gurobi (needs the `full` environment)
    xfail: str = None  # Reason, for known failures
    deterministic: bool = True  # Solves with deterministic=True and must reproduce its recorded output
    bench: bool = False  # Timed by tests/benchmark.py
    features: tuple = ()  # Language features the example exercises (see FEATURES and docs/v2/features.md)


CASES = [
    # Discrete
    Case('blocksworld.run -a incremental', length=4, features=('pddl',)),
    Case('blocksworld.run -a adaptive', length=4, features=('pddl',)),
    Case('blocksworld.run -a focused', length=4, features=('pddl',)),
    Case('blocksworld.run -a binding', length=4, features=('pddl',)),
    Case('blocksworld.run_derived', cost=4, features=('derived',)),
    Case('kitchen.run', cost=32, bench=True, features=('streams', 'fluent', 'negated', 'derived', 'costs', 'debug')),
    Case('temporal.run', cost=4, length=8, bench=True, features=('streams', 'derived', 'costs')),
    Case('ipc.rovers.run', length=10, features=('pddl',)),
    Case('ipc.satellites.run', length=9, features=('pddl',)),
    # Language features
    Case('advanced.adl.run', length=1, features=('adl',)),
    Case('advanced.counting.run', length=1, bench=True, features=('streams',)),
    Case('advanced.defer.run', length=5, valid=None, features=('streams', 'deferred', 'derived')),
    Case('advanced.exogenous.run', length=5, valid=None, features=('streams', 'exogenous', 'deferred')),
    Case('advanced.fluent.run', length=5, features=('streams', 'fluent')),
    Case('advanced.multi_cost', cost=4, features=('streams', 'functions', 'costs')),
    Case('advanced.satisfy.run', length=2, features=('streams', 'satisfaction')),
    Case('advanced.satisfy.run -a adaptive', length=2, features=('streams', 'satisfaction')),
    Case('advanced.storage', length=5, bench=True, features=('streams', 'debug')),
    Case('advanced.wild.run', cost=47.312, length=4, bench=True, features=('streams', 'functions', 'costs')),
    # Belief space
    Case('table_belief.run', cost=10.532, length=10, features=('belief', 'derived', 'costs')),
    Case(
        'discrete_belief.run -t 300',
        timeout=400,
        marks=('slow',),
        deterministic=False,
        features=('streams', 'functions', 'predicates', 'belief', 'derived', 'costs'),
    ),  # Anytime until max_time
    # Optimizers
    Case(
        'cashpoint.run', cost=48, length=3, marks=('gurobi',), features=('streams', 'functions', 'optimizer', 'costs')
    ),
    Case(
        'continuous_tamp.run -g',
        max_cost=150,
        length=9,
        marks=('gui', 'gurobi'),
        features=('streams', 'functions', 'optimizer', 'negated', 'deferred', 'derived', 'costs'),
    ),
    # 2D task and motion planning (Tkinter viewers)
    Case(
        'continuous_tamp.run',
        max_cost=150,
        length=9,
        marks=('gui',),
        features=('streams', 'functions', 'deferred', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.run -s',
        max_cost=150,
        length=9,
        marks=('gui',),
        features=('streams', 'functions', 'deferred', 'derived', 'costs', 'constraints'),
    ),
    Case(
        'continuous_tamp.run -p blocked -n 3',
        max_cost=180,
        timeout=300,
        marks=('gui',),
        bench=True,
        features=('streams', 'functions', 'deferred', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.run -a incremental -c',
        max_cost=150,
        length=9,
        marks=('gui',),
        bench=True,
        features=('streams', 'functions', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.run -a focused',
        marks=('gui',),
        xfail=FOCUSED_TAMP_BUG,
        features=('streams', 'functions', 'deferred', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.run -a binding',
        marks=('gui',),
        xfail=FOCUSED_TAMP_BUG,
        features=('streams', 'functions', 'deferred', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.ffrob.run',
        cost=9,
        length=9,
        marks=('gui',),
        features=('streams', 'wild', 'negated', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.idtmp.run',
        cost=9,
        length=9,
        marks=('gui',),
        features=('streams', 'fluent', 'derived', 'costs'),
    ),
    Case(
        'continuous_tamp.diverse',
        cost=9,
        length=9,
        marks=('gui',),
        features=('streams', 'functions', 'deferred', 'derived', 'costs'),
    ),
    Case('continuous_tamp.satisfy', max_cost=150, length=12, marks=('gui',), features=('streams', 'satisfaction')),
    Case(
        'discrete_tamp.run',
        cost=10,
        length=8,
        marks=('gui',),
        features=('streams', 'functions', 'rules', 'negated', 'derived', 'costs'),
    ),
    Case(
        'discrete_tamp.push.run',
        cost=20,
        length=8,
        marks=('gui',),
        features=('streams', 'functions', 'rules', 'wild', 'negated', 'derived', 'costs'),
    ),
    Case(
        'motion.run',
        max_cost=2,
        marks=('gui',),
        bench=True,
        deterministic=False,
        features=('streams', 'functions', 'derived', 'costs', 'constraints'),
    ),  # Anytime until max_time
    # PyBullet
    Case(
        'pybullet.kuka.run',
        length=10,
        timeout=300,
        marks=('gui',),
        features=('streams', 'fluent', 'negated', 'derived'),
    ),
    Case(
        'pybullet.pr2.run -teleport',
        cost=7,
        length=9,
        timeout=300,
        marks=('gui',),
        bench=True,
        features=('streams', 'functions', 'negated', 'deferred', 'derived', 'costs'),
    ),
    Case(
        'pybullet.pr2.run',
        cost=7,
        length=9,
        timeout=600,
        marks=('gui', 'slow'),
        features=('streams', 'functions', 'negated', 'deferred', 'derived', 'costs'),
    ),
    Case('pybullet.pr2_rovers.run', length=7, timeout=300, marks=('gui',), features=('streams',)),
    Case(
        'pybullet.tamp.run -n 2',
        max_cost=50,
        length=8,
        timeout=300,
        marks=('gui',),
        features=('streams', 'functions', 'negated', 'derived', 'costs'),
    ),
    Case(
        'pybullet.tamp.run -n 3 -teleport',
        max_cost=80,
        length=12,
        timeout=300,
        marks=('gui',),
        features=('streams', 'functions', 'negated', 'derived', 'costs'),
    ),
    Case(
        'pybullet.pr2_belief.run',
        timeout=600,
        marks=('gui', 'slow'),
        features=('streams', 'belief', 'derived', 'costs'),
    ),
    Case(
        'pybullet.turtlebot_rovers.run',
        timeout=300,
        marks=('gui',),
        xfail=TURTLEBOT_REPLAY_BUG,
        deterministic=False,
        features=('streams', 'negated', 'derived'),
    ),  # Its deterministic sampling budget runs out of time
]
