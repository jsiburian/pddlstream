"""A PlannerSession reuses the stream results that a change to the world leaves valid."""

from pddlstream.algorithms.session import PlannerSession, WorldPatch
from pddlstream.core.problem import PDDLProblem, Status
from pddlstream.language.generator import from_fn
from pddlstream.language.stream import StreamInfo
from test_semantics import DOMAIN_PDDL

STREAMS = """(define (stream session)
  (:stream sample :inputs (?s) :domain (Seed ?s) :outputs (?y) :certified (Other ?y))
  (:stream extend :inputs (?y) :domain (Other ?y) :outputs (?x) :certified (and (Value ?x) (Good ?x))))"""
DOMAIN = DOMAIN_PDDL.replace('(Other ?y)', '(Other ?y) (Seed ?s)')
WORLD = {'walls': 0}  # Something extend reads besides its inputs


def make_session(**stream_info):
    stream_map = {'sample': from_fn(lambda s: (s,)), 'extend': from_fn(lambda y: (y + 10 + WORLD['walls'],))}
    problem = PDDLProblem(DOMAIN, {}, STREAMS, stream_map, [('Seed', 1)], ('Done',))
    info = {'sample': StreamInfo(reads=()), 'extend': StreamInfo(reads=('walls',)), **stream_info}
    return PlannerSession(problem, stream_info=info, algorithm='adaptive', deterministic=True, verbose=False)


def counts(session, field):
    return {name: getattr(report, field) for name, report in session.report.items()}


def test_unchanged_world_reuses_everything():
    session = make_session()
    first = session.solve()
    assert counts(session, 'fresh') == {'sample': 1, 'extend': 1}
    second = session.solve()
    assert (second.status is Status.SUCCEEDED) and (second.plan == first.plan)
    assert counts(session, 'reused') == {'sample': 1, 'extend': 1}
    assert counts(session, 'fresh') == {'sample': 0, 'extend': 0}


def test_fact_patch_drops_dependents():
    session = make_session()
    session.solve()
    assert session.apply(WorldPatch(add=[('Seed', 2)], remove=[('Seed', 1)])) == {}  # Both streams read only inputs
    result = session.solve()
    assert [action.args for action in result.plan] == [(12,)]
    assert counts(session, 'lost') == {'sample': 1, 'extend': 1}  # Seed 1 no longer holds, nor Other 1 with it
    assert counts(session, 'fresh') == {'sample': 1, 'extend': 1}


def test_changed_name_invalidates_readers():
    session = make_session()
    session.solve()
    WORLD['walls'] = 100
    try:
        assert session.apply(WorldPatch(changed=['walls'])) == {'extend': 1}
        result = session.solve()
    finally:
        WORLD['walls'] = 0
    assert [action.args for action in result.plan] == [(111,)]
    assert counts(session, 'invalidated') == {'sample': 0, 'extend': 1}
    assert counts(session, 'fresh') == {'sample': 0, 'extend': 1}


def test_unknown_reads_are_invalidated_by_any_change():
    session = make_session(sample=StreamInfo())  # reads=None
    session.solve()
    assert session.apply(WorldPatch(changed=['anything'])) == {'sample': 1}


def test_other_support_keeps_fact():
    session = make_session()
    session.solve()
    assert session.invalidate('extend') == 1
    session.solve()
    assert counts(session, 'reused') == {'sample': 1, 'extend': 0}
    assert counts(session, 'fresh') == {'sample': 0, 'extend': 1}


def test_sessions_are_isolated():
    first, second = make_session(), make_session()
    plans = [session.solve().plan for session in [first, second, first, second]]
    assert all(plan == plans[0] for plan in plans)
    assert first.registry is not second.registry
