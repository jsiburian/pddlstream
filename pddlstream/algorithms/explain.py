"""Explanations of solve results: why a solve found no plan, or what its plan relies on."""

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pddlstream.algorithms.certificate import PlanCertificate
from pddlstream.core.problem import Status

GAPS = {  # What each gap (stats['gaps']) means
    'planner_timeout': 'a FastDownward search ran out of time (max_planner_time)',
    'planner_memory': 'a FastDownward search ran out of memory',
    'planner_error': 'a FastDownward search failed',
    'max_skeletons': 'the skeleton limit (max_skeletons) was reached',
    'max_effort': 'streams were pruned by max_effort',
    'unreached_facts': 'facts above the complexity limit were never planned with',
    'max_time': 'time ran out (max_time)',
    'max_memory': 'memory ran out (max_memory)',
    'stream_errors': "streams raised errors (stream_errors='fail')",
    'incomplete_streams': 'streams ran out of outputs without being declared complete (strict=True)',
}


@dataclass
class Explanation:
    """Why a solve ended as it did. The stream and skeleton details need a solve with trace=True."""

    status: str
    termination: str | None
    gaps: list
    outcomes: dict  # Stream calls by outcome
    streams: dict = field(default_factory=dict)  # Stream calls by stream and outcome
    skeletons: list = field(default_factory=list)  # How far each skeleton was bound (SkeletonProgress)
    supports: dict = field(default_factory=dict)  # The plan's supporting stream calls by stream
    assumptions: list = field(default_factory=list)  # Preimage facts that no stream certified

    def to_dict(self):
        return dataclasses.asdict(self)

    def __str__(self):
        lines = [f'Status: {self.status} (termination: {self.termination})']
        if self.gaps and (self.status != Status.SUCCEEDED.value):
            lines.append('The search was cut short, so this does not prove that no plan exists:')
            lines.extend(f'  - {GAPS.get(gap, gap)}' for gap in self.gaps)
        if self.outcomes:
            lines.append(f'Stream calls: {format_counts(self.outcomes)}')
        failing = [name for name, outcomes in self.streams.items() if not outcomes.get('yielded')]
        if failing and (self.status != Status.SUCCEEDED.value):
            lines.append('Streams that never produced outputs:')
            lines.extend(f'  - {name}: {format_counts(self.streams[name])}' for name in failing)
        for skeleton in self.skeletons:
            stuck = f', next: {skeleton["next_stream"]}' if skeleton['next_stream'] else ''
            lines.append(f'Skeleton {skeleton["skeleton"]}: bound {skeleton["bound"]}/{skeleton["length"]}{stuck}')
        if self.supports:
            lines.append(f'Plan supports: {format_counts(self.supports)}')
        if self.assumptions:
            lines.append(f'Assumed facts: {self.assumptions}')
        if not self.streams and (self.status != Status.SUCCEEDED.value):
            lines.append('Solve with trace=True for details by stream and skeleton.')
        return '\n'.join(lines)


def format_counts(counts):
    return ', '.join(f'{name}: {num}' for name, num in counts.items())


def explain(result):
    """An Explanation of a SolveResult."""
    stats = result.stats or {}
    streams = defaultdict(Counter)
    skeletons = []
    for event in result.trace or []:
        if event['event'] == 'Call':
            streams[event['stream']][event['outcome']] += 1
        elif event['event'] == 'SkeletonProgress':
            skeletons.append({key: value for key, value in event.items() if key != 'event'})
    supports, assumptions = {}, []
    certificate = result.certificate
    if isinstance(certificate, PlanCertificate):
        supports = dict(Counter(support.stream for support in certificate.supports))
        assumptions = certificate.internal + certificate.unsupported
    return Explanation(
        status=result.status.value if result.status else None,
        termination=stats.get('termination'),
        gaps=list(stats.get('gaps', [])),
        outcomes=dict(stats.get('outcomes', {})),
        streams={name: dict(outcomes) for name, outcomes in streams.items()},
        skeletons=skeletons,
        supports=supports,
        assumptions=assumptions,
    )
