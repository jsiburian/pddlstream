"""Datasets for learning from solves: rows made from value-level traces (SolveResult.trace, with trace=True), stored
as JSON lines.

A call row has the stream, its inputs (as strings), the call's index for its instance, the outcome, how many results
and facts it produced, and whether the procedure ran (fresh). A skeleton row has the skeleton's streams and actions,
its optimistic cost, and how far binding it got. Both carry the labels given, e.g. the problem's name.
"""

import json

CALL_FIELDS = ['stream', 'inputs', 'call_index', 'outcome', 'results', 'facts', 'fresh']


def rows_from_trace(trace, **labels):
    """The dataset rows of a trace."""
    progress = {event['skeleton']: event for event in trace if event['event'] == 'SkeletonProgress'}
    rows = []
    for event in trace:
        if event['event'] == 'Call':
            rows.append({'kind': 'call', **labels, **{field: event[field] for field in CALL_FIELDS}})
        elif event['event'] == 'SkeletonAdded':
            reached = progress.get(event['skeleton'], {})
            rows.append(
                {
                    'kind': 'skeleton',
                    **labels,
                    'skeleton': event['skeleton'],
                    'streams': event['stream_plan'],
                    'actions': event['action_plan'],
                    'cost': event['cost'],
                    'bound': reached.get('bound'),
                    'length': reached.get('length'),
                    'next_stream': reached.get('next_stream'),
                }
            )
    return rows


def write_jsonl(rows, path, append=True):
    with open(path, 'a' if append else 'w') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')


def read_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]
