"""Times the benchmark cases in tests/cases.py and compares them with a saved baseline.

Usage: pixi run bench [--save] [--repeat N] [--baseline PATH]

Each case runs `repeat` times through tests/harness.py, and its fastest solve time is compared with the baseline:
more than 10% slower warns and more than 20% fails (exit code 1), ignoring slowdowns under 50 ms. Search counters
(iterations, evaluations, cost) are shown next to the baseline's but do not fail the run, because sampling still
varies between runs.
Timings depend on the machine, so the baseline lives outside git (default: .benchmarks/baseline.json).
"""
import argparse
import json
import pathlib
import platform
import sys
import tempfile

from cases import CASES
from helpers import ROOT, run_case

WARN_RATIO, FAIL_RATIO = 1.1, 1.2
MIN_SLOWDOWN = 0.05  # Seconds; timer noise for the fastest cases
COUNTERS = ['iterations', 'evaluations', 'cost']


def measure(case, repeat):
    times, counters = [], None
    for _ in range(repeat):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(case.example, cwd=tmp, timeout=case.timeout, display='gui' in case.marks)
        if result['error'] or not result['calls'] or not result['calls'][-1]['solved']:
            raise RuntimeError('{} did not solve: {}'.format(case.example, result['error']))
        times.append(sum(call['elapsed'] for call in result['calls']))
        summary = result['calls'][-1]['summary'] or {}
        counters = {name: summary.get(name) for name in COUNTERS}
    return {'time': min(times), 'times': times, 'counters': counters}


def compare(current, baseline):
    if baseline is None:
        return 'new', '{:8.2f}s'.format(current['time'])
    ratio = current['time'] / baseline['time']
    status = 'FAIL' if ratio > FAIL_RATIO else 'warn' if ratio > WARN_RATIO else 'ok'
    if current['time'] - baseline['time'] < MIN_SLOWDOWN:
        status = 'ok'
    changed = ['{}: {} -> {}'.format(key, baseline['counters'].get(key), value)
               for key, value in current['counters'].items() if baseline['counters'].get(key) != value]
    return status, '{:8.2f}s vs {:8.2f}s ({:+.0%}) {}'.format(
        current['time'], baseline['time'], ratio - 1, '; '.join(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--save', action='store_true', help='Overwrites the baseline with this run')
    parser.add_argument('--repeat', type=int, default=3, help='Runs per case (the fastest counts)')
    parser.add_argument('--baseline', type=pathlib.Path, default=ROOT / '.benchmarks' / 'baseline.json')
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())['cases'] if args.baseline.exists() else {}
    results, failed = {}, False
    for case in (case for case in CASES if case.bench):
        results[case.example] = measure(case, args.repeat)
        status, message = compare(results[case.example], baseline.get(case.example))
        failed |= (status == 'FAIL')
        print('{:5} {:45} {}'.format(status, case.example, message), flush=True)

    if args.save:
        args.baseline.parent.mkdir(exist_ok=True)
        args.baseline.write_text(json.dumps({'machine': platform.node(), 'python': platform.python_version(),
                                             'cases': results}, indent=2))
        print('Saved', args.baseline)
    sys.exit(int(failed and not args.save))


if __name__ == '__main__':
    main()
