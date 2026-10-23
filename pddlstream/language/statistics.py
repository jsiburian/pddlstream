import os
from collections import namedtuple

from pddlstream.core.context import get_config
from pddlstream.language.constants import is_plan
from pddlstream.utils import INF, CreationHash, ensure_dir, read_pickle, write_pickle

# Deprecated: pass load_statistics=False or save_statistics=False to solve() instead of assigning these
LOAD_STATISTICS = True
SAVE_STATISTICS = True

CACHE_DIR_VARIABLE = 'PDDLSTREAM_CACHE_DIR'
LEGACY_DATA_DIR = 'statistics/py3/'  # Where earlier versions saved statistics, relative to the working directory
DEFAULT_SEARCH_OVERHEAD = 1e2  # TODO: update this over time
EPSILON = 1e-6
# Can also include the overhead to process skeletons

Stats = namedtuple('Stats', ['p_success', 'overhead'])

# TODO: ability to "burn in" streams by sampling artificially to get better estimates


def safe_ratio(numerator, denominator, undefined=None):
    if denominator == 0:
        return undefined
    return float(numerator) / denominator


def geometric_cost(cost, p):
    return safe_ratio(cost, p, undefined=INF)


def check_effort(effort, max_effort):
    if max_effort is None:
        return True
    return effort < max_effort  # Exclusive


def compute_plan_effort(stream_plan, **kwargs):
    # TODO: compute effort in the delete relaxation way
    if not is_plan(stream_plan):
        return INF
    if not stream_plan:
        return 0
    return sum(result.get_effort(**kwargs) for result in stream_plan)


##################################################


def get_cache_dir():
    """$PDDLSTREAM_CACHE_DIR, or pddlstream/ in the user's cache directory ($XDG_CACHE_HOME, by default ~/.cache)."""
    if os.environ.get(CACHE_DIR_VARIABLE):
        return os.environ[CACHE_DIR_VARIABLE]
    user_cache_dir = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(user_cache_dir, 'pddlstream')


def get_data_path(stream_name):
    """Statistics are saved per stream file, named by its (define (stream <name>) ...)."""
    return os.path.join(get_cache_dir(), 'statistics', f'{stream_name}.pkl')


def load_data(pddl_name):
    if not get_config().load_statistics:
        return {}
    for filename in [get_data_path(pddl_name), os.path.join(LEGACY_DATA_DIR, f'{pddl_name}.pkl')]:
        if os.path.exists(filename):
            return read_pickle(filename)  # TODO: try/except
    return {}


def load_stream_statistics(externals):
    if not externals:
        return
    pddl_name = externals[0].pddl_name  # TODO: ensure the same
    # TODO: fresh restart flag
    data = load_data(pddl_name)
    for external in externals:
        if external.name in data:
            external.load_statistics(data[external.name])


##################################################


def dump_total_statistics(externals):
    print('\nTotal External Statistics')
    for external in externals:
        external.dump_total()


##################################################


def merge_data(external, previous_data):
    # TODO: compute distribution of successes given feasible
    # TODO: can estimate probability of success given feasible
    # TODO: single tail hypothesis testing (probability that came from this distribution)
    distribution = []
    for instance in external.instances.values():
        if instance.results_history:
            # TODO: also first attempt, first success
            last_success = -1
            for i, results in enumerate(instance.results_history):
                if results:
                    distribution.append(i - last_success)
                    last_success = i
    combined_distribution = previous_data.get('distribution', []) + distribution
    # TODO: count num failures as well
    # Alternatively, keep metrics on the lower bound and use somehow
    # Could assume that it is some other distribution beyond that point
    return {
        'calls': external.total_calls,
        'overhead': external.total_overhead,
        'successes': external.total_successes,
        'distribution': combined_distribution,
    }
    # TODO: make an instance method


def write_stream_statistics(externals, verbose):
    # TODO: estimate conditional to affecting history on skeleton
    # TODO: estimate conditional to first & attempt and success
    # TODO: relate to success for the full future plan
    # TODO: Maximum Likelihood Exponential - average (biased in general)
    if not externals:
        return
    if verbose:
        dump_total_statistics(externals)
    pddl_name = externals[0].pddl_name  # TODO: ensure the same
    previous_data = load_data(pddl_name)
    data = {}
    for external in externals:
        if not hasattr(external, 'instances'):
            continue  # TODO: SynthesizerStreams
        previous_statistics = previous_data.get(external.name, {})
        data[external.name] = merge_data(external, previous_statistics)

    if not get_config().save_statistics:
        return
    filename = get_data_path(pddl_name)
    ensure_dir(filename)
    write_pickle(filename, data)
    if verbose:
        print('Wrote:', filename)


##################################################


class PerformanceInfo:
    def __init__(self, p_success=1 - EPSILON, overhead=EPSILON, effort=None, estimate=False):
        # TODO: make info just a dict
        self.estimate = estimate
        if self.estimate:
            p_success = overhead = effort = None
        if p_success is not None:
            assert 0.0 <= p_success <= 1.0
        if overhead is not None:
            assert 0.0 <= overhead
        self.p_success = p_success
        self.overhead = overhead
        self.effort = effort

    def __repr__(self):
        return f'{self.__class__.__name__}{repr(self.__dict__)}'


class Performance(CreationHash):
    def __init__(self, name, info):
        self.name = name.lower()
        self.info = info
        self.initial_calls = 0
        self.initial_overhead = 0.0
        self.initial_successes = 0
        # TODO: online learning vs offline learning
        self.online_calls = 0
        self.online_overhead = 0.0
        self.online_successes = 0

    @property
    def total_calls(self):
        return self.initial_calls + self.online_calls

    @property
    def total_overhead(self):
        return self.initial_overhead + self.online_overhead

    @property
    def total_successes(self):
        return self.initial_successes + self.online_successes

    def load_statistics(self, statistics):
        self.initial_calls = statistics['calls']
        self.initial_overhead = statistics['overhead']
        self.initial_successes = statistics['successes']

    def update_statistics(self, overhead, success):
        self.online_calls += 1
        self.online_overhead += overhead
        self.online_successes += success

    def _estimate_p_success(self, reg_p_success=1.0, reg_calls=1):
        # TODO: use prior from info instead?
        return safe_ratio(
            self.total_successes + reg_p_success * reg_calls, self.total_calls + reg_calls, undefined=reg_p_success
        )

    def _estimate_overhead(self, reg_overhead=1e-6, reg_calls=1):
        # TODO: use prior from info instead?
        return safe_ratio(
            self.total_overhead + reg_overhead * reg_calls, self.total_calls + reg_calls, undefined=reg_overhead
        )

    def get_p_success(self):
        # TODO: could precompute and store
        if self.info.p_success is None:
            return self._estimate_p_success()
        return self.info.p_success

    def get_overhead(self):
        if self.info.overhead is None:
            return self._estimate_overhead()
        return self.info.overhead

    def could_succeed(self):
        return self.get_p_success() > 0

    def _estimate_effort(self, search_overhead=DEFAULT_SEARCH_OVERHEAD):
        p_success = self.get_p_success()
        return geometric_cost(self.get_overhead(), p_success) + (1 - p_success) * geometric_cost(
            search_overhead, p_success
        )

    def get_effort(self, **kwargs):
        if self.info.effort is None:
            return self._estimate_effort(**kwargs)
        elif callable(self.info.effort):
            return 0  # This really is a bound on the effort
        return self.info.effort

    def get_statistics(self, negate=False):  # negate=True is for the "worst-case" ordering
        sign = -1 if negate else +1
        return Stats(p_success=self.get_p_success(), overhead=sign * self.get_overhead())

    def dump_total(self):
        print(
            f'External: {self.name} | n: {self.total_calls:d} | p_success: {self._estimate_p_success():.3f} | '
            f'overhead: {self._estimate_overhead():.3f}'
        )

    def dump_online(self):
        if not self.online_calls:
            return
        print(
            f'External: {self.name} | n: {self.online_calls:d} | '
            f'p_success: {safe_ratio(self.online_successes, self.online_calls):.3f} | '
            f'mean overhead: {safe_ratio(self.online_overhead, self.online_calls):.3f} | '
            f'overhead: {self.online_overhead:.3f}'
        )


##################################################

# TODO: cannot easily do Bayesian hypothesis testing because might never receive ground truth when empty
# In some cases, the stream does finish though

# Estimate probability that will generate result
# Need to produce belief that has additional samples
# P(Success | Samples) = estimated parameter
# P(Success | ~Samples) = 0
# T(Samples | ~Samples) = 0
# T(~Samples | Samples) = 1-p

# TODO: estimate a parameter conditioned on successful streams?
# Need a transition fn as well because generating a sample might change state
# Problem with estimating prior. Don't always have data on failed streams

# Goal: estimate P(Success | History)
# P(Success | History) = P(Success | Samples) * P(Samples | History)
