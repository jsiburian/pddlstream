"""Where stream procedures run. A stream call has two parts: Instance.run_procedure runs the user's procedure (and
touches only that instance's generator), and Instance.finish_call commits its outcome to the solver (objects,
results, statistics). Commits always happen on the solver's thread, in the order the algorithm makes its calls.

An executor runs procedures (SolverConfig.executor; by default SerialExecutor). A parallel executor also starts the
procedures of calls that the algorithm is likely to make next (prefetching), so they run while the solver works.
Because commits keep their order, a solve with pure procedures (whose outputs depend only on their inputs and earlier
calls of the same instance) makes the same calls and finds the same plans with any executor.
"""

import itertools
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import NamedTuple

from pddlstream.language.generator import get_next


class PendingCall(NamedTuple):
    """The outcome of a procedure run that finish_call has not committed yet."""

    output: object = None  # What the procedure returned or yielded
    enumerated: bool = False  # Whether the procedure has no further outputs
    error: Exception | None = None  # What the procedure raised
    seconds: float = 0.0  # How long the procedure ran
    ran: bool = True  # False when the call replays the instance's history instead


REPLAY = PendingCall(ran=False)


class Done:
    """A handle on a procedure run that already finished."""

    def __init__(self, pending):
        self.pending = pending

    def result(self):
        return self.pending


def timed(procedure, *args):
    """Runs procedure, returning a PendingCall with its output (or error) and running time."""
    start = time.perf_counter()
    try:
        output, enumerated = procedure(*args)
    except Exception as error:
        return PendingCall(error=error, seconds=time.perf_counter() - start)
    return PendingCall(output, enumerated, seconds=time.perf_counter() - start)


##################################################


class SerialExecutor:
    """Runs each procedure on the solver's thread when its call is made (the default)."""

    parallel = False
    lookahead = 0  # How many upcoming calls to start ahead

    def submit(self, instance):
        return Done(instance.run_procedure())

    def prefetch(self, instances):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BatchHandle:
    """A handle on one call of a batch procedure run."""

    def __init__(self, future, index):
        self.future, self.index = future, index

    def result(self):
        return self.future.result()[self.index]


def run_batch(procedure, input_tuples):
    """Runs a BatchProcedure on several calls' inputs, returning a PendingCall for each."""
    start = time.perf_counter()
    try:
        outputs = list(procedure.batch_fn(input_tuples))
        if len(outputs) != len(input_tuples):
            raise ValueError(f'A batch procedure returned {len(outputs)} output lists for {len(input_tuples)} inputs')
    except Exception as error:
        return len(input_tuples) * [PendingCall(error=error, seconds=time.perf_counter() - start)]
    seconds = (time.perf_counter() - start) / len(input_tuples)
    return [PendingCall(output, True, seconds=seconds) for output in outputs]


class ParallelExecutor(SerialExecutor):
    """Prefetching for executors that run procedures elsewhere: up to lookahead upcoming calls are started ahead, at
    most one per instance (an instance's generator is never entered twice at once). The calls of a batch procedure
    (from_batch_fn) among them are started as one batch."""

    parallel = True

    def __init__(self, lookahead=4):
        self.lookahead = lookahead

    def offloads(self, instance):
        """Whether the executor runs instance's procedure off the solver's thread."""
        return not instance.runs_inline()

    def submit(self, instance):
        if not self.offloads(instance):
            return super().submit(instance)
        return self.submit_elsewhere(instance)

    def submit_elsewhere(self, instance):
        raise NotImplementedError()

    def prefetch(self, instances):
        batches = {}
        for instance in itertools.islice(instances, self.lookahead):
            if not (self.offloads(instance) and instance.can_prefetch()):
                continue
            if any(instance in group for group in batches.values()):
                continue
            procedure = instance.batch_procedure()
            if procedure is None:
                instance.set_prefetched(self.submit(instance))
            else:
                batches.setdefault(procedure, []).append(instance)
        for procedure, group in batches.items():
            for instance, handle in zip(group, self.submit_batch(procedure, group)):
                instance.set_prefetched(handle)

    def submit_batch(self, procedure, instances):
        return [self.submit(instance) for instance in instances]


class ThreadExecutor(ParallelExecutor):
    """Runs procedures on threads, for procedures that release the GIL (numpy, simulators, network or GPU calls).

    :param workers: threads in the default pool
    :param resources: pool sizes by resource name; streams declare theirs with StreamInfo(resource=...). Procedures of
        resource 'main' run on the solver's thread, for procedures that are not thread-safe.
    :param lookahead: how many upcoming calls to start ahead (by default, workers)
    """

    def __init__(self, workers=4, resources=None, lookahead=None):
        super().__init__(workers if lookahead is None else lookahead)
        self.sizes = {None: workers, **(resources or {})}
        self.pools = {}

    def pool(self, resource):
        if resource not in self.pools:
            self.pools[resource] = ThreadPoolExecutor(self.sizes.get(resource, 1), thread_name_prefix='pddlstream')
        return self.pools[resource]

    def submit_elsewhere(self, instance):
        return self.pool(instance.info.resource).submit(copy_context().run, instance.run_procedure)

    def submit_batch(self, procedure, instances):
        resource = instances[0].info.resource
        input_tuples = [instance.get_input_values() for instance in instances]
        future = self.pool(resource).submit(copy_context().run, run_batch, procedure, input_tuples)
        return [BatchHandle(future, index) for index in range(len(instances))]

    def close(self):
        for pool in self.pools.values():
            pool.shutdown()
        self.pools.clear()


##################################################


def seed_worker(seed, index):
    random.seed(seed + index)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return
    np.random.seed(seed + index)


def run_request(procedures, generators, request):
    """Runs a worker's procedure for one call: (output, enumerated)."""
    key, name, input_values, fluents, kind = request
    procedure = procedures[name]
    if kind == 'function':
        return procedure(*input_values), True
    if key not in generators:
        generators[key] = procedure(*input_values, **({} if fluents is None else {'fluents': fluents}))
    return get_next(generators[key], default=[])


def serve_worker(connection, make_procedures, seed, index):
    """A worker process's loop: runs the procedures that make_procedures() returns, keeping each instance's
    generator."""
    seed_worker(seed, index)
    procedures = make_procedures()
    generators = {}
    while True:
        request = connection.recv()
        if request is None:
            return
        pending = timed(run_request, procedures, generators, request)
        try:
            connection.send(pending)
        except Exception as error:  # The output or error could not be pickled
            connection.send(PendingCall(error=RuntimeError(f'{request[1]} returned an unpicklable value: {error!r}')))


class ProcessExecutor(ParallelExecutor):
    """Runs the procedures of one resource in worker processes, e.g. simulators that must not share a process. Each
    worker calls make_procedures() once to build its own procedures (by stream or function name) and seeds random and
    numpy with seed plus its index. Each instance always runs on the same worker, which keeps its generator. Other
    procedures run on the solver's thread. Inputs and outputs must be picklable.

    :param make_procedures: a picklable function (e.g. defined at module level) returning {name: procedure}
    :param resource: the StreamInfo resource whose procedures run in the workers
    """

    def __init__(self, make_procedures, resource='process', workers=2, seed=0, lookahead=None):
        super().__init__(workers if lookahead is None else lookahead)
        self.make_procedures, self.resource, self.seed = make_procedures, resource, seed
        self.num_workers = workers
        self.workers = None
        self.worker_from_instance = {}
        self.key_from_instance = {}

    def start(self):
        context = multiprocessing.get_context('spawn')
        self.workers = []
        for index in range(self.num_workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=serve_worker, args=(child, self.make_procedures, self.seed, index), daemon=True
            )
            process.start()
            child.close()
            # One thread per worker sends its requests in order and waits for each reply
            self.workers.append((process, parent, ThreadPoolExecutor(1, thread_name_prefix='pddlstream-process')))

    def offloads(self, instance):
        return (instance.info.resource == self.resource) and not instance.runs_inline()

    def submit_elsewhere(self, instance):
        if self.workers is None:
            self.start()
        if instance not in self.key_from_instance:
            self.key_from_instance[instance] = len(self.key_from_instance)
            self.worker_from_instance[instance] = self.key_from_instance[instance] % self.num_workers
        _, connection, thread = self.workers[self.worker_from_instance[instance]]
        request = instance.procedure_request(self.key_from_instance[instance])
        return thread.submit(round_trip, connection, request)

    def close(self):
        for process, connection, thread in self.workers or []:
            thread.shutdown()
            try:
                connection.send(None)
            except OSError:  # pragma: no cover
                pass
            process.join(timeout=5)
            if process.is_alive():  # pragma: no cover
                process.terminate()
        self.workers = None


def round_trip(connection, request):
    connection.send(request)
    return connection.recv()
