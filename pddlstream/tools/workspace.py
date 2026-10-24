"""What the agent tools act on: problems that the host program registers (with their Python procedures), a planner
session for each, and the results of searches."""

import itertools

from pddlstream.algorithms.session import PlannerSession


class Workspace:
    """Problems by id, each with a PlannerSession, and the results of runs by id. The host program adds problems and
    procedures; tools only refer to them by id, so an agent can never inject facts or code through a tool call."""

    def __init__(self):
        self.problems = {}  # problem_id: (PDDLProblem, stream_info, config)
        self.sessions = {}
        self.results = {}  # run_id: (problem_id, SolveResult)
        self.hints = {}  # problem_id: plan skeleton to try first
        self.goals = {}  # problem_id: an accepted proposed goal
        self._run_ids = itertools.count()

    def add_problem(self, problem_id, problem, stream_info=None, **config):
        """Registers a problem (replacing any problem with this id and its session)."""
        if problem_id in self.sessions:
            self.sessions.pop(problem_id).close()
        self.goals.pop(problem_id, None)
        self.problems[problem_id] = (problem, stream_info or {}, config)

    def get_problem(self, problem_id):
        if problem_id not in self.problems:
            raise KeyError(f'Unknown problem {problem_id!r}; known: {sorted(self.problems)}')
        return self.problems[problem_id]

    def session(self, problem_id):
        if problem_id not in self.sessions:
            problem, stream_info, config = self.get_problem(problem_id)
            self.sessions[problem_id] = PlannerSession(problem, stream_info=stream_info, verbose=False, **config)
        return self.sessions[problem_id]

    def add_result(self, problem_id, result):
        run_id = f'run{next(self._run_ids)}'
        self.results[run_id] = (problem_id, result)
        return run_id

    def get_result(self, run_id):
        if run_id not in self.results:
            raise KeyError(f'Unknown run {run_id!r}')
        return self.results[run_id]

    def last_result(self, problem_id):
        runs = [result for pid, result in self.results.values() if pid == problem_id]
        return runs[-1] if runs else None

    def close(self):
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()
