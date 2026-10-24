# pddlstream

PDDLStream is a planning framework comprised of an action language and suite of algorithms for Artificial Intelligence (AI) planning in the presence of sampling procedures.
PDDLStream extends Planning Domain Definition Language (PDDL) by introducing streams, declarative specifications of sampling procedures.
PDDLStream algorithms are domain independent and solve PDDLStream problems with only a blackbox description of each sampler.
The motivating application of PDDLStream was for general-purpose robot Task and Motion Planning (TAMP). 

The [default](https://github.com/caelan/pddlstream) **pddlstream** branch ([main](https://github.com/caelan/pddlstream/tree/stable)) is the newest stable "release" of **pddlstream**.
The [downward](https://github.com/caelan/pddlstream/tree/master) **pddlstream** branch is the most recent and advanced version of **pddlstream** but also is somewhat experimental.

<!--https://www.markdownguide.org/basic-syntax/-->
<!--https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet-->

## Publications

* [PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning](https://arxiv.org/abs/1802.08705
)
<!--* [STRIPStream: Planning In Infinite Domains](https://arxiv.org/abs/1701.00287)-->

## Citation

Caelan R. Garrett, Tomás Lozano-Pérez, Leslie P. Kaelbling. PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning, International Conference on Automated Planning and Scheduling (ICAPS), 2020.

## Contact

Caelan Garrett: [username]@csail.mit.edu

## History

PDDLStream is the "third version" of the PDDLStream/STRIPStream planning framework, intended to supersede previous versions:

1) https://github.com/caelan/stripstream
2) https://github.com/caelan/ss

PDDLStream makes several representational and algorithmic improvements over these versions.
Most notably, it adheres to PDDL conventions and syntax whenever possible and contains several new algorithms. 
<!--An implementation of STRIPStream that uses PDDL for the specifiation of actions and streams.-->
<!--https://github.com/caelan/pddlstream/compare/master...stable-->

## Installation

Dependencies, including the compilers used to build FastDownward, are managed with [pixi](https://pixi.sh):
```
$ git clone --recursive https://github.com/caelan/pddlstream.git
$ cd pddlstream
pddlstream$ pixi run setup
```
`pixi run setup` fetches the FastDownward and pybullet-planning submodules and compiles FastDownward.
pixi installs **pddlstream** in editable mode, so `import pddlstream` works from any directory inside the environment (`pixi shell`, or `pixi run python ...`).

Environments:
* `default`: the library, the FastDownward toolchain, and the dependencies of the Tkinter and PyBullet examples (Python 3.14, NumPy 2).
* `full`: adds pygraphviz and matplotlib (`visualize=True`) and gurobipy for the optimizer examples. Use it with `pixi run -e full ...`; Gurobi requires a license for non-trivial models.
* `py311`: the `default` environment on Python 3.11 (the version Isaac Sim pins).

Other tasks:
* `pixi run build-fd`: rebuild FastDownward. It compiles on every core; add `-j4` to limit memory use.
* `pixi run clean-fd`: delete the FastDownward build (needed after switching compilers).
* `pixi run build-ikfast`: optionally compile the PR2 IKFast solver; PyBullet IK is used otherwise.
* `pixi run test`: run the fast test suite. Every headless example must solve, and the validator must accept its plan.
* `pixi run test-gui`: run the examples that open viewers (Tkinter and PyBullet) under `xvfb-run`.
* `pixi run -e full test-gurobi`: run the optimizer examples. Add `-m slow` to any pytest run for the multi-minute examples.
* `pixi run bench`: time the benchmark examples against a baseline saved with `pixi run bench --save`.
* `pixi run lint`: check the code with ruff; `pixi run format` fixes what it can and formats the code. To skip the formatting commit in `git blame`, run `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

PDDLStream requires Python 3.11 or newer.
pddlstream looks for FastDownward in the `downward` submodule of its source checkout, so install it in editable mode, or point `PDDLSTREAM_FD_PATH` to a compiled checkout of the fork (for example after a non-editable `pip install .`).
FastDownward is loaded the first time a solver needs it, so `import pddlstream` works without a build.
Without pixi, install CMake and a C++ compiler, then run `CMAKE_POLICY_VERSION_MINIMUM=3.5 CXXFLAGS=-Wno-template-body ./downward/build.py` and `pip install -e .` (the `CXXFLAGS` setting is only needed for GCC 15 and newer).
See FastDownward's [documentation](http://www.fast-downward.org/ObtainingAndRunningFastDownward) for more detailed build instructions.

Make sure to recursively update **pddlstream**'s submodules when pulling new commits, then rebuild FastDownward:
```
pddlstream$ ./update.sh
```

## Documentation

* [Writing streams](docs/writing-streams.md): problems, stream declarations and procedures, options and debugging.
* [Replanning with planner sessions](docs/sessions.md): reusing stream results as the world changes.
* [Agent tools](docs/agents.md): JSON tools for agents to inspect problems, test samplers, plan and validate.
* [Architecture](docs/architecture.md): how the package is organized and what happens during a solve.
* [Migrating from earlier versions](docs/migration.md): new options, results and import paths, and what was removed.

## Examples

This repository contains several robotic and non-robotic PDDLStream example domains.
The commands below assume an activated environment (`pixi shell`); otherwise, prefix them with `pixi run`.

### PyBullet

PyBullet and the [pybullet-planning](https://github.com/caelan/pybullet-planning) submodule are included in the default environment.

Examples:
* PR2 TAMP: `pddlstream$ python -m examples.pybullet.tamp.run`
* PR2 Cleaning and Cooking: `pddlstream$ python -m examples.pybullet.pr2.run`
* Turtlebot Rovers: `pddlstream$ python -m examples.pybullet.turtlebot_rovers.run`
* PR2 Rovers: `pddlstream$ python -m examples.pybullet.pr2_rovers.run`
* PR2 Planning and Execution: `pddlstream$ python -m examples.pybullet.pr2_belief.run`
* Kuka Cleaning and Cooking: `pddlstream$ python -m examples.pybullet.kuka.run`

<!--[![Kuka IIWA](https://img.youtube.com/vi/3HJrkgIGK7c/0.jpg)](https://www.youtube.com/watch?v=3HJrkgIGK7c)-->
[<img src="https://img.youtube.com/vi/Uc0fogLsPMI/0.jpg" height="200">](https://www.youtube.com/watch?v=Uc0fogLsPMI)
&emsp;[<img src="https://img.youtube.com/vi/HVD8SpmguYs/0.jpg" height="200">](https://www.youtube.com/watch?v=HVD8SpmguYs)
&emsp;[<img src="https://img.youtube.com/vi/oWr6m12nXcM/0.jpg" height="200">](https://www.youtube.com/watch?v=oWr6m12nXcM)
<img src="images/pybullet_belief.png" height="150">
&emsp;[<img src="https://img.youtube.com/vi/XcxsU0VuRUI/0.jpg" height="200">](https://www.youtube.com/watch?v=XcxsU0VuRUI)
&emsp;[<img src="https://img.youtube.com/vi/3HJrkgIGK7c/0.jpg" height="200">](https://www.youtube.com/watch?v=3HJrkgIGK7c)

See https://github.com/caelan/pybullet-planning for more information about my PyBullet planning primitives library.

### Python TKinter

Tkinter is included in the default environment.

Examples:
* 1D Continuous TAMP: `pddlstream$ python -m examples.continuous_tamp.run`
* 2D Motion Planning: `pddlstream$ python -m examples.motion.run`
* Discrete TAMP: `pddlstream$ python -m examples.discrete_tamp.run`
* Discrete TAMP with pushing: `pddlstream$ python -m examples.discrete_tamp.push.run`

<img src="images/discrete_tamp.png" height="100">&emsp;<img src="images/continuous_tamp.png" height="100">&emsp;<img src="images/motion.png" height="100">

### Pure Python

Simple examples that only need the core library:
* Blocksworld: `pddlstream$ python -m examples.blocksworld.run`
* Blocksworld with Derived Predicates: `pddlstream$ python -m examples.blocksworld.run_derived`
* Kitchen (debug streams): `pddlstream$ python -m examples.kitchen.run`

<!--* Discrete Belief: `pddlstream$ python -m examples.table_obs.run`
* Discrete Belief Space: `pddlstream$ python -m examples.discrete_belief.run`
`$ pip install numpy scipy`-->

### Advanced Functionality

Test cases or advanced (and undocumented) functionality:
* Action Description Language (ADL): `pddlstream$ python -m examples.advanced.adl.run`
* Deferred streams (postponed evaluation): `pddlstream$ python -m examples.advanced.defer.run`
* Exogenous streams (observations): `pddlstream$ python -m examples.advanced.exogenous.run`
* Fluent streams (state constraints): `pddlstream$ python -m examples.advanced.fluent.run`
* Constraint satisfaction: `pddlstream$ python -m examples.advanced.satisfy.run`
* Wild streams (ad hoc certification with `WildOutput`): `pddlstream$ python -m examples.discrete_tamp.push.run` and `pddlstream$ python -m examples.continuous_tamp.ffrob.run`
* Objects created by streams (cutting carrots): `pddlstream$ python -m examples.advanced.wild.run`

### International Planning Competition (IPC)

Unmodified PDDL IPC examples solved using PDDLStream's modified translator:
* Rovers: `pddlstream$ python -m examples.ipc.rovers.run`
* Satellites: `pddlstream$ python -m examples.ipc.satellites.run`

## Applications

External projects that make use of PDDLStream:
* Online TAMP - https://github.com/caelan/SS-Replan
* Automated Construction - https://github.com/caelan/pb-construction
* Learning + TAMP (LTAMP) - https://github.com/caelan/LTAMP
* pyrobosim - https://github.com/sea-bass/pyrobosim
* PDDLStream + Drake examples - https://github.com/RobotLocomotion/6-881-examples
<!--https://github.com/rachelholladay/ftamp-->

## Algorithms

PDDLStream is a planning framework comprised of a **single** planning language but **multiple** planning algorithms.
Some of the algorithms are radically different than others (e.g. Incremental vs Focused) and thus the planning time can also substantially vary.
The **Adaptive** algorithm typically performs best for domains with many possible sampling pathways, such as robot manipulation domains.
<!--For domains with few sampling pathways, the **Incremental** algorithm typically performs best (link to specific examples)-->

The meta procedure `solve(...)` (in `pddlstream/algorithms/meta.py`) allows the user to toggle between available algorithms using the keyword argument **algorithm={}**.
It and the other names that most programs need can be imported from the package itself, e.g. `from pddlstream import solve, PDDLProblem, And, StreamInfo, from_gen_fn`.
Its options are the fields of `SolverConfig` (`pddlstream/core/config.py`), which also documents them: pass them as keyword arguments, or build a config once and reuse it with `solve(problem, config=config)`, where keyword arguments override its fields.
Misspelled options raise a `TypeError` before solving starts.
The solvers return a `SolveResult`, which unpacks as `plan, cost, certificate` like before; `result.status` says whether a plan was found (`Status.SUCCEEDED`), a limit was reached first (`Status.FAILED`), or the solver concluded that no plan exists (`Status.INFEASIBLE`), and `result.stats` holds the printed summary (iterations, search, sampling and total time, ...).
The solvers report progress through Python's `logging`, with loggers named after their modules under `pddlstream`.
Without a logging configuration, they print it to stdout while solving, as before; `logging.getLogger('pddlstream').setLevel(logging.WARNING)` keeps only the warnings, and an application that configures logging receives the messages through its own handlers.

Property descriptions:
* **Method**: the python function that calls the algorithm
<!--* **Subroutine**: TBD-->
* **Negated streams**: whether the algorithm supports inverting test streams
* **Fluent streams**: whether the algorithm supports fluent streams that additionally condition on the fluent state
* **Wild streams**: whether the algorithm supports streams that additionally can certify ad hoc facts
<!--* **Semantic attachments (SA)**: TBD-->

### Adaptive

* **Method**: [solve_adaptive(...)](https://github.com/caelan/pddlstream/blob/da1e52aeeff9a22ef647e95b617dd71904cdbe49/pddlstream/algorithms/focused.py#L229)
* **Negated streams**: supported 
* **Fluent streams**: supported 
* **Wild streams**: supported 
<!--* **Semantic attachments (SA)**: TBD-->

### Binding

* **Method**: [solve_binding(...)](https://github.com/caelan/pddlstream/blob/da1e52aeeff9a22ef647e95b617dd71904cdbe49/pddlstream/algorithms/focused.py#L215)
* **Negated streams**: supported 
* **Fluent streams**: supported 
* **Wild streams**: supported 
<!--* **Semantic attachments (SA)**: TBD-->

### Focused

* **Method**: [solve_focused_original(...)](https://github.com/caelan/pddlstream/blob/da1e52aeeff9a22ef647e95b617dd71904cdbe49/pddlstream/algorithms/focused.py#L201)
* **Negated streams**: supported 
* **Fluent streams**: supported 
* **Wild streams**: supported 
<!--* **Semantic attachments (SA)**: TBD-->

### Incremental

* **Method**: [solve_incremental(...)](https://github.com/caelan/pddlstream/blob/da1e52aeeff9a22ef647e95b617dd71904cdbe49/pddlstream/algorithms/incremental.py#L700)
* **Negated streams**: not supported 
* **Fluent streams**: not supported 
* **Wild streams**: supported 
<!--* **Semantic attachments (SA)**: TBD-->

### Reproducible runs

With `solve(..., deterministic=True)` and a fixed `PYTHONHASHSEED` (e.g. `PYTHONHASHSEED=0`), seeded runs repeat exactly.
Decisions that otherwise depend on elapsed time, such as the adaptive algorithm's sampling budget, then count work units (stream calls and searches) instead of seconds, and saved stream statistics are not loaded.
`max_time` is still measured in seconds, so a run that stops because of it may not repeat.
Seed your samplers (`random`, `numpy.random`) as usual.

Without `deterministic=True`, the solvers load the stream statistics (success rates and overheads) saved by earlier runs, so runs also depend on what ran before.
The statistics are saved per stream file in `~/.cache/pddlstream/statistics/` (`$XDG_CACHE_HOME/pddlstream/` if set, or `$PDDLSTREAM_CACHE_DIR`); delete them to start fresh, or pass `load_statistics=False` or `save_statistics=False` to `solve()`.
Statistics that earlier versions saved in `./statistics/` are still read.

## Search Subroutines

Many (but not all) **pddlstream** algorithms have a discrete planning phase that can be implemented using any finite state-space search algorithm, such as Breadth-First Search (BFS) and Uniform-Cost Search (UCS).
However, because **pddlstream** extends PDDL, this planning phase can also be implemented by state-of-the-art classical planning algorithms, which leverage the factored structure of action languages such as PDDL to vastly improve empirical planning efficiency.
Best-first heuristic search algorithms, which automatically derive heursitics in a *domain-independent* manner, are one example class of these algorithms.

### FastDownward

**pddlstream** comes pre-packaged with [FastDownward](http://www.fast-downward.org/), a prolific library that contains many best-first heuristic search PDDL planning algorithms.
I've preconfigured a small number of effective and general search algorithms in [SEARCH_OPTIONS](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L81), which can be toggled using the keyword argument [planner=?](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L391). I've roughly ranked them in order of least lazy (lowest cost) to most lazy (lowest runtime):
* [Dijkstra/Uniform-Cost Search (UCS)](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L87) - optimal but slowest
* [hmax A*](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L91) - optimal but slow
* [lmcut A*](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L93) - optimal but slow
* [ff eager astar](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L97)
* [ff eager {1,...,5} weighted astar](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L138) - recommended (w=2)
* [ff lazy {1,...,5} weighted astar](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L129)
* [ff eager greedy](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L99)
* [ff lazy greedy](https://github.com/caelan/pddlstream/blob/4914667a13a80831cadaf115a70938e9f93b021e/pddlstream/algorithms/downward.py#L103)

The runtime of the discrete planning phase varies depending on the selected search algorithm.
For many non-adversarial problems, these algorithms will either solve a problem instantenously or, if they aren't greedy enough, not terminate within 10 minutes.
I recommend starting with a greedier configuration and moving torward a less greedy one if desired. 

### Other PDDL Planners

Any PDDL planning algorithm could be used in the place of [FastDownward](http://www.fast-downward.org/); however, a caveat is that some of these planners are only implemented to support a limited set of representational features (e.g. no conditional effects, no derived predicates, etc.), which can make both modeling more difficult and ultimately planning less efficient in many real-world (non-IPC) planning domains.
While I heavily recommend [FastDownward](http://www.fast-downward.org/), some PDDL planners that I've interfaced with in the past with some success include:
(This repository now calls only FastDownward; its old hooks for Temporal FastDownward, pyplanners, Cerberus, TPSHE, OPTIC, tflap and Forbid Iterative were removed.)

Classical Planners
* [FastDownward](http://www.fast-downward.org/)
* [FastForward](https://github.com/caelan/FastForward-X)
* [pyplanners](https://github.com/caelan/pyplanners)
* [Cerberus](https://github.com/ctpelok77/fd-red-black-postipc2018)
* [LAPTK](https://github.com/LAPKT-dev/LAPKT-public/)
* [YSHAP](https://github.com/nojhan/descarwin) <!--https://github.com/aldukeman/yahsp2-->
<!--* [pyperplan](https://github.com/aibasel/pyperplan)-->
<!-- https://github.mit.edu/caelan/stripstream/tree/master/stripstream/algorithms/search -->
<!-- https://github.mit.edu/caelan/ss/tree/master/ss/algorithms -->

Numeric Planners:
* [Metric FF](https://github.com/caelan/Metric-FF)
* [SMTPlan](https://github.com/KCL-Planning/SMTPlan)

Temporal Planners:
* [Temporal FastDownward](https://github.com/caelan/TemporalFastDownward)
* [TPSHE](https://github.com/aig-upf/temporal-planning)

Diverse Planners:
* [Forbid Iterative](https://github.com/IBM/forbiditerative)
* [kstar](https://github.com/ctpelok77/kstar)
* [symk](https://github.com/speckdavid/symk)

<!-- International Planning Competition (IPC) -->

## Resources

* [Recent Talk](https://youtu.be/YfCh5qMYGuQ)
* [Recent Overview](https://youtu.be/7xLPogztQss)
* [Recent Tutorial](https://youtu.be/JNOk1rylDpU)
* [PDDLStream Tutorial](https://web.mit.edu/caelan/www/presentations/6.881_19-11-12.pdf)
* [The AI Planning & PDDL Wiki](https://planning.wiki/)
* [Planning Domain Definition Language (PDDL)](http://users.cecs.anu.edu.au/~patrik/pddlman/writing.html)
* [Derived Predicates](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume28/coles07a-html/node18.html)
* [FastDownward](http://www.fast-downward.org/)
<!--* [PDDLStream Tutorial](https://web.mit.edu/caelan/www/presentations/6.881_TAMP.pdf)-->
