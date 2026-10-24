# ADR-002: FastDownward 26.6 and a planner backend

Status: accepted. Date: 2026-10-24.

## Context

pddlstream's `downward` submodule is a fork, `caelan/downward`, at upstream release-20.06 plus 599 commits (April
2022). FastDownward has since released 26.6.0. The translator is now an installable package
(`fast-downward.translate`), and the search component has changed its option syntax. The V2 plan asked whether to
move to 26.6, support both, or stay. A `PlannerBackend` (translate, ground, search, cost scale) was to be added only
if both were supported.

## Spike

The spike ran in a scratch directory and committed no code.

- **Build.** 26.6 built with the same toolchain as the fork (`build.py release -j2`).
- **Translator.** A loader imported `fast_downward.translate` under the module names that pddlstream uses (`pddl`,
  `normalize`, `instantiate`, ...) and added shims for the API differences below. The fast suite then ran with the
  26.6 translator and the fork's search.
- **Search.** The fork's translator produced the SAS files, and 26.6's search ran them with pddlstream's 27 search
  configurations. The fast suite ran with this mix too.

## Findings: translator

The translator is far from a drop-in replacement. pddlstream depends on features of the fork that upstream does not
have. Shims covered these:

- **Cost scale.** The fork multiplies costs by `COST_SCALE` in `NumericConstant` and rounds them up. 26.6 rejects
  fractional costs.
- **Action groundings.** `Action.instantiate` and `Effect.instantiate` take a `predicate_to_atoms` index, and
  `instantiate.get_atoms_by_predicate` builds it.
- **Back-references.** Grounded actions keep `action` and `var_mapping`, and grounded axioms keep `axiom` and
  `var_mapping`.
- **Package layout.** The modules live in a package, and options are an object (`get_options()`), not module
  attributes.
- **Parser.** Every parser function takes an error-reporting context first. `parse_task_pddl` became
  `parse_problem_pddl` (with constant names). The parser rejects quantifiers without variables, which pddlstream
  generates.
- **Return values.** `explore` also returns the instantiated goal, and `fact_groups.compute_groups` requires the
  negated goal atoms.

One more fork feature was left unshimmed: grounded actions keep `effect_mappings`, the variable mapping of each
conditional effect that pddlstream uses to recover which effects fired. Supporting it means reimplementing
`Action.instantiate`. With every other shim in place, 98 of 257 fast tests failed, nearly all on this attribute.

26.6 still hashes conditions by their class object, so pddlstream's patch that hashes class names (needed for
determinism) still applies.

## Findings: search

- **SAS format.** Both versions use SAS version 3.
- **Axioms.** 26.6 rejects axioms that the fork's translator writes for derived predicates: "Value of variable
  affected by axiom must be default value 1 in precondition, but is 0".
- **Option syntax.** Heuristics no longer take `transform=adapt_costs(cost_type=X)`. The mechanical rewrite
  `h(transform=adapt_costs(cost_type=X))` → `eval_modify_costs(h(), cost_type=X)`, plus dropping
  `transform=no_transform()`, makes all 27 configurations run.
- **Fast suite with the fork's translator and 26.6's search.** 245 of 257 tests passed, and every recorded output
  that ran was unchanged. The 12 failures are problems with derived predicates (the axiom check above) and
  continuous problems that rely on them. The suite took 13 minutes instead of 1, because failing solves ran to their
  time limits.

## Decision

Stay on the fork for 2.0, and add no `PlannerBackend` abstraction yet. Neither upgrade works without changes inside
pddlstream, and an abstraction with one working implementation would have no consumer.

## Path to 26.x (for 2.1)

1. **Search first.** It is the smaller change.
   - Make pddlstream's SAS encoding of derived predicates satisfy 26.6's axiom rule, so that derived variables have
     their default value in axiom preconditions.
   - Choose the search-option dialect by FastDownward version (the rewrite above).
   - The search binary can then be upgraded on its own, with the fork's translator.
2. **Then the translator.** Move the fork's patches into pddlstream.
   - pddlstream already reimplements grounding steps (`fastdownward/grounding.py`,
     `algorithms/scheduling/reinstantiate.py`). There it can compute the back-references, `effect_mappings` and the
     predicate index itself.
   - Scale costs before translation, and call the context-taking parser functions.
   - pddlstream then works with the unmodified upstream translator package.
3. **Re-record the outputs once** (a deliberate output change), and compare the benchmark timings before switching
   the default.

Both versions are GPL-3.0, so licensing does not affect the choice.
