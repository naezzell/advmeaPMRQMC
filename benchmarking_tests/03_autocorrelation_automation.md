# Autocorrelation and convergence automation

## Question

Can warmup, measurement spacing, production length, and beta convergence be
selected without human inspection?

## Hypothesis

Signed-ratio diagnostics applied to independent streams can replace visual
warmup and thinning choices, while a joint blocked jackknife can preserve the
covariance needed by nonlinear heat-capacity and gap estimators.

## Code/Environment

- Implementation: `experiments/study_stats.py` (commit containing this note).
- Deterministic tests: `tests/study_stats_test.py`.
- Empirical run IDs: none yet; this slice validates analysis mechanics rather
  than simulation performance.

## Protocol

- Linearize every reweighted observable as `signed_numerator - estimate*sign`
  before estimating its autocorrelation.
- Estimate integrated autocorrelation time with Geyer's initial-positive
  sequence and reject a trace when the configured lag cap is reached.
- Independently require a dyadic blocking-error plateau with at least 50
  blocks.
- On multiple streams, require split rank-normalized R-hat at most 1.01 and an
  early/late mean difference below two pooled standard errors.
- Choose a conservative jackknife block length of ten times the largest
  accepted energy IAT and compute energy, heat capacity, energy
  susceptibility, fidelity susceptibility, and their effective-gap ratio from
  each common leave-one-block-out sample.
- Declare beta convergence only after two successive beta increments satisfy
  both a two-standard-error consistency test and the observable's production
  precision target.

## Results

Five deterministic unit tests pass. They cover signed ratios, relative IAT for
independent and AR(1) traces, converged and shifted-chain R-hat, covariance-
preserving nonlinear jackknife estimates, and the two-increment beta rule.
The pre-existing seven campaign-planner tests also pass after integration.

## Interpretation

The core statistical decisions are now machine-readable in each run's
`analysis.json`, and campaign-level `summary.csv`, `summary.json`, and
`beta_convergence.json` files can be regenerated from archived traces. This is
an implementation result, not evidence that the defaults are efficient for
TFIM production runs.

## Limitations

- The current analysis uses energy to choose the common block length; advanced
  observables may require a larger observable-specific block length.
- Per-rank/per-ladder traces are required before multi-stream R-hat and drift
  gates can be exercised by the production executables.
- Four-chain automatic extension and measurement-interval selection remain to
  be connected to the campaign runner.

## Report-ready Claims

None; simulation data have not yet been collected.

## Open Questions

- How stable are automated choices across independent tuning seeds?
- Should the production block length be the maximum IAT over all requested
  observables rather than the energy IAT alone?
