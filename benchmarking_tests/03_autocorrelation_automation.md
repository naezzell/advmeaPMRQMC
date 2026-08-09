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
- Correctness protocol: `STREAM-SMOKE-20260809`; compact result
  `results/stream_trace_smoke.json`. This validator predates content-derived
  campaign run IDs and is not used for a performance claim.

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

The independent-stream output was exercised end-to-end on a two-spin TFIM.
Fixed PMR at beta 3 and PT at beta 0.25, 0.6, 1.3, and 3 agreed with exact
diagonalization under the preconfigured `max(0.03, 6 SE)` smoke tolerance. The
three PT edges had acceptance 0.726, 0.659, and 0.501, with 134 round trips.
All expected per-rank trace files passed schema checks. This deliberately short
run validates wiring and gross correctness only; its uncertainty is too large
for efficiency comparisons.

The first content-addressed campaign smoke, run `aee1cea9a2fae828`, used four
chains of a 2x2 OBC TFIM at `(Gamma,beta)=(1,0.5)`. It passed the five-SE exact
diagonalization gates for energy density and specific heat density, but failed
production gates without any tolerance change: R-hat was 1.027, total ESS was
205, fewer than 50 common jackknife blocks remained, and both precision targets
were missed. The adaptive decision therefore doubles both warmup and sampling.
The compact record is `results/desktop_smoke_aee1cea9a2fae828.json`.

Three automatic successor rounds reached 800 warmup and 8000 sampling updates
per chain (`b163f97dd43e5b40`). ESS rose from 205 to 1280 and the energy-density
estimate moved from -0.743(36) to -0.857(19), close to the exact -0.8630.
Nevertheless, R-hat stayed above threshold (1.051 in the largest run) and the
0.1% energy precision target remained far away, so all four runs correctly
remain failed. See `results/adaptive_smoke_series.json`.

## Limitations

- The current analysis uses energy to choose the common block length; advanced
  observables may require a larger observable-specific block length.
- Per-rank/per-ladder traces are required before multi-stream R-hat and drift
  gates can be exercised across four independent production ladders. The trace
  schema is now emitted, but the campaign-level cross-run gate is pending.
- Four-chain automatic extension and measurement-interval selection remain to
  be evaluated on nontrivial TFIM sizes. The runner now creates immutable
  doubled-resource successor runs and stops at the per-job or 24-hour cap.
- For this tiny smoke, simulator compilation took about 10.9 seconds while the
  simulator-reported wall time was 0.11 seconds. Build caching is needed before
  end-to-end timings on short jobs are interpretable.

## Report-ready Claims

None; the smoke data validate trace plumbing but are intentionally too short
for a performance or physics claim.

## Open Questions

- How stable are automated choices across independent tuning seeds?
- Should the production block length be the maximum IAT over all requested
  observables rather than the energy IAT alone?
- Why does rank-normalized R-hat remain above 1.01 for this small discrete
  model even after the energy mean approaches the exact result? Compare raw,
  folded, and bulk/tail R-hat implementations before production.
