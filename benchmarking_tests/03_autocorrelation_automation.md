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
  early/late mean difference below two pooled standard errors for the raw
  signed energy numerator, raw signed driving-term numerator, sign, and (trace
  schema 3 onward) instantaneous expansion order. Keep the centered
  signed-ratio linearization for IAT estimation, where its influence-function
  interpretation is appropriate.
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
production gates without any tolerance change: corrected component-wise R-hat
was 1.015, total ESS was 205, fewer than 50 common jackknife blocks remained,
and both precision targets were missed. The adaptive decision therefore doubles
both warmup and sampling. The compact records are
`results/desktop_smoke_aee1cea9a2fae828.json` and
`results/adaptive_smoke_series.json`.

Three automatic successor rounds reached 800 warmup and 8000 sampling updates
per chain (`b163f97dd43e5b40`). ESS rose from 205 to 1280 and the energy-density
estimate moved from -0.743(36) to -0.857(19), close to the exact -0.8630.
The largest run's corrected R-hat was 1.001 and passed the drift gate. It still
failed the blocking requirement and 0.1% energy precision target, so all four
runs remain failed for defensible reasons. The former 1.027--1.051 R-hat values
are explicitly retained as superseded fields in
`results/adaptive_smoke_series.json`.

The original R-hat calculation ranked chain-by-chain centered signed-ratio
linearizations. For a discrete PMR observable, small chain-specific centering
offsets split otherwise identical energy mass points into chain-specific pooled
ranks and created a false between-chain signal. Reanalysis of the 3x3 anchor
showed conventional split R-hat near 1 and raw signed-component rank R-hat at
most 1.0003, while the old method returned about 1.04. A regression test now
covers identical discrete chains. Trace schema 3 also records instantaneous
`expansion_order`, allowing the requested q convergence gate on all new runs.

## Limitations

- The current analysis uses energy to choose the common block length; advanced
  observables may require a larger observable-specific block length.
- Existing schema-2 traces cannot retrospectively test instantaneous expansion
  order. They still test signed energy, driving-term, and sign components.
- Four-chain automatic extension and measurement-interval selection remain to
  be evaluated on nontrivial TFIM sizes. The runner now creates immutable
  doubled-resource successor runs and stops at the per-job or 24-hour cap.
- For this tiny smoke, simulator compilation took about 10.9 seconds while the
  simulator-reported wall time was 0.11 seconds. Build caching is needed before
  end-to-end timings on short jobs are interpretable.
- Campaign reanalysis currently recomputes every unchanged trace. In the 12-run
  model-control campaign, Python diagnostics took minutes while individual
  standard-representation simulations took about 12 seconds. Analysis results
  need trace-checksum-based caching and vectorized/FFT autocorrelation kernels.

## Report-ready Claims

None; the smoke data validate trace plumbing but are intentionally too short
for a performance or physics claim.

## Open Questions

- How stable are automated choices across independent tuning seeds?
- Should the production block length be the maximum IAT over all requested
  observables rather than the energy IAT alone?
- Should q and advanced-observable numerators be included in the conservative
  maximum R-hat, or reported as separate diagnostic families?
