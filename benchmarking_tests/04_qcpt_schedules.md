# QCPT schedule search

## Question

Which short beta-Gamma paths improve effective sampling after crossed-weight
overhead is included, while also producing useful marginals for the planned
beta and Gamma sweeps?

## Hypothesis

Mixed schedules can improve transport near difficult points, and their
non-endpoint samples can reduce the total cost of production sweeps. The best
endpoint schedule need not maximize total sweep ESS per core-hour, so both
metrics must remain visible.

## Code/Environment

- Candidate generation: `qcpt_schedule()` in `experiments/study.py`.
- All-slot analysis and selection: `experiments/study_stats.py`.
- Desktop tuning matrix: `configs/desktop_pilot.json`, which declares the
  production grid `Gamma={2.8,3.044,3.2}` and `beta/L={0.5,1,2}` and selects
  by objective-grid ESS/core-hour.
- Exact smoke matrix: `configs/qcpt_schedule_smoke.json`; L=2, Gamma=1,
  beta=2, five schedules, four tuning seeds, and a declared eight-coordinate
  `(Gamma,beta/L)` sweep objective.
- Empirical run IDs: pending.

## Protocol

Compare four-slot pure-beta, geometric mixed paths with powers 1/2, 1, and 2,
and a hot classical dogleg. Use four tuning seeds (1100, 2100, 3100, 4100).
Each stationary slot is analyzed as a production marginal at its own
`(Gamma,beta)` coordinate.

Reject a candidate unless it has four independent seeds, positive mean sign,
no qmax event, converged per-slot traces, worst-edge acceptance at least 0.15,
and at least one observed round trip across the seed ensemble. Rank survivors
by both target-slot and sweep-wide utility: median endpoint ESS/core-hour,
total sweep ESS/core-hour, and coverage of the requested `(Gamma,beta)` figure
grid remain separate columns rather than being collapsed prematurely. The
production decision must account for the fact that beta and Gamma marginals
are themselves desired physics data, including ground-state convergence—not
discarded tempering overhead. `analyze` writes
`schedule_ranking.csv/json`; a selected winner generates
`production_plan.csv` with disjoint seeds 5100, 6100, 7100, and 8100.

The selection policy is explicit campaign configuration: `target`, `sweep`,
or `objective`. Objective mode sums ESS only at stationary slots matching the
declared figure grid (with an explicit numerical tolerance), reports the
covered fraction, and breaks equal objective efficiency by coverage and then
endpoint efficiency. L-specific objective grids are supported through `by_L`.

## Results

The deterministic selector tests confirm that a high-ESS candidate with a
0.10 worst-edge acceptance is rejected in favor of a lower-ESS candidate that
passes the 0.15 transport gate; objective selection chooses a lower-endpoint
schedule when it provides more ESS on the requested figure grid; and only
coordinates in that grid contribute to objective ESS and coverage. The
264-row desktop plan is byte-for-byte deterministic across repeated planning
(SHA-256 `510bb9daedd50e8349f99fa17fd5b5b5ec696f60d0c913ec42f15b406e1e7f05`).
The exact L=2 campaign then completed an adaptive two-level search. At 50,000
production updates every schedule had at least one blocking failure. At
100,000, pure-beta and all three diagonal paths pass sign, qmax, transport, and
per-slot convergence gates; the classical dogleg is rejected by its 0.095
worst-edge acceptance despite many round trips. All 80 tuning marginals agree
with exact diagonalization.

Pure-beta is selected for the declared sweep objective. Its median objective
ESS/core-hour is 1.519 million with 50% grid coverage. The diagonal schedules
cover 25% and yield 0.609--0.714 million objective ESS/core-hour. Matched-seed
pure/diagonal objective ratios are 2.25--2.69, with every bootstrap 95%
interval excluding one. This is predominantly a useful-marginal coverage gain:
at the target endpoint, pure/diagonal median ratios are 0.95--1.06 and every
interval includes one.

The dogleg also covers 50% and has 1.264 million objective ESS/core-hour, so its
pure/dogleg interval includes one; it loses on the preregistered acceptance
gate. Four disjoint pure-beta production runs (`1ce2cac962450bc5`,
`70f35b98e79a5a51`, `f0e1b9776116201c`, `fcb005adea87281d`) pass exact and
all per-slot convergence gates but miss the strict precision target. Compact
results and 208,238,609 bytes of raw-trace checksums are recorded in
`results/qcpt_schedule_smoke.json`.

## Interpretation

Schedule selection is now reproducible and explicitly values the whole
production sweep without conflating tuning and production samples. QCPT is
therefore evaluated as a reusable sweep engine as well as a mixing aid for the
hardest coordinate.

This first empirical ranking supports the user's sweep interpretation of QCPT:
pure-beta wins because every slot lies on the requested beta curve, not because
it is demonstrably faster at the hardest endpoint. Mixed paths should be built
from desired figure coordinates if their intermediate marginals are expected
to pay for crossed-weight overhead.

## Limitations

- No bootstrap uncertainty is yet attached to the schedule ranking.
- Four slots cannot cover a dense two-dimensional grid in one desktop ladder;
  multiple ladder paths are still required.
- Exact coordinate matching means schedules intended to contribute production
  marginals should be designed from the declared grid, rather than merely pass
  near it.
- Adaptive resource levels are ranked separately: warmup, production updates,
  and measurement interval are part of candidate identity and the selected
  production template. A failed short parent therefore cannot poison its
  successful extension or cause production to copy the wrong budget.
- Executed adaptive rows are recovered from immutable run manifests, not a
  transient `extension_plan.csv`. Production promotion and further extensions
  therefore retain the exact successful resource level even after the next
  extension plan is generated.
- Production rows preserve the winning physical/resource specification but are
  content-identified at the current clean source revision, so provenance gates
  do not reject promotion merely because analysis/ranking code was committed
  after tuning.
- The classical dogleg does not yet have a Wolff move at its Gamma=0 slots.
- Each run contains one four-slot ladder; the four tuning seeds are ranked as
  an ensemble, but a joint cross-seed R-hat is not yet computed.
- Crossed-weight evaluation is about one quarter of four-core simulator time in
  this smoke, while per-run compilation dominates end-to-end campaign time.
- This compares QCPT schedules with one another, not yet QCPT against matched
  fixed-PMR and beta-only controls.

## Report-ready Claims

At L=2, Gamma=1, beta=2, all five QCPT schedule families produce exact
stationary marginals. Pure-beta is the sweep-objective winner after the dogleg
fails the acceptance gate; its advantage over diagonal paths is resolved for
requested-grid ESS/core-hour but not for hardest-endpoint ESS/core-hour. This
is a workflow/schedule result, not evidence of critical-point acceleration.

## Open Questions

- Does ranking by endpoint ESS choose a materially different path than ranking
  by total sweep ESS?
- Should final production allocate equal time to slots or optimize precision
  across the desired figure grid?
