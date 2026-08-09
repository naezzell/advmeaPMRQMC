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
No TFIM schedule has yet been empirically ranked.

## Interpretation

Schedule selection is now reproducible and explicitly values the whole
production sweep without conflating tuning and production samples. QCPT is
therefore evaluated as a reusable sweep engine as well as a mixing aid for the
hardest coordinate.

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
- The classical dogleg does not yet have a Wolff move at its Gamma=0 slots.

## Report-ready Claims

None; the selection mechanics are tested but schedule performance is unmeasured.

## Open Questions

- Does ranking by endpoint ESS choose a materially different path than ranking
  by total sweep ESS?
- Should final production allocate equal time to slots or optimize precision
  across the desired figure grid?
