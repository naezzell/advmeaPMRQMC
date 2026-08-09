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
- Desktop tuning matrix: `configs/desktop_pilot.json`.
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

## Results

The deterministic selector test confirms that a high-ESS candidate with a
0.10 worst-edge acceptance is rejected in favor of a lower-ESS candidate that
passes the 0.15 transport gate. No TFIM schedule has yet been empirically
ranked.

## Interpretation

Schedule selection is now reproducible and explicitly values the whole
production sweep without conflating tuning and production samples. QCPT is
therefore evaluated as a reusable sweep engine as well as a mixing aid for the
hardest coordinate.

## Limitations

- No bootstrap uncertainty is yet attached to the schedule ranking.
- Four slots cannot cover a dense two-dimensional grid in one desktop ladder;
  multiple ladder paths are still required.
- The classical dogleg does not yet have a Wolff move at its Gamma=0 slots.

## Report-ready Claims

None; the selection mechanics are tested but schedule performance is unmeasured.

## Open Questions

- Does ranking by endpoint ESS choose a materially different path than ranking
  by total sweep ESS?
- Should final production allocate equal time to slots or optimize precision
  across the desired figure grid?
