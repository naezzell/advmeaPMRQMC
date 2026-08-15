# Fast fidelity-susceptibility estimator

## Question

Does the reduced asymptotic complexity produce an end-to-end improvement at
the expansion orders encountered in TFIM campaigns?

## Hypothesis

The fast estimator should agree configuration by configuration with the
reference estimator and become increasingly valuable as q grows, but at small
q measurement overhead may be negligible relative to PMR updates and process
startup.

## Code/Environment

- Fast and reference implementations: `mainqmc.hpp`.
- Runtime control: protocol `advanced` versus `advanced_slow`.
- Timing matrix: `configs/estimator_microbenchmark.json`.
- Fast run IDs: `c7f8e45657933304`, `42080dd0c521a17b`,
  `19f2ff510a7cd97e`, `eff7485cc3712251`.
- Reference run IDs: `43f10e1bf6f061cd`, `4ee1ee6f395a60b3`,
  `e5ec042a8d25034e`, `68df9d88de7c4677`.
- Machine-readable result: `results/estimator_l2_ensemble.json`.

## Protocol

At L=2 and 3 near Gamma=3.044 with beta/L=1, compare matched four-seed runs
with no observables, energy/heat capacity, ES only, FS only, ES+FS fast, and
ES+FS reference protocols. Measure every update to expose estimator cost.
Record cumulative per-rank measurement core-seconds separately from simulation
wall/core-hours and retain q, update rate, ESS, and precision.

Fast and reference protocols use the same RNG seeds and consume no RNG during
measurement, so their Markov paths must match. A correctness claim requires
matching signed trace values on identical configurations before timing ratios
are considered. A speedup claim still requires a seed-ensemble bootstrap 95%
interval excluding one.

## Results

At L=2, Gamma=3.044, beta=2, mean q was 23.1--23.9 and observed maximum q was
44. Across four seeds, every fast/reference pair matched all 40,000 trace rows
and 34 scientific columns; the largest absolute discrepancy was
`5.7e-14`. The fast estimator's median measurement-component speedup was 4.69
with a deterministic paired-bootstrap 95% interval [4.62, 5.45]. Its median
launch-plus-simulation wall speedup was 3.87 [3.57, 4.31]. Core counts and
Markov paths were identical, so the latter is also the ESS/core-hour ratio for
each matched pair.

In the seed-1400 protocol decomposition, measurement core-seconds were 3.00 for
no observables, 2.90 for energy/heat capacity, 3.18 for ES-only, 10.74 for
FS-only, 10.96 for fast ES+FS, and 51.87 for reference ES+FS. Thus FS dominated
measurement cost at this q, while adding ES to FS was negligible.

## Interpretation

At q around 23, the asymptotic improvement is already large enough to reduce
end-to-end simulation wall time under the deliberately measurement-heavy
every-update protocol. This does not imply the same wall ratio after production
measurement spacing is optimized.

## Limitations

- The first matrix measures every update by design and is not itself an
  optimized production interval.
- This report-ready speedup is scoped to L=2 and q up to 44; L=3 and larger-q
  scaling remain unmeasured.
- With only four paired seeds, the percentile bootstrap interval is discrete;
  more seeds would better resolve timing variability.
- Slow L=3 cases may hit the ten-minute microbenchmark cap; such failures will
  be retained as right-censored observations.

## Report-ready Claims

For the L=2 TFIM microbenchmark at Gamma=3.044 and beta=2, the fast FS
implementation was configuration-wise equivalent to the reference estimator
and reduced median measurement time by 4.69x (paired-bootstrap 95% interval
[4.62, 5.45]) and end-to-end simulation wall time by 3.87x [3.57, 4.31] when
measuring every update.

## Open Questions

- At what q does measurement time cease to be negligible?
- Does ES+FS together reuse enough intermediate work to beat separate passes?
