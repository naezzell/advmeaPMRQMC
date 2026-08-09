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
- Empirical run IDs: pending.

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

Instrumentation and protocol generation are implemented. The two-spin
fixed/PT exact smoke still passes after timing was added. No estimator timing
result is claimed yet.

## Interpretation

Separating measurement core-seconds from total runtime will show whether the
asymptotic estimator improvement matters end to end or merely shifts a
subdominant component.

## Limitations

- The first matrix measures every update by design and is not itself an
  optimized production interval.
- Slow L=3 cases may hit the ten-minute microbenchmark cap; such failures will
  be retained as right-censored observations.

## Report-ready Claims

None.

## Open Questions

- At what q does measurement time cease to be negligible?
- Does ES+FS together reuse enough intermediate work to beat separate passes?
