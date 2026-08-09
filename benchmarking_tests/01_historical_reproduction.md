# Historical reproduction

## Question

Can the current implementation reproduce selected historical TFIM results with
less compute and a more explicit convergence protocol?

## Hypothesis

The current fixed sampler and fast estimator should reproduce an exactly
tractable historical anchor, while automated gates should expose which
observables—not just which runs—need more sampling.

## Code/Environment

- Simulation source: `6e030fb`; analysis correction: commit containing this
  note.
- Machine: Intel i5-6600K, four cores, WSL2 Linux, GCC/OpenMPI wrapper 9.4.
- Compact result: `results/historical_3x3_obc_beta1.json`.
- Archived-data comparison: `results/historical_3x3_archive_comparison.json`;
  generator `experiments/compare_historical.py`.
- Archive: `benchmarking_tests/artifacts/historical_anchors`; 40,207,636 bytes
  of stream traces. SHA-256 of the deterministic ordered checksum manifest:
  `5a5ee51dfb2716abf0ac88faf9a3df5fcf201cadef14d9aa1b1523840238c71e`.

## Protocol

Run the standard 3x3 OBC TFIM at `Gamma=0.5`, `beta=1` with all cheap and
advanced observables. Use four seed groups (1000, 2000, 3000, 4000); each group
contains four independent MPI chains, 100,000 warmup updates, 1,000,000
production updates, and measurement interval 100. Compare energy density and
specific heat density to full 512-dimensional exact diagonalization using the
preconfigured `max(5 SE, 0.03)` gate. Apply the production precision and
component-wise convergence gates without post-hoc changes.

## Results

Run IDs were `03a568f16f6da728`, `356e83aa7b480102`,
`41117f9436c4b660`, and `24ee3feebb20a41d`. Every seed group passed exact
energy and heat-capacity checks. Inverse-variance combination across the four
groups gives:

| Observable | PMR-QMC estimate | Exact | Production target |
|---|---:|---:|---|
| Energy density | -1.318807(632) | -1.31921749 | passed in all groups |
| Specific heat density | 0.30245(31) | 0.30085197 | passed in all groups |
| Energy susceptibility density | 0.10366(41) | — | 5% failed in all groups |
| Fidelity susceptibility density | 0.00923(56) | — | 5% failed in all groups |
| Effective gap ratio | 5.606(120) | — | 10% passed in all groups |

Component-wise R-hat ranged from 1.0000 to 1.0003. Three seed groups passed
the two-standard-error drift test; seed 1000 failed the energy-numerator drift
test. Per-group energy ESS ranged from 24,315 to 25,360. Total simulator core
time across the four groups was 0.0138 core-hours, excluding compilation and
analysis.

The same four quantities were compared directly with the archived paper row in
`data_plotting_misc/data/advmea_3by3tfim_verify_data.csv` (archive SHA-256
`939938fc5fb999fc90dac0a1062cbfd6d25c4721ccadec80472e5513a6328bd4`).
Energy, heat capacity, off-diagonal ES, and off-diagonal FS differ from the
archive by 0.55, 0.49, 0.29, and 0.14 combined standard errors, respectively.
All therefore pass the required three-combined-SE historical gate without a
tolerance change.

## Interpretation

The current implementation obtains reliable energy and heat capacity for this
small anchor very cheaply. ES and FS remain the controlling precision cost:
although their combined estimates are informative, no individual seed group
met the preregistered 5% relative-error target. The ratio passed its looser 10%
target because covariance is retained by the joint blocked jackknife.

This is both an exact validation and a historical reproduction for the four
shared standard observables. It also confirms that adaptive extension should
be observable-specific rather than treating energy convergence as
certification of FS.

The archived row reports 40 billion aggregate production updates and 52,734
CPU-seconds, versus 16 million production updates and about 49.8 simulator
core-seconds across the four current seed groups. That contrast is encouraging
but is not yet a speedup claim: the archived job measured additional custom
A/B observables, used a different measurement interval and machine, and
obtained roughly five-to-six times smaller standard errors for the shared
quantities.

## Limitations

- The inverse-variance line is a preliminary combination of independently
  jackknifed run estimates, not a single 16-chain joint jackknife.
- Schema-2 traces did not record instantaneous q; schema 3 adds it for future
  convergence tests.
- One seed group's drift failure prevents calling the full production protocol
  passed, despite exact agreement.
- Compilation and Python analysis time are excluded from the simulator core
  time stated above.
- Historical and current resource counts are not a matched time-to-precision
  protocol, so their raw ratios must not be reported as estimator or sampler
  speedups.

## Report-ready Claims

At 3x3 OBC, `Gamma=0.5`, `beta=1`, current fixed PMR-QMC agrees with exact
diagonalization for energy density and specific heat density in four
independent seed groups under the preregistered tolerance. Energy, specific
heat, off-diagonal ES, and off-diagonal FS also reproduce the archived paper
row within 0.55 combined standard errors or less. Per-seed-group ES and FS need
more samples than energy and heat capacity to meet their production precision
target.

## Open Questions

- Does extending production alone resolve ES/FS precision, or does their
  measurement interval need retuning?
- Should the final ensemble estimate use a joint 16-chain blocked jackknife?
