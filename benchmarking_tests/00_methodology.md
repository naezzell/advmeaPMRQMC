# Methodology and provenance

## Question

How should changes to PMR-QMC be compared so that correctness, estimator cost,
Markov-chain mixing, and total resource use are not conflated?

## Hypothesis

Equal-resource controls, independent tuning and production seeds, signed raw
time series, and content-addressed manifests are sufficient to identify which
part of the workflow produces a meaningful improvement.

## Code/Environment

- Branch: `benchmark-speedup`
- Initial branch point: `a43ecd6`
- Desktop envelope: four logical CPUs and 12 GiB RAM under WSL2
- Raw artifact threshold: 5 MiB per file
- Default per-job wall limit: two hours
- Default desktop campaign wall limit: 24 hours

The run manifest, rather than this prose, is authoritative for the exact Git
revision, compiler, Python, MPI, CPU, command, and random seeds of a run.

## Protocol

1. Generate an immutable plan and content-derived run IDs before execution.
2. Validate estimators on exactly solvable systems before ranking performance.
3. Tune warmup, sampling interval, QCPT schedule, and stopping thresholds on
   tuning seeds that are excluded from production comparisons.
4. Compare methods with equal core counts.  Report both wall time and
   core-hours.
5. Preserve sign and signed observable numerators per independent stream.
6. Report update throughput, measurement cost, expansion order, average sign,
   autocorrelation, effective sample rate, time to precision, and QCPT flow.
7. Promote a speedup only when a bootstrap 95% interval for the median
   ESS-rate or time-to-precision ratio excludes one.

Planning and execution are separate operations.  `study.py plan` expands a
human-readable JSON configuration into an immutable CSV whose run IDs hash the
model, physical parameters, method, measurements, schedule, simulation budget,
seed, and source commit.  Execution refuses a plan from a different commit.
This prevents a resumed campaign from silently mixing implementations.

### Default statistical gates

| Quantity | Default gate |
|---|---:|
| Split-chain rank-normalized R-hat | <= 1.01 |
| Minimum effective samples | 400 |
| Minimum effectively independent blocks | 50 |
| Energy-density relative standard error | 0.1% |
| Specific heat, ES, and FS relative standard error | 5% |
| Effective-gap relative standard error | 10% |
| QCPT worst adjacent acceptance | >= 0.15 |

Ground-state convergence is observable-specific.  At fixed `(L, lambda)`, the
default beta sequence is `beta/L = 0.5, 1, 2`, extended to 4 and 8 until the
last two increments agree within two pooled standard errors and the configured
precision target.

## Results

This note defines the protocol; it contains no performance result.

The first dry planning test generated the critical desktop matrix without
running QMC.  It exposed a control-count bug: schedule candidates from a mixed
beta-PT/QCPT matrix were initially applied to beta-only PT even though those
names do not alter its geometric schedule.  The resulting duplicate run IDs
were rejected, and the expansion now gives beta-only PT exactly one schedule.

## Interpretation

The historical code used conservative human-chosen update counts and a
between-rank error comparison.  The new protocol makes those choices explicit
and reproducible while retaining conservative failure states.

## Limitations

- Statistical diagnostics cannot prove global equilibration.
- A desktop ceiling is hardware- and precision-dependent.
- SSE comparisons cover shared observables only; specialized PMR estimators
  require PMR-internal measurement controls.

## Report-ready Claims

No scientific claim is made until implementation and validation runs exist.

## Open Questions

- Which observable is the first to fail beta convergence as `L` grows?
- Is critical slowing, PMR update cost, divided-difference evaluation, or the
  FS estimator the dominant desktop bottleneck?
- Does QCPT improve target-slot ESS per core-hour after crossed-weight cost?
