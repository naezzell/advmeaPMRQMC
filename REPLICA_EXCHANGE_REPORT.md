# Beta-only replica exchange: implementation and validation report

## Executive summary

The `replicate-exchange` branch adds beta-only parallel tempering (PT) to the
PMR QMC code that was present on `speedup-fs`. The implementation keeps MPI
ranks at fixed beta slots and exchanges complete PMR configurations between
neighboring slots using an exact four-weight Metropolis ratio.

During validation, the first convergence experiment exposed a correctness
bug: the exchange acceptance ratio was evaluated using the live QMC
configuration, but the serialized configuration sent to the neighboring rank
was stale. This violated detailed balance and caused the cold PT slot to
converge to the wrong value. The exchange routine now serializes the live PMR
state immediately before evaluating and communicating an exchange.

After the correction:

- A two-spin transverse-field Ising model agrees with direct 4x4
  diagonalization at every beta in the ladder.
- A negative-control run with the stale-snapshot code fails by 20 to 56
  standard errors, showing that the small test is sensitive to the bug.
- A ten-spin random 3-regular TFIM agrees with exact 1024x1024 diagonalization
  at every beta tested.
- On that ten-spin instance, PT is correct but slower than an equal-rank
  fixed-beta calculation. It is therefore a correctness result, not evidence
  of a PT speedup.

The implementation and validation suite were committed as:

```text
f2f8e98 Add validated beta-only parallel tempering
```

## Branch comparison

The comparison in this report is:

```text
speedup-fs (5c542d6) -> replicate-exchange (f2f8e98)
```

The branch adds or changes 20 source files, with 3,041 insertions and 214
deletions. Generated binaries, plots, and benchmark output were not committed.

The main changes are:

| Area | Files | Purpose |
|---|---|---|
| PT executable | `PMRQMC_pt_mpi.cpp` | MPI beta-slot layout, exchanges, measurements, checkpointing, and diagnostics |
| PMR state support | `mainqmc.hpp`, `divdiff.hpp` | Runtime beta/tau, cross-beta log weights, PMR snapshots, and stable extended-exponent logarithms |
| Fixed-beta control | `PMRQMC_mpi.cpp` | Signed time-series output for like-for-like convergence comparisons |
| Schedule/layout | `pt_schedule.hpp` | Schedule parsing, validation, hashing, exchange-ratio helper, and layout checks |
| Experiment drivers | `experiments/pt_*.py`, `experiments/exact_diagonalization.py` | Reproducible runs, exact references, plots, and parameter sweeps |
| Minimal validation | `experiments/validate_pt_minimal.py` | Dependency-free two-spin exact/PT/fixed-beta correctness test |
| Tests/build/docs | `tests/`, `Makefile`, `README.md` | Unit tests, build targets, usage, and validation instructions |

## What the PT implementation does

### Fixed beta slots and moving configurations

Each MPI rank owns a fixed temperature slot. A schedule row contains `beta`
and optionally `tau`; omitted `tau` defaults to `beta/2`. Neighboring ranks
periodically propose to exchange their PMR configurations. Beta values do not
move between ranks.

With multiple independent ladders, the MPI world is split in two ways:

- A ladder communicator performs neighboring exchanges.
- A temperature communicator combines measurements from independent ladders
  occupying the same beta slot.

The MPI size must be an integer multiple of the number of beta values.

### Runtime beta and tau

The original compile-time `beta` and `tau` macros are retained as defaults but
copied into mutable `run_beta` and `run_tau` values. Each PT rank configures
its factorial and divided-difference tables for its assigned slot. The
single-temperature executables continue to use the generated defaults.

### Exact exchange acceptance ratio

For configurations `x` and `y` at neighboring slots `i` and `j`, the code
uses

```text
log R = log pi_j(x) + log pi_i(y) - log pi_i(x) - log pi_j(y).
```

`GetLogWeightAtBeta` evaluates a PMR configuration at another beta without
changing the active chain. `ExExFloat::log_abs` evaluates the logarithm without
first converting an extended-exponent value to an ordinary double.

Alternating even and odd edges are proposed at the configured exchange
cadence. This ensures that a rank participates in at most one neighboring
exchange at a boundary.

### PMR configuration snapshots

The exchanged snapshot contains the base spin state, expansion order `q`, and
operator sequence. After import, the receiving rank reconstructs derived PMR
state and evaluates it using the beta assigned to that rank.

### Measurements and output

The PT executable produces:

- Per-temperature observables and standard errors.
- Adjacent-edge attempts, acceptances, and acceptance fractions.
- Trajectory occupancy, endpoint visits, and complete round trips.
- Optional per-measurement time series containing the sign, raw observable,
  signed numerator, updates, and elapsed time.

The fixed-beta MPI executable has matching time-series support. Running means
are calculated as

```text
sum(signed observable) / sum(sign),
```

not as the unweighted mean of raw observable values.

### Checkpointing

PT checkpoints include the PMR snapshot, RNG state, completed update count,
exchange parity, trajectory/flow state, and swap counters. Checkpoints are
written atomically and validated against the schedule hash, rank layout, and
expected record sizes before resuming.

## The stale-snapshot correctness bug

### Observed symptom

In the original MAX2SAT convergence run, the exact cold-beta transverse
magnetization was `0.485347`, while the PT cold slot approached approximately
`0.526744`. The fixed-beta result was `0.475305`. Healthy-looking swap rates
and many reported round trips initially made the PT result appear plausible.

### Root cause

The PMR configuration was serialized once before entering the update loop.
Local QMC updates then changed the live state, but the serialized `current`
buffer was not refreshed before an exchange.

Consequently:

1. Cross-beta weights were computed from the live, updated configuration.
2. The Metropolis decision was made from those live weights.
3. A different, stale configuration was sent and imported when the proposal
   was accepted.

The accepted transition therefore did not correspond to the transition whose
probability had been evaluated. Detailed balance with respect to the product
of beta-slot distributions was lost.

Round trips did not detect this problem because trajectory labels moved after
accepted proposals even though the wrong serialized states were exchanged.

### Correction

The participating ranks now call

```cpp
export_PMR_snapshot(current);
```

at the exchange boundary, immediately before calculating cross-beta weights
and sending the snapshot. Thus the state used in the acceptance ratio is the
state actually exchanged.

Existing results produced by the stale-snapshot implementation are invalid
and cannot be repaired through post-processing.

## Validation strategy

Validation was performed in increasing order of system size:

1. Unit tests of schedule parsing, layout, exchange-ratio ordering, endpoint
   tracking, signed-ratio time series, and convergence logic.
2. A two-spin model with direct 4x4 diagonalization at every beta.
3. A deliberate negative control using the stale-snapshot code.
4. A ten-spin TFIM with direct 1024x1024 diagonalization.

The current test status is:

```text
7 Python/C++ tests passed
PMRQMC_pt_mpi.cpp compiled successfully with mpicxx
```

The remaining compiler messages are pre-existing warnings about `sprintf`
and a dangling `else`; they are not PT validation failures.

## Two-spin validation

### Model and run

The minimal model is

```text
H = -J Z1 Z2 - Gamma (X1 + X2)
J = 1, Gamma = 0.7
observable = (X1 + X2) / 2
betas = 0.25, 0.6, 1.3, 3.0
```

The run used 5,000 thermalization updates, 100,000 sampling updates, and an
exchange attempt after every update. The unusually frequent exchange cadence
strongly exercises the snapshot path. Exact values were computed with a
standard-library Jacobi diagonalization, which is separately checked against
the closed-form two-spin result.

### Corrected results

| Method | Beta | Exact | QMC mean | Standard error | Absolute error | Result |
|---|---:|---:|---:|---:|---:|---|
| Fixed beta | 3.0 | 0.729466 | 0.741705 | 0.013361 | 0.012239 | Pass |
| PT | 0.25 | 0.169804 | 0.170456 | 0.010133 | 0.000652 | Pass |
| PT | 0.6 | 0.360402 | 0.363520 | 0.008105 | 0.003118 | Pass |
| PT | 1.3 | 0.561578 | 0.581131 | 0.008190 | 0.019553 | Pass |
| PT | 3.0 | 0.729466 | 0.726283 | 0.009081 | 0.003183 | Pass |

Adjacent swap acceptances were `0.763`, `0.628`, and `0.453`. The run recorded
11,727 complete round trips.

### Negative control with stale snapshots

The same experiment was repeated after deliberately removing the snapshot
refresh. It failed at every beta:

| Beta | Exact | Stale-snapshot PT | Standard error | Error in SE units |
|---:|---:|---:|---:|---:|
| 0.25 | 0.169804 | 0.046982 | 0.006082 | 20.2 |
| 0.6 | 0.360402 | 0.174248 | 0.005892 | 31.6 |
| 1.3 | 0.561578 | 0.253185 | 0.005529 | 55.8 |
| 3.0 | 0.729466 | 0.359227 | 0.008748 | 42.3 |

This establishes that the minimal validation is capable of detecting the
specific exchange bug rather than merely producing a passing smoke test.

Run the corrected validation with:

```text
make validate-pt
```

It writes a JSON report, raw convergence CSV, and plots versus updates per
replica and elapsed time in a new temporary directory.

## Most recent validation: ten-spin TFIM

### Model and parameters

The latest validation used a deterministic random 3-regular TFIM:

```text
H = -sum_(i,j in edges) Zi Zj - Gamma sum_i Xi
n = 10
Gamma = 1.0
graph seed = 2019
observable = (1/n) sum_i Xi
betas = 0.1, 0.25, 0.55, 1.2, 3.0
target beta = 3.0
independent PT ladders = 1
MPI ranks in each comparison = 5
thermalization updates = 20,000
sampling updates = 200,000
measurement interval = 20 updates
exchange interval = 10 updates
```

The Hilbert-space dimension is 1,024. Dense exact diagonalization gave a
ground-state energy of `-16.692471` and an exact beta-3 transverse
magnetization of `0.343396331`.

### PT marginal at every beta

| Beta | Exact | PT mean | PT standard error | Absolute error |
|---:|---:|---:|---:|---:|
| 0.1 | 0.098678 | 0.092313 | 0.004509 | 0.006365 |
| 0.25 | 0.229906 | 0.229706 | 0.004901 | 0.000200 |
| 0.55 | 0.358519 | 0.351198 | 0.005961 | 0.007321 |
| 1.2 | 0.345882 | 0.345416 | 0.006106 | 0.000466 |
| 3.0 | 0.343396 | 0.335559 | 0.005886 | 0.007838 |

All five PT marginals are statistically consistent with direct
diagonalization. The largest discrepancy is about 1.4 reported standard
errors.

### Cold-slot comparison

| Metric | Fixed beta | PT cold slot |
|---|---:|---:|
| Exact value | 0.343396 | 0.343396 |
| Final estimate | 0.341318 | 0.335559 |
| Block standard error | 0.002815 | 0.005249 |
| Absolute error | 0.002079 | 0.007838 |
| Sustained convergence update | 84,400 | 136,800 |
| Integrated autocorrelation, measurements | 5.906 | 3.610 |
| Estimated effective samples | 8,466 | 2,770 |
| External run wall time, seconds | 4.447 | 6.901 |
| Effective samples per external second | 1,904 | 401 |

The convergence criterion is the first checkpoint after which the cumulative
mean never again leaves its method-specific accuracy band. The accuracy bands
were approximately `0.0100` for fixed beta and `0.0105` for PT.

The time recorded inside the executables at the convergence update was about
1.38 seconds for fixed beta and 3.93 seconds for PT. The external wall times
above also include short-run MPI launch and process-management overhead.

### Exchange diagnostics

| Neighboring edge | Beta pair | Attempts | Accepted | Acceptance |
|---:|---|---:|---:|---:|
| 0 | 0.1 <-> 0.25 | 11,000 | 6,096 | 0.554 |
| 1 | 0.25 <-> 0.55 | 11,000 | 2,625 | 0.239 |
| 2 | 0.55 <-> 1.2 | 11,000 | 2,558 | 0.233 |
| 3 | 1.2 <-> 3.0 | 11,000 | 2,016 | 0.183 |

The run completed 368 round trips. No rank reached `qmax`; the maximum
observed expansion order was 28 with `qmax = 300`. The mean sign was 1.0 in
both the fixed and PT cold-slot samples.

### Interpretation

This experiment supports correctness: each beta slot samples the expected
thermal observable, including the cold slot that previously showed bias.

It does not show a speedup. The fixed-beta control puts all five MPI ranks at
beta 3, while the one-ladder PT run spends only one rank at beta 3 and uses the
other four to enable temperature movement. On this small, readily mixing
TFIM, that trade is unfavorable. Fixed beta reaches the accuracy band earlier
in both updates and time and produces more effective samples per second.

A fair claim that PT improves difficult-instance mixing would require harder
instances, several independent graph and RNG seeds, multiple ladder designs,
and ranking only configurations that first pass exact correctness checks at
small `n`.

## Reproducing the ten-spin experiment

With NumPy available for exact diagonalization:

```text
python3 experiments/pt_convergence_gap.py /tmp/pt_tfim_n10_validation \
  --model tfim --n 10 --gamma 1.0 --seed 2019 \
  --rng-seed-offset 1000 \
  --betas 0.1,0.25,0.55,1.2,3.0 --ladders 1 \
  --Tsteps 20000 --steps 200000 --steps-per-measurement 20 \
  --updates-per-exchange 10 --qmax 300 --nbins 100 \
  --block-measurements 20 --stability-blocks 10 \
  --tolerance-sigma 2.0 --absolute-tolerance 0.01 \
  --exact-reference --oversubscribe
```

With Matplotlib available, generate convergence and PT diagnostics with:

```text
python3 experiments/plot_pt_convergence_gap.py \
  /tmp/pt_tfim_n10_validation
```

The current local artifacts are:

```text
/private/tmp/pt_tfim_n10_validation/comparison_report.json
/private/tmp/pt_tfim_n10_validation/convergence.png
/private/tmp/pt_tfim_n10_validation/wall_time.png
/private/tmp/pt_tfim_n10_validation/pt_diagnostics.png
/private/tmp/pt_tfim_n10_validation/all_beta_validation.csv
```

These paths are temporary run artifacts and are not part of the Git commit.

## Current state and next steps

The `replicate-exchange` branch now has a tested beta-only PT implementation,
an exact small-system correctness gate, signed convergence traces, mixing
diagnostics, checkpoint/resume support, and systematic benchmark drivers.

Recommended next steps are:

1. Run the exact correctness gate across several ten- or twelve-spin TFIM and
   MAX2SAT seeds.
2. Use more than one independent ladder when estimating PT uncertainty.
3. Tune beta points to raise the lowest adjacent acceptance without adding
   unnecessary near-unity edges.
4. Compare median time to sustained accuracy over multiple graph and RNG
   seeds rather than interpreting one convergence curve as a general result.
5. Keep equal-rank fixed-beta controls so any claimed speedup includes the
   opportunity cost of non-cold PT slots.
