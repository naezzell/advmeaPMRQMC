# QCPT Validation Report

Status: complete. The validated implementation is in commit `17b883c`.

## Scope

This follow-on extends the validated beta-only replica exchange implementation to ordered paths in
`(beta, gamma)` for

`H(gamma) = H_fixed + gamma H_gamma`.

The implementation preserves the legacy single-Hamiltonian interface and adds:

- split Hamiltonian preparation with component-wise coefficient merging;
- non-mutating target-parameter PMR weights;
- QCPT path schedules with optional `tau`;
- live-snapshot exchanges, independent ladders, signed traces, flow diagnostics, and a distinct atomic checkpoint format;
- `ABS_WEIGHTS` coverage;
- reproducible exact two-spin, MAX2SAT, checkpoint, and pilot drivers.

## Reproduction environment

- Base commit: `f2f8e980962c2ac6098366fed5b1a870c21dd8d9`
- Validated implementation commit: `17b883c`
- Compiler: Apple clang 16.0.0 (`clang-1600.0.26.6`)
- Python: 3.13.3
- MPI: Open MPI 5.0.7
- Platform: macOS, Accelerate used by the exact real-matrix helper

The raw outputs below are intentionally outside the repository:

- `/tmp/qcpt_exact_signed_fix`
- `/tmp/qcpt_exact_abs`
- `/tmp/qcpt_checkpoint_pass`
- `/tmp/qcpt_checkpoint_abs`
- `/tmp/qcpt_max2sat_signed_final`
- `/tmp/qcpt_pilot_smoke2`
- `/var/folders/2w/_70fq7w11c36fkhz8pnqp8gw0000gp/T/pmrqmc_pt_validation_9_tpn562`

## Automated gates

All gates passed.

```text
make test                         10 tests: OK
make all                          OK
make validate-pt                  OK
validate_qcpt.py                  signed: OK, ABS_WEIGHTS: OK
validate_checkpoint_resume.py    signed: QCPT + beta: OK
validate_checkpoint_resume.py    ABS_WEIGHTS QCPT + beta: OK
validate_qcpt_max2sat.py          4/4 campaigns: OK
qcpt_pilot.py                     OK
```

Commands used for the principal campaigns:

```bash
make test
./prepare.bin H.txt A.txt B.txt
make all
make validate-pt

python3 experiments/validate_qcpt.py /tmp/qcpt_exact_signed_fix \
  --Tsteps 5000 --steps 100000 --nbins 50 \
  --updates-per-exchange 10 --oversubscribe

python3 experiments/validate_qcpt.py /tmp/qcpt_exact_abs \
  --Tsteps 5000 --steps 100000 --nbins 50 \
  --updates-per-exchange 10 --oversubscribe --absolute-weights

python3 experiments/validate_checkpoint_resume.py /tmp/qcpt_checkpoint_pass \
  --Tsteps 5000 --steps 20000 --nbins 20 --checkpoint-every 5000 \
  --updates-per-exchange 10 --oversubscribe

python3 experiments/validate_checkpoint_resume.py /tmp/qcpt_checkpoint_abs \
  --Tsteps 5000 --steps 20000 --nbins 20 --checkpoint-every 5000 \
  --updates-per-exchange 10 --oversubscribe --absolute-weights

python3 experiments/validate_qcpt_max2sat.py /tmp/qcpt_max2sat_signed_final \
  --Tsteps 2000 --steps 200000 --nbins 40 \
  --instance-seeds 11,12 --qmc-seeds 1000,2000 --oversubscribe

python3 experiments/qcpt_pilot.py /tmp/qcpt_pilot_smoke2 \
  --Tsteps 1000 --steps 10000 --nbins 10 --oversubscribe
```

## Fully mixed two-spin exact validation

The fixture is the required model in `tests/qcpt_fixed.txt` and `tests/qcpt_gamma.txt`. Both components contain diagonal and off-diagonal terms; `ZZ` and `XX` occur in both components. The path is:

| beta | gamma |
|---:|---:|
| 0.25 | 1.50 |
| 0.70 | 0.75 |
| 1.60 | 0.25 |
| 3.50 | 0.00 |

The table reports exact diagonalization, fixed-parameter QMC, and the QCPT marginal. Each row passed the larger of `6 * reported SE` and `0.025`; the JSON artifact contains the individual standard errors and tolerances.

| point | observable | exact | fixed QMC | QCPT |
|---|---|---:|---:|---:|
| (0.25, 1.50) | E | -0.614905 | -0.623731 | -0.613413 |
| (0.25, 1.50) | Z1 | -0.074613 | -0.075600 | -0.074200 |
| (0.25, 1.50) | Xavg | 0.155046 | 0.166564 | 0.151486 |
| (0.70, 0.75) | E | -0.936552 | -0.947461 | -0.956442 |
| (0.70, 0.75) | Z1 | -0.207655 | -0.242000 | -0.208000 |
| (0.70, 0.75) | Xavg | 0.259139 | 0.260601 | 0.284300 |
| (1.60, 0.25) | E | -1.086588 | -1.082108 | -1.085345 |
| (1.60, 0.25) | Z1 | -0.359578 | -0.358000 | -0.351400 |
| (1.60, 0.25) | Xavg | 0.270471 | 0.241860 | 0.284338 |
| (3.50, 0.00) | E | -1.098773 | -1.098049 | -1.096892 |
| (3.50, 0.00) | Z1 | -0.539753 | -0.530400 | -0.534600 |
| (3.50, 0.00) | Xavg | 0.220809 | 0.240834 | 0.221031 |

The explicitly combined Hamiltonian agrees with the split representation under identical fixed seeds. The signed and `ABS_WEIGHTS` campaigns both passed; their deterministic reported values are identical for this fixture.

Target-weight sensitivity controls also passed:

| deliberate incomplete evaluator | disagreement |
|---|---:|
| freeze diagonal energy at local gamma, q=0 | 0.0811911 |
| reuse local off-diagonal product, q=2 | 0.00899646 |

Thus both principal target-weight mistakes are detected by the test-only controls.

## Exact n=10 3-regular MAX2SAT campaign

Four campaigns passed: instance seeds `11, 12`, QMC seeds `1000, 2000`, with 200,000 sampling updates. The exact reference was computed at every path point for energy and transverse magnetization.

| instance | QMC seed | point | exact E | QCPT E | exact M | QCPT M |
|---:|---:|---|---:|---:|---:|---:|
| 11 | 1000 | (0.25,1.50) | 2.901316 | 2.952595 | 0.092911 | 0.089268 |
| 11 | 1000 | (0.70,0.75) | 2.231968 | 2.260355 | 0.124918 | 0.124984 |
| 11 | 1000 | (1.60,0.25) | 1.356722 | 1.358465 | 0.082789 | 0.087736 |
| 11 | 1000 | (3.50,0.00) | 0.392594 | 0.351850 | 0.000000 | 0.000000 |
| 11 | 2000 | (0.25,1.50) | 2.901316 | 2.864577 | 0.092911 | 0.098193 |
| 11 | 2000 | (0.70,0.75) | 2.231968 | 2.219578 | 0.124918 | 0.133505 |
| 11 | 2000 | (1.60,0.25) | 1.356722 | 1.359061 | 0.082789 | 0.081103 |
| 11 | 2000 | (3.50,0.00) | 0.392594 | 0.390450 | 0.000000 | 0.000000 |
| 12 | 1000 | (0.25,1.50) | 2.753391 | 2.788120 | 0.092776 | 0.087528 |
| 12 | 1000 | (0.70,0.75) | 1.891122 | 1.859559 | 0.123809 | 0.125355 |
| 12 | 1000 | (1.60,0.25) | 0.869782 | 0.901992 | 0.081047 | 0.079932 |
| 12 | 1000 | (3.50,0.00) | 0.134962 | 0.131000 | 0.000000 | 0.000000 |
| 12 | 2000 | (0.25,1.50) | 2.753391 | 2.730588 | 0.092776 | 0.093577 |
| 12 | 2000 | (0.70,0.75) | 1.891122 | 1.903931 | 0.123809 | 0.125157 |
| 12 | 2000 | (1.60,0.25) | 0.869782 | 0.835518 | 0.081047 | 0.082851 |
| 12 | 2000 | (3.50,0.00) | 0.134962 | 0.137550 | 0.000000 | 0.000000 |

All four campaigns attempted every swap edge, completed round trips, produced finite sign-reweighted estimates, and did not hit `qmax`. Representative edge acceptance was 0.629–0.675, 0.466–0.504, and 0.342–0.494 across the four runs; round trips were 1,348–1,605 per trajectory.

## Legacy beta-only and checkpoint evidence

The legacy exact two-spin beta-only regression passed at all fixed/PT beta values:

| method | beta | exact | estimate | SE | acceptance/flow |
|---|---:|---:|---:|---:|---|
| fixed | 3.00 | 0.729466 | 0.741705 | 0.013361 | — |
| PT | 0.25 | 0.169804 | 0.170456 | 0.010133 | — |
| PT | 0.60 | 0.360402 | 0.363520 | 0.008105 | — |
| PT | 1.30 | 0.561578 | 0.581131 | 0.008190 | — |
| PT | 3.00 | 0.729466 | 0.726283 | 0.009081 | — |

The beta-only swap acceptance was `0.763, 0.628, 0.453`, with 11,727 round trips in the final run.

Checkpoint/resume comparisons were byte-stable for observables, swaps, and flow in both QCPT and legacy beta-only modes. This passed in both signed and `ABS_WEIGHTS` builds:

| mode | weight mode | observables | swaps | flow |
|---|---|---|---|---|
| QCPT | signed | identical | identical | identical |
| beta-only | signed | identical | identical | identical |
| QCPT | ABS_WEIGHTS | identical | identical | identical |
| beta-only | ABS_WEIGHTS | identical | identical | identical |

## Equal-rank medium pilot

The pilot used one `n=20` instance, two independent ladders, and eight ranks in each control. It is an instrumentation result, not a universal performance claim.

| quantity | beta-only PT | mixed QCPT |
|---|---:|---:|
| wall seconds | 0.5722 | 0.5533 |
| average sign | 1.0 | 1.0 |
| effective samples | 183.39 | 151.74 |
| effective samples/core-hour | 144,220 | 123,403 |
| integrated autocorrelation (measurements) | 10.91 | 13.18 |
| adjacent acceptance | 0.480, 0.257, 0.092 | 0.485, 0.334, 0.271 |
| endpoint visits | 4,400 | 4,400 |
| round trips | 38 | 62 |
| mean q | 1.542 | 0.156 |
| max q | 14 | 4 |
| qmax hit | false | false |
| crossed-weight fraction | 0.352 | 0.129 |

The fixed control completed in 0.4814 seconds on eight ranks. The pilot demonstrates the requested instrumentation and healthy exchange/flow diagnostics; it does not establish a general speedup.

## Resolved failures and permanent regressions

1. The first legacy compile path used a generated `parameters.hpp` without `gamma`, and the fallback collided with the C math `gamma()` symbol. The fallback is now defined locally in `mainqmc.hpp` only when absent. The legacy exact/PT regression and `make all` cover this case.
2. The first fully mixed run disagreed at `gamma=0` for `Xavg`. The split observable generator had selected a gamma-only equivalent permutation relation, biasing the observable when the gamma component vanished. Relation selection now prefers fixed-supported representations; `test_split_observable_prefers_fixed_supported_relation` permanently covers it.
3. Resumed runs initially emitted zero-valued observables because checkpoint loading bypassed the normal initialization of the valid-observable mask. `configure_valid_observables()` is now called before both fresh initialization and checkpoint loading; the deterministic checkpoint harness covers signed and absolute modes.
4. The n=10 driver initially failed while serializing tuple-valued exact-point keys, and zero-variance summaries were not JSON-safe. The driver now serializes points as lists and normalizes zero standard deviations; the final four-campaign artifact passes.
5. A 50,000-update n=10 campaign under-resolved one seed. The final 200,000-update rerun passed all four seeds without changing tolerances, dropping seeds, or changing correctness logic.
6. A sandbox-only MPI launch failed before program startup because Open MPI could not bind local sockets. Re-running with the required external MPI permission passed; this was not a program or validation failure.

## Claims and limitations

The implementation now supports real, nonnegative gamma on ordered beta/gamma paths, arbitrary supported diagonal/off-diagonal terms in both split components, exact crossed target weights, signed and absolute weight modes, multiple ladders, and distinct QCPT checkpoint/resume.

The exact required fully mixed fixture, legacy beta-only regression, n=10 campaign, and equal-rank pilot all pass. The current report does not claim performance scaling beyond the recorded pilot, nor does it claim validation of unpublished `N=60` instances. Classical ICM and `q>0` isoenergetic moves remain outside this follow-on.
