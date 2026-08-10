-----------------------------------------------------------------------------------------------------------

This repository contains `C++` simulation code, data, and `Python` plotting scripts and utilities in support of the works:
- <a href="https://doi.org/10.1038/s41524-025-01891-0">[1] Nic Ezzell, Lev Barash, Itay Hen, A universal black-box quantum Monte Carlo approach to quantum phase transitions, npj Computational Materials (2025).</a> (see also <a href="https://arxiv.org/abs/2408.03924"> the arXiv</a>)
- <a href="https://doi.org/10.1016/j.cpc.2026.110019">[2] Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, Computer Physics Communications (2026).</a> (see also <a href="https://arxiv.org/abs/2504.07295"> the arXiv</a>)

# Paper Errata and Improvements
In an ongoing follow up work, a diligent and careful graduate student, Josh Kao, found some typos and suggested improvements to divided difference relations and their proofs from the above papers. At this point, the corrections are minor, so we simply collect them on this GitHub page. Henceforth let <a href="https://arxiv.org/abs/2408.03924">[a]</a> refer to "A universal..." and <a href="https://arxiv.org/abs/2504.07295">[b]</a> to "Advanced measurement...". 
1. There is a typo in our proof of the Lemma 3 in <a href="https://arxiv.org/abs/2408.03924">[a]</a> wherein equations (E26)--(E28) should have an outer factor of $e^{-(\beta/2)[x_0, \ldots, x_r]}$ and not the present one with $-(\beta / 2 - \tau)$. This same typo unfortunately got propogated also to Appendix B.2 <a href="https://arxiv.org/abs/2504.07295">[b]</a> in equations (B15)--(B17).
2. The estimator in Theorem 2/ Corollary 3 in <a href="https://arxiv.org/abs/2408.03924">[a]</a> can be simplified (but not in total complexity). Specifically, the first two terms in the $r$ sum can be thought of as the result of applying the Leibniz rule to $f(x) = e^{-(\beta / 2) x}$ and $g(x) = x e^{-(\beta / 2) x}$. Since this simplifies to the function $h(x) = x e^{-\beta x}$, we could then just compute $h[x_0, \ldots, x_q]$ which, by the Leibniz rule between $x$ and $e^{-\beta x}$ (as in the estimator for $\langle H \rangle$ which is Eq.57 in <a href="https://arxiv.org/abs/2504.07295">[b]</a>), becomes just two terms: $x_0 e^{-\beta [x_0, \ldots, x_q]} + e^{-\beta [x_1, \ldots, x_q]}$. This is simplifies the result visually and in coding it, but this does not help simplify the weird $(i - r)$ term, so the complexity is the same. 

-----------------------------------------------------------------------------------------------------------
# Attribution
Most of the simulation code was (privately) forked from the `C++` library <a href="https://github.com/LevBarash/PMRQMC">PMRQMC</a>, as developed in: Lev Barash, Arman Babakhani, Itay Hen, A quantum Monte Carlo algorithm for arbitrary spin-1/2 Hamiltonians, Physical Review Research 6, 013281 (2024).

# Instructions
We give some instructions for setting up, compiling, and executing our `C++` code, with an emphasis on numerical experiments relevant to <a href="https://arxiv.org/abs/2408.03924">[1]</a>,<a href="https://arxiv.org/abs/2504.07295">[2]</a>. Additional details can be found in the aforementioned spin-1/2 repoistory that we forked from <a href="https://github.com/LevBarash/PMRQMC">PMRQMC</a> as well as <a href="https://arxiv.org/abs/2504.07295">Appendix A of [2] </a>. 

## The simple default experiment
The simplest way to test out our code is to simply execute the `test_run.sh` script, i.e., run `./test_run.sh` on shell after possibly giving permission with `chmod +x test_run.sh` or a similar command. This generates a file called `single_thread_output.txt` which summarizes the various observables measured, but note that for speed, our default simulation parameters do not lead to thermalized results. User inputs can be changed by adjusting the following files
- `H.txt` -- contains the Hamiltonian to be simulated
- `A.txt` -- a custom observable to measure
- `B.txt` -- a different custom observable to measure
- `parameters.hpp` -- contains editable simulation parameters such as number of QMC updates and inverse teperature
- `test_run.sh` -- compiles and executes script as a single threaded application (default) or can be changed to perform a multithreaded MPI simulation over user-specififed number of cores. At 5+ cores, automatic thermalization testing is performed.

## Explanation of default H.txt, A.txt, and so on
The files H.txt, A.txt, and B.txt are rotations of the H_unrotated.txt, A_unrotated.txt, and B_unrotated.txt under U.txt. That is, A = U A_unrotated U^{\dagger}. This exact set of H, A, and B are used in the study of both <a href="https://arxiv.org/abs/2408.03924">[1]</a> and <a href="https://arxiv.org/abs/2504.07295">[2]</a>, specifically, they correspond to a random rotation of the 2q PRL model. This example highlights the ease with which one can study highly non-trivial models with our code. That this model has no severe sign problem is the result of post-selection on the empirical average sign as shown in `data_plotting_misc/plot_scripts/blackbox_2q_rotated_pre_selection.ipynb` (see [here](https://github.com/naezzell/advmeaPMRQMC/blob/main/data_plotting_misc/plot_scripts/blackbox_2q_rotated_pre_selection.ipynb) and in the supplementary section S4 and Figure S5 in <a href="https://doi.org/10.1038/s41524-025-01891-0">[1]</a>). 

## The 2q model, TFIM, and XXZ driver scripts
In `experiments/prl_2q_model.py`, `experiments/tfim_driver.py` and `experiments/xxz_driver.py`, we have written `Python` scripts which simplify preparation, compilation, and execution of our simulation for the 2q PRL model, the TFIM model and the XXZ model. Executing these codes is as simple as (1) using a version of `Python` that has `numpy`, (2) adjusting simultion parameters by editing the driver file (i.e., see the bottom of `experiments/tfim_driver.py`), and (3) running `python tfim_driver.py` or executing the `Python` script by any other desired means.

These scripts, with appropriate modifitions for our local HPC cluster, were used to perform the simulations in our manuscript.

## Quick tour of directories

For ease of understanding, we briefly explain each directory in our codebase.
- `./` -- `C++` simulation files (modified from <a href="https://github.com/LevBarash/PMRQMC">PMRQMC</a>)
- `experiments` -- `Python` scripts to set up, compile, and run TFIM and XXZ experiments and rotations thereof (new; can be used to generate data from <a href="https://arxiv.org/abs/2408.03924">[1]</a>,<a href="https://arxiv.org/abs/2504.07295">[2]</a>)
- `data_plotting_misc` -- data and plot scripts to reproduce all figures from <a href="https://arxiv.org/abs/2408.03924">[1]</a>,<a href="https://arxiv.org/abs/2504.07295">[2]</a>
- `legacy_code` -- an earlier version of our code alongside `Python` scripts to run the 2 qubit PRL model (new)
- `utils` -- i/o, exact calculations, Hamiltonian building and rotations code

# Misc notes of importance

## Valid values for Tsteps, steps, and stepsPerMeasurement

The meaning behind these parameters can be adjusted either in `parameters.hpp` or in a respective driver file such as `experiments/tfim_driver.py`. For the most part, they are straightforward integer values...
- Tsteps: number of equilibration steps before observables sampled
- steps: total number of QMC updates after Tsteps
- stepsPerMeasurement: a sample measurement is taken every stepsPerMeasurement

### Opt-in beta-annealed initialization

`PMRQMC.bin`, `PMRQMC_mpi.bin`, and the PT/QCPT executable accept an optional
beta ramp before production:

```text
--beta-anneal
--anneal-start-factor 0.001
--anneal-interval K
```

With `--beta-anneal`, `Tsteps` is the number of annealing updates, not a
separate static-beta equilibration. The default ramp is linear from
`0.001 * target_beta` to the target, and beta is retargeted every `K` calls to
`update()`. The default `K` is `N`, so one "sweep" in the paper-inspired
protocol corresponds to one nominal sweep of `N` update calls here. Tau is
scaled throughout the ramp to preserve the target `tau/beta` ratio. The exact
target is forced after update `Tsteps`, and the next update begins production.

For a fixed run, runtime targets avoid recompiling for every endpoint:

```text
./PMRQMC.bin --target-beta 3.0 --target-tau 1.5 \
  --beta-anneal --anneal-interval 100
```

If `--target-tau` is omitted, it defaults to half the target beta. In PT and
QCPT, every slot follows its own beta ramp. Exchanges, flow accounting, and
exchange RNG draws are disabled during the ramp. Production-relative exchange
parity starts at the endpoint; QCPT keeps each slot's target gamma fixed for
the entire beta ramp.

An absolute schedule can replace the factor-based ramp:

```text
# completed_updates beta_slot_0 beta_slot_1 ...
0       0.001 0.002 0.004
50000   0.05  0.10  0.20
100000  0.10  0.20  0.40
```

Pass it with `--beta-anneal-schedule FILE`. Rows are cumulative update
coordinates; the first must be zero, the last must equal `Tsteps`, and the
last beta in each column must equal that slot's target. Columns must be
positive, finite, and nondecreasing. The piecewise-linear curve is evaluated
only at `--anneal-interval` boundaries. Factor and file modes cannot be used
together.

To run independent fixed simulations for several endpoints, compile once and
launch a fresh process per target with:

```text
python3 experiments/beta_anneal_driver.py RUN_DIRECTORY \
  --betas 0.5,1,2,4 --Tsteps 100000 --steps 1000000 \
  --anneal-interval 100 --mpi-ranks 4
```

Each target receives a separate output directory. `manifest.json` records the
target, tau, observed RNG seeds, ramp settings, schedule and plan hashes,
command, and wall time. Repeat `--absolute-schedule BETA:FILE` for custom
per-target schedules. Checkpoint identity includes the annealing plan, so an
interrupted ramp can resume only with matching settings. Omitting all
annealing and runtime-target options preserves the legacy fixed/PT behavior
and checkpoint layout.

However, there is one subtlety. Underlying our QMC estimates is a binning analysis (see <a href="https://arxiv.org/abs/2307.06503">Appendix B of arXiv:2307.06503</a>) with a parameter `Nbins` (this is also set in `parameters.hpp` and for the drivers in `utils/ioscripts.py`). For valid statistical analysis, is is required that steps/stepsPerMeasurement/Nbins > Nbins, and it is better if this value is an integer. Larger values of Nbins improves the accuracy of error bar estimation. By default, we set `Nbins=100` in the `utils` for speed, which is generally sufficient, but a safer choice is `Nbins=200` or more.

## On our figures and data in `data_plotting_misc`
- Figures generated in support of <a href="https://arxiv.org/abs/2408.03924">[1]</a> are contained in [data_plotting_misc/plot_scripts/blackbox_paper_figures](https://github.com/naezzell/advmeaPMRQMC/tree/main/data_plotting_misc/plot_scripts/blackbox_paper_figures) and those for <a href="https://arxiv.org/abs/2504.07295">[2]</a> are contained in [data_plotting_misc/plot_scripts/advmeas_figures](https://github.com/naezzell/advmeaPMRQMC/tree/main/data_plotting_misc/plot_scripts/advmeas_figures)
- The scripts to generate these figures are jupyter notebooks contained in `data_plotting_misc/plot_scripts`, with decriptive names (also see the README.md in respective subdirectory).
- The data for these scripts comes from `data_plotting_misc/data`
- The subdirectory `data_plotting_misc/hamiltonians_and_observables` contains the random obserables used to verify and test our code in <a href="https://arxiv.org/abs/2504.07295">[2]</a>
- The subdirectory `data_plotting_misc/wrangling_scripts` contains scripts used convert raw QMC outputs as text files to the csv files used for plotting. These are not meant to be used literally, since the details are specific to how we ran and processed data on USC HPCs, but rather, as an example of how one might read the output of individual QMC simulations in a larger pipeline.

## On rotating Hamiltonians by random unitaries
As part of our work in <a href="https://arxiv.org/abs/2408.03924">[1]</a> and <a href="https://arxiv.org/abs/2504.07295">[2]</a>, we rotated a 2 qubit Hamiltonian with a 100 qubit random unitary. This is an interesting construction in its own right, and the code to perform this task is purely written in `Python`, decoupled from the `C++` QMC simulation code. The necessary infrastructure is mostly contained in `utils/pauli_manipulations.py`.

As for using it in practice, each of the drivers in `experiments/` as well as `legacy_code/correlator_experiments` and `legacy_code/fidsus_experiments` support random rotations of the target Hamiltonian (i.e., 2 qubit PRL model, the TFIM, and the XXZ).

Finally, the very in the weeds lack of sign problem detail is the result of post-selection on the empirical average sign as shown in `data_plotting_misc/plot_scripts/blackbox_2q_rotated_pre_selection.ipynb` (see [here](https://github.com/naezzell/advmeaPMRQMC/blob/main/data_plotting_misc/plot_scripts/blackbox_2q_rotated_pre_selection.ipynb) and in the supplementary section S4 and Figure S5 in <a href="https://doi.org/10.1038/s41524-025-01891-0">[1]</a>). 

## Beta-only parallel tempering

For a quick end-to-end correctness check, run the two-spin validation rather
than the full benchmark sweep:

```text
make validate-pt
```

This constructs a four-temperature PT ladder, a separate fixed-beta run at
the cold endpoint, and a direct 4x4 diagonalization using only the Python
standard library. It checks the transverse magnetization at every PT beta,
requires at least one complete round trip, writes `validation_report.json` in
the printed temporary directory, and exits nonzero if a result is outside the
larger of six reported standard errors and an absolute tolerance of 0.025.
It also writes `convergence.csv`, `convergence_updates.svg`, and
`convergence_time.svg`; the plots compare the sign-reweighted running means of
the fixed-beta and PT cold-slot estimators with the exact result. PNG copies
are generated automatically when `rsvg-convert` is available.

The beta-only PMR replica-exchange executable keeps one rank at each fixed
temperature and exchanges complete PMR configurations at synchronized update
boundaries. Compile it after `prepare.cpp` has generated `hamiltonian.hpp`:

```text
mpicxx -O3 -std=c++11 -o PMRQMC_pt_mpi.bin PMRQMC_pt_mpi.cpp
mpirun -n <temperatures * independent_ladders> ./PMRQMC_pt_mpi.bin \
  --schedule tempering_schedule.txt --updates-per-exchange 10 \
  --output-prefix run_name
```

Each schedule row is `beta tau`; a missing `tau` defaults to `beta/2`. Beta
values must be positive and strictly increasing, and every tau must be in
`[0,beta]`. MPI size must be an integer multiple of the number of rows;
the quotient is the number of independent ladders. The optional
`--independent-ladders` argument checks that layout explicitly.

The run writes `run_name_observables.csv`, `run_name_swaps.csv`, and
`run_name_flow.csv`. Observables, signs, and derived quantities are reduced
among ranks occupying the same temperature. Swap files report edge attempts
and accepted swaps; flow files report trajectory occupancy. `--checkpoint-every`
creates atomic, versioned per-rank checkpoints, and `--resume` requires a
complete set whose schedule hash and rank coordinates match the current run.

`qmax` remains a compile-time capacity in `parameters.hpp`; increase it when a
rank reaches the warning threshold. The Python helper
`experiments/pt_driver.py` writes schedules, sets the MPI size, and exposes
`updates_per_exchange`, independent ladders, `qmax`, and output prefixes.

For a larger convergence-time comparison, use
`experiments/pt_convergence_gap.py`. It generates a deterministic random
3-regular transverse-field instance, runs a fixed-beta MPI baseline and a
beta-tempered run with the same MPI size, and records the cold-slot observable
trace in addition to the usual PT swap and flow files:

```text
python3 experiments/pt_convergence_gap.py /tmp/pt_gap --n 20 --ladders 2 \
  --Tsteps 100000 --steps 1000000 --steps-per-measurement 100
```

On a workstation with fewer advertised MPI slots than the requested ranks,
append `--oversubscribe`; omit it inside a scheduler allocation.

The default ladder has 11 temperatures, so this command uses 22 MPI ranks for
each run. The baseline places all ranks at the cold beta; PT uses the same 22
ranks as two independent 11-temperature ladders. The script writes
`comparison_report.json`, `fixed_beta/trace.csv`, and
`beta_tempered/trace.csv`. Its convergence time is the first sustained entry
of the cumulative observable mean into a tolerance band around the final
cold-slot PT estimate. Repeat several seeds and inspect the traces; the
reported ratio is an instance-dependent diagnostic, not a universal speedup.

The lower-level MPI binaries also accept `--timeseries-prefix FILE`. The fixed
MPI trace is averaged over all fixed-beta ranks; the PT trace contains one row
per temperature and measurement, averaged over independent ladders. The
observable used by the example is `obs_0`, the first observable passed to
`prepare.bin`. Each trace stores both the raw estimator and its signed
numerator. Convergence plots use the cumulative ratio
`sum(signed_obs_0)/sum(sign)`; older traces without `signed_obs_0` must be
regenerated and are deliberately rejected by the plotting script.

Plot the resulting convergence and PT diagnostics from the command line with:

```text
python3 experiments/plot_pt_convergence_gap.py /tmp/pt_gap
```

This writes `convergence.png`, `wall_time.png`, and `pt_diagnostics.png`.
`wall_time.png` is the practically relevant comparison: it shows both the
running estimate and its error against elapsed wall-clock time. Install
`matplotlib` if it is not already available; use `--show` for interactive
display or `--output-dir DIR` to place the images elsewhere.

For an exact cold-beta reference on small systems (`n <= 12`), add
`--exact-reference` to the benchmark command, which uses dense exact
diagonalization and writes `exact_reference.json`:

```text
python3 experiments/pt_convergence_gap.py /tmp/pt_gap_exact \
  --model max2sat --n 10 \
  --betas 0.1,0.3,0.7,1.5,3.0,5.0 --ladders 1 \
  --Tsteps 100000 --steps 1000000 --steps-per-measurement 100 \
  --exact-reference
```

Alternatively, compute the exact value after an existing benchmark with
`python3 experiments/exact_diagonalization.py /tmp/pt_gap`, then plot it with
`python3 experiments/plot_pt_convergence_gap.py /tmp/pt_gap --reference exact`.

This implementation is PMR replica exchange over beta only. It is not
isoenergetic cluster Monte Carlo (ICM), and it does not implement the separate
beta-Gamma tempering algorithm. Exact reproduction of historical plots also
requires the original unpublished instances, seeds, swap cadence, and
hardware; the included deterministic benchmark preset records the instances
it generates rather than claiming bit-for-bit reproduction.

### Systematic correctness and ladder loop

Use `experiments/pt_benchmark_loop.py` to sweep independent instance seeds,
beta schedules, exchange cadences, and numbers of independent ladders. Every
grid point uses the same MPI size for PT and its fixed-cold-beta control, and
the loop writes `runs.csv`, `summary.csv`, and the complete per-run artifacts:

```text
/path/to/python-with-numpy experiments/pt_benchmark_loop.py /tmp/pt_loop \
  --model max2sat --n 10 --seeds 11,12,13 \
  --rng-seeds 1000,2000 \
  --schedule coarse=0.1,0.3,0.7,1.5,3,5 \
  --schedule geometric=0.1,0.219,0.478,1.046,2.287,5 \
  --exchange-cadences 5,10,50 --independent-ladders 1,2 \
  --exact-reference --require-correctness --oversubscribe
```

Treat the loop as two stages. First, run `n <= 12` with exact references and
require both fixed beta and every PT candidate to pass the accuracy gate. A
failure here is an implementation or estimator problem, not evidence for a
poor ladder. Second, run larger hard instances over many seeds and rank only
the candidates that passed stage one. Compare median time to accuracy and
effective mixing per wall time; also inspect the worst adjacent acceptance,
temperature flow, complete round trips, average sign, and `qmax` warnings.

The fixed-beta control intentionally spends every rank at the target beta,
whereas PT spends only one rank per ladder there. This equal-resource control
is the relevant test of a PT speedup. A useful ladder has neither a near-zero
acceptance bottleneck nor uniformly near-one acceptance with needless
temperatures. Tune `beta_min` until the hot replica mixes rapidly, then move
intermediate betas toward bottleneck edges and repeat the seed ensemble.

### Tiny 3-regular MAX2SAT demonstration

`experiments/small_max2sat_pt_demo.py` generates a six-variable, nine-clause
3-regular MAX2SAT Hamiltonian with a small transverse field, then compares a
fixed-beta run against a five-temperature beta-only PT run in an isolated
directory:

```text
python3 experiments/small_max2sat_pt_demo.py /tmp/max2sat_pt_demo \
  --Tsteps 200 --steps 2000
```

The script writes the clauses to `instance.json` and summarizes fixed-beta
statistics, cold-slot observables, swap acceptance, and complete PT round
trips in `comparison_report.json`. The short example demonstrates temperature
mobility; longer runs are needed for a quantitative statistical-efficiency
claim.

### Beta-gamma quantum-classical parallel tempering

Prepare a split Hamiltonian `H_fixed + gamma H_gamma`; equivalent Pauli terms
remain component-wise, including terms that cancel at a particular Gamma:

```text
./prepare.bin --hamiltonian-fixed H_fixed.txt --hamiltonian-gamma H_gamma.txt [observable ...]
make PMRQMC_qcpt_mpi.bin
mpirun -n <path-points * independent-ladders> ./PMRQMC_qcpt_mpi.bin \
  --schedule qcpt_schedule.txt --updates-per-exchange 10 \
  --independent-ladders 2 --output-prefix run_name
```

QCPT schedule rows are `beta gamma [tau]`; omitted `tau` defaults to `beta/2`.
Beta and Gamma may vary in either direction. Crossed weights rebuild every
diagonal energy and off-diagonal matrix element from the snapshot at the
target `(beta,gamma)`. QCPT outputs identify `slot,beta,gamma,tau` and use
the distinct `.qcptckpt` checkpoint format.
