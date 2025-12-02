-----------------------------------------------------------------------------------------------------------

This repository contains `C++` simulation code, data, and `Python` plotting scripts and utilities in support of the works:
- <a href="https://doi.org/10.1038/s41524-025-01891-0">[1] Nic Ezzell, Lev Barash, Itay Hen, A universal black-box quantum Monte Carlo approach to quantum phase transitions, npj Computational Materials (2025).</a> (see also <a href="https://arxiv.org/abs/2408.03924"> the arxiv</a>)
- <a href="https://arxiv.org/abs/2504.07295">[2] Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, arXiv:2504.07295 (2025).</a>

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
The files H.txt, A.txt, and B.txt are rotations of the H_unrotated.txt, A_unrotated.txt, and B_unrotated.txt under U.txt. That is, A = U A_unrotated U^{\dagger}. This exact set of H, A, and B are used in the study of both <a href="https://doi.org/10.1038/s41524-025-01891-0">[1]<\a> and <a href="https://arxiv.org/abs/2504.07295">[2] </a>, specifically, they correspond to a random rotation of the 2q PRL model. This example highlights the ease with which one can study highly non-trivial models with our code. That this model has no severe sign problem is the result of post-selection on the empirical average sign as shown in `data_plotting_misc/plot_scripts/blackbox_2q_rotated_pre_selection.ipynb` and in the supplementary section S4 and Figure S5 in <a href="https://doi.org/10.1038/s41524-025-01891-0">[1]<\a>. 

## The 2q model, TFIM, and XXZ driver scripts
In `prl_2q_model.py`, `experiments/tfim_driver.py` and `experiments/xxz_driver.py`, we have written `Python` scripts which simplify preparation, compilation, and execution of our simulation for the 2q PRL model, the TFIM model and the XXZ model. Executing these codes is as simple as (1) using a version of `Python` that has `numpy`, (2) adjusting simultion parameters by editing the driver file (i.e., see the bottom of `experiments/tfim_driver.py`), and (3) running `python tfim_driver.py` or executing the `Python` script by any other desired means.

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

However, there is one subtlety. Underlying our QMC estimates is a binning analysis (see <a href="https://arxiv.org/abs/2307.06503">Appendix B of arXiv:2307.06503</a>) with a parameter `Nbins` (this is also set in `parameters.hpp` and for the drivers in `utils/ioscripts.py`). For valid statistical analysis, is is required that steps/stepsPerMeasurement/Nbins > Nbins, and it is better if this value is an integer. Larger values of Nbins improves the accuracy of error bar estimation. By default, we set `Nbins=100` in the `utils` for speed, which is generally sufficient, but a safer choice is `Nbins=200` or more.

## On our figures and data in `data_plotting_misc`
- The figures presented in <a href="https://arxiv.org/abs/2408.03924">[1]</a>,<a href="https://arxiv.org/abs/2504.07295">[2]</a> are present in `data_plotting_misc/plot_scripts/figures`.
- The scripts to generate these figures are jupyter notebooks contained in `data_plotting_misc/plot_scripts`, with decriptive names (also see the README.md in respective subdirectory).
- The data for these scripts comes from `data_plotting_misc/data`
- The subdirectory `data_plotting_misc/hamiltonians_and_observables` contains the random obserables used to verify and test our code in <a href="https://arxiv.org/abs/2504.07295">[2]</a>
- The subdirectory `data_plotting_misc/wrangling_scripts` contains scripts used convert raw QMC outputs as text files to the csv files used for plotting. These are not meant to be used literally, since the details are specific to how we ran and processed data on USC HPCs, but rather, as an example of how one might read the output of individual QMC simulations in a larger pipeline.

## On rotating Hamiltonians by random unitaries
As part of our work in <a href="https://arxiv.org/abs/2408.03924">[1]</a>, we rotated a 2 qubit Hamiltonian with a 100 qubit random unitary. This is an interesting construction in its own right, and the code to perform this task is purely written in `Python`, decoupled from the `C++` QMC simulation code. The necessary infrastructure is mostly contained in `utils/pauli_manipulations.py`.

As for using it in practice, each of the drivers in `experiments/` as well as `legacy_code/correlator_experiments` and `legacy_code/fidsus_experiments` support random rotations of the target Hamiltonian (i.e., 2 qubit PRL model, the TFIM, and the XXZ).

Finally, the very in the weeds but important detail in <a href="https://arxiv.org/abs/2408.03924">[1]</a> that our random ensemble of Hamiltonians does not have a sign problem comes from the empirical post-selection done in `data_plotting_misc/plot_scripts/before_fig2_post_selection.ipynb`.

