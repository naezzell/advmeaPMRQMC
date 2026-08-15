##
# This code was written in support of the experiments carried out in:
# * Nic Ezzell, Lev Barash, Itay Hen, Exact and universal quantum Monte Carlo estimators for energy susceptibility and fidelity susceptibility, arXiv:2408.03924 (2024).
# * Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, arXiv:2504.07295 (2025).

# Description: Code to build and execute main QMC simulation
##

#!/bin/bash

# code to set up simulation
g++ -O3 -std=c++23 -o prepare.bin prepare.cpp
#./prepare.bin H.txt A.txt A.txt A.txt A.txt A.txt B.txt B.txt B.txt B.txt B.txt A.txt A.txt A.txt A.txt A.txt
./prepare.bin H.txt

# compiles and executes simulation on a single thread
#g++ -O3 -std=c++11 -o PMRQMC.bin PMRQMC.cpp
#./PMRQMC.bin > single_thread_output.txt

#
# Below, we include ways to compile with MPI to perform lazy
# parallelization. When relevant, applies automatic jackknife
# bootstrapping, i.e., to compute specific heat. Also, at 5+
# cores, our code automatically performs equilibration testing.
#

# compile and run MPI locally (tested on MAC OSX)
# mpicxx -O3 -std=c++11 -o PMRQMC_mpi.bin PMRQMC_mpi.cpp
mpicxx -O3 -std=c++23 -o PMRQMC_mpi.bin PMRQMC_mpi.cpp
mpirun -n 6 ./PMRQMC_mpi.bin > therm_test_output.txt

# compile and run MPI on an HPC that uses the slurm scheduler
#mpicxx -O3 -o PMRQMC_mpi.bin PMRQMC_mpi.cpp
#srun --mpi=pmix_v2 -o PMRQMC_mpi-%j.out -e slurm-%j.out -n $SLURM_NTASKS ./PMRQMC_mpi.bin
