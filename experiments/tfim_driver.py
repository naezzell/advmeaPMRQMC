"""
Suscepibility experiment driver
for transverse-field Ising model (TFIM),
H = -\sum_{<i,j>}Z_i Z_j - lam*\sum_{i}X_i (stand_tfim = True)
H = -\sum_{<i,j>}X_i X_j - lam*\sum_{i}Z_i (stand_tfim = False)
and random rotations thereof.

Supports 1D chain, 2D square lattice,
or 2D triangular lattice with or
without periodic boundary conditions (PBC)

This driver was written in support of the experiments carried out in:
// * Nic Ezzell, Lev Barash, Itay Hen, Exact and universal quantum Monte Carlo estimators for energy susceptibility and fidelity susceptibility, arXiv:2408.03924 (2024).
// * Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, arXiv:2504.07295 (2025).
"""
# %%
import subprocess, datetime, sys, os
import numpy as np
sys.path.append("..")
sys.path.append("../utils")
from pauli_manipulations import PauliTerm, PauliH, PauliU
from ioscripts import make_all_stand_param_fstr
# import for stand_tfim = True
from build_pauliH_recipes import (build_1d_tfim_zz, build_1d_tfim_x,
                                  build_square_tfim_zz, build_square_tfim_x,
                                  build_triangle_tfim_zz, build_triangle_tfim_x)
# import for stand_tfim = False
from build_pauliH_recipes import (build_1d_tfim_z, build_1d_tfim_xx,
                                  build_square_tfim_z, build_square_tfim_xx,
                                  build_triangle_tfim_z, build_triangle_tfim_xx)

def main(nt, L, lat, pbc, lam, beta, tau, parity, strnow, eps=None, l=None, Tsteps=1000000, steps=10000000, stepsPerMeasurement=10,
         save=True, restart=False, stand_tfim=True):
    # ==============================================
    # Choose random parametrs if not given 
    # ==============================================
    if lat == "chain":
        n = L
    else:
        n = L * L
    if l is None:
        l = np.random.choice(range(1, 2*n + 2))
        
    # make relevant directory name, copy over important files
    #dir_name = f"../{lat}_tfim/square_tfim_L_{L}_lam_{lam}"
    dir_name = ".."
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # ==============================================
    # Building H.txt
    # ==============================================
    if stand_tfim is True:
        if lat == "chain":
            h0 = build_1d_tfim_zz(L, pbc)
            h1 = build_1d_tfim_x(L)
        elif lat == "square":
            h0 = build_square_tfim_zz(L, pbc)
            h1 = build_square_tfim_x(L)
        elif lat == "triangle":
            h0 = build_triangle_tfim_zz(L, pbc)
            h1 = build_triangle_tfim_x(L)
    else:
        if lat == "chain":
            h0 = build_1d_tfim_xx(L, pbc)
            h1 = build_1d_tfim_z(L)
        elif lat == "square":
            h0 = build_square_tfim_xx(L, pbc)
            h1 = build_square_tfim_z(L)
        elif lat == "triangle":
            h0 = build_triangle_tfim_xx(L, pbc)
            h1 = build_triangle_tfim_z(L)

    # rotate the Hamiltonian
    u = PauliU(n)
    if l > 0:
        u.set_as_random(l, eps)
        uh0 = h0.conjugate(u)
        uh1 = h1.conjugate(u)
    else:
        uh0 = h0
        uh1 = h1
    uh = uh0 + lam * uh1
    # save Hamiltonian file
    with open(f"{dir_name}/H.txt", 'w') as f:
        f.write(uh.to_pmr_str())

    # ==============================================
    # Building observables A and B
    # ==============================================
    # A = X1
    px = PauliTerm(1.0, [1], ['X'], n)
    pzz = PauliTerm(1.0, [2,3], ['Z', 'Z'], n)
    a = PauliH(n, [px, pzz])
    # B = sum of 5 random Paulis if you uncomment this
    """
    pauli_terms = []
    for j in range(5):
        c = np.random.uniform(-1, 1)
        p = PauliTerm(c, [1], ['X'], n)
        k = np.random.choice(range(1, 3+1))
        p.make_random(k)
        pauli_terms.append(p)
    """
    # B as in paper, if you want random comment this out
    coeffs = [-0.773712, 0.155294, -0.966529]
    term1 = PauliTerm(coeffs[0], [3,9], ['X', 'X'], n)
    term2 = PauliTerm(coeffs[1], [3,6,9], ['Z', 'Z', 'Z'], n)
    term3 = PauliTerm(coeffs[2], [1,6,7], ['Y', 'X', 'Z'], n)
    pauli_terms = [term1, term2, term3]
    b = PauliH(n, pauli_terms)
    with open(f"{dir_name}/A_unrotated.txt", 'w') as f:
        f.write(a.to_pmr_str())
    with open(f"{dir_name}/B_unrotated.txt", 'w') as f:
        f.write(b.to_pmr_str())
    if l > 0:
        a = a.conjugate(u)
        b = b.conjugate(u)
    with open(f"{dir_name}/A.txt", 'w') as f:
        f.write(a.to_pmr_str())
    with open(f"{dir_name}/B.txt", 'w') as f:
        f.write(b.to_pmr_str())
    # ==============================================
    # Running PMR QMC
    # ==============================================
    param_str = make_all_stand_param_fstr(beta, tau, Tsteps, steps, stepsPerMeasurement, parity, save, restart)
    with open(f"{dir_name}/parameters.hpp", "w") as f:
        f.write(param_str)
    # compile and run PMR-QMC
    job_file = "job_file.sh"
    temp_out_file = f"{lat}_tfim_experiment_{strnow}.txt"
    with open(f"{dir_name}/{job_file}", 'w') as fh:
        fh.write("#!/bin/bash\n")
        # compile and run
        fh.write("g++ -O3 -std=c++11 -o prepare.bin prepare.cpp\n")
        fh.write("./prepare.bin H.txt A.txt A.txt A.txt A.txt A.txt B.txt B.txt B.txt B.txt B.txt A.txt A.txt A.txt A.txt A.txt\n")
        if nt > 1:
            fh.write("mpicxx -O3 -std=c++11 -o PMRQMC_mpi.bin PMRQMC_mpi.cpp\n")
            #fh.write(f"mpirun -n {nt} ./PMRQMC_mpi.bin > {temp_out_file}\n")
            fh.write(f"mpirun -n {nt} ./PMRQMC_mpi.bin")
        else:
            fh.write("g++ -O3 -std=c++11 -o PMRQMC.bin PMRQMC.cpp\n")
            #fh.write(f"./PMRQMC.bin > {temp_out_file}")
            fh.write(f"./PMRQMC.bin")
    subprocess.run(f"cd {dir_name}; chmod +x {job_file}; ./{job_file}", shell=True)
    print(f"Successfully executed {job_file}")

    return
# %%

# %%
if __name__=="__main__":
    # get current date-time
    now = datetime.datetime.now()
    strnow = now.strftime("%Y-%m-%d_%H-%M-%S")
    # hard-coding input parameters
    # ======================================
    # basic model parameters
    # ======================================
    # lat -- lattice type: chain, square, triangle
    lat = "square"
    # L -- lattice size: L spins for chain, LxL for square, triangle
    L = 4
    # pbc -- periodic boundary conditions: 0 is False 1 is True
    pbc = 0
    # lam -- transverse field strength: a float
    lam = 1.0
    # ======================================
    # Basic simulation parameters
    # ======================================
    # beta -- inverse temperature: positive float
    beta = 0.1
    # Tsteps -- equilibration steps: positive int
    Tsteps = 1000000
    # steps -- QMC updates: positive int
    steps = 10000000
    # stepsPer... -- : # QMC updates per measurement: positive int
    stepsPerMeasurement = 1000
    # ======================================
    # Simulation meta-parameters
    # ======================================
    # save -- if True, saves QMC simulation state at end of calculation: Bool
    save = False
    # restart -- if True, restarts calculation where left off: Bool
    restart = False
    # nt -- number of threads to run with MPI: positive int
    # for nt > 1, may fail due to OSX permissions. If so,
    # simply execute ./job_file.sh directly in base directory.
    nt = 6
    # ======================================
    # Advanced parameters 
    # ======================================
    # tau -- imaginary time: float in range [0.0, beta]
    tau = beta / 2
    # stand_tfim -- decides if standard (see top of this file): Bool
    # was True for arXiv:2504.07295 and False for arXiv:2408.03924v1
    stand_tfim = True
    # l -- number of terms in random Pauli unitary: non-negative int
    # rotates Hamiltonian if l > 0 (see Appendix A, arXiv:2408.03924v1)
    l = 0
    # parity -- samples within parity of: 0, 1, -1
    # only works for TFIM with stand_tfim = False
    # (see Appendix B, arXiv:2408.03924v2)
    # 0 is used for arXiv:2504.07295 and +1 for arXiv:2408.03924v1
    parity = 1
    # ======================================
    # Set-up and execute simulation
    # ======================================
    main(nt, L, lat, pbc, lam, beta, tau, parity, strnow, l=l, Tsteps=Tsteps, steps=steps, stepsPerMeasurement=stepsPerMeasurement,save=save, restart=restart, stand_tfim=stand_tfim)
 # %%
