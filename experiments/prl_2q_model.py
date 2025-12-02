"""
Suscepibility experiment driver the 2 qubit PRL model in
[10.1103/PhysRevLett.100.100501].
H = Z_1 Z_2 + \gamma (X_1 + X_2) + \lambda(Z_1 + Z_2)
and random rotations thereof.

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
from ioscripts import make_all_stand_param_fstr, make_no_stand_param_fstr

def main(nt, n, gam, lam, beta, tau, strnow, seed, eps=None, l=None, Tsteps=1000000, steps=10000000, stepsPerMeasurement=10, save=True, restart=False):
    # ==============================================
    # Choose random parametrs if not given 
    # ==============================================
    np.random.seed(seed)
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
    if n < 2:
        raise ValueError("n must be greater than or equal to 2")
    # define basic 2q model from PRL Hamiltonian
    x1 = PauliTerm(1.0, [1], ['X'], n)
    x2 = PauliTerm(1.0, [2], ['X'], n)
    z1 = PauliTerm(1.0, [1], ['Z'], n)
    z2 = PauliTerm(1.0, [2], ['Z'], n)
    z1z2 = PauliTerm(1.0, [1, 2], ['Z', 'Z'], n)
    h0 = PauliH(n, [z1z2, gam * x1, gam * x2])
    h1 = PauliH(n, [z1, z2])

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
    a = PauliH(n, [px])
    # by default, B is a particular random B
    term1 = PauliTerm(0.683403, [1], ['Z'], n)
    term2 = PauliTerm(-0.643777, [1,2], ['Y', 'Z'], n)
    term3 = PauliTerm(-0.662378, [2], ['Z'], n)
    term4 = PauliTerm(0.738353, [1,2], ['Y', 'Y'], n)
    term5 = PauliTerm(-0.920660, [1,2], ['X', 'X'], n)
    pauli_terms = [term1, term2, term3, term4, term5]
    # uncomment below if you want B to be
    # B = sum of 5 random Paulis
    """
    pauli_terms = []
    for j in range(5):
        c = np.random.uniform(-1, 1)
        p = PauliTerm(c, [1], ['X'], n)
        if n >= 3:
            k = np.random.choice(range(1, 3+1))
        else:
            k = k = np.random.choice(range(1, n+1))
        p.make_random(k)
        pauli_terms.append(p)
    """
    b = PauliH(n, pauli_terms)
    with open(f"{dir_name}/A_unrotated.txt", 'w') as f:
        f.write(a.to_pmr_str())
    with open(f"{dir_name}/B_unrotated.txt", 'w') as f:
        f.write(b.to_pmr_str())
    with open(f"{dir_name}/U.txt", 'w') as f:
        f.write(u.to_pmr_str())
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
    #param_str = make_all_stand_param_fstr(beta, tau, Tsteps, steps, stepsPerMeasurement, 0, save, restart)
    param_str = make_no_stand_param_fstr(beta, tau, Tsteps, steps,stepsPerMeasurement, 0, save, restart)
    with open(f"{dir_name}/parameters.hpp", "w") as f:
        f.write(param_str)
    # compile and run PMR-QMC
    job_file = "job_file.sh"
    temp_out_file = f"n_{n}_l_{l}__2q_prl_experiment_{strnow}.txt"
    with open(f"{dir_name}/{job_file}", 'w') as fh:
        fh.write("#!/bin/bash\n")
        # compile and run
        fh.write("g++ -O3 -std=c++11 -o prepare.bin prepare.cpp\n")
        #fh.write("./prepare.bin H.txt A.txt A.txt A.txt A.txt A.txt B.txt B.txt B.txt B.txt B.txt A.txt A.txt A.txt A.txt A.txt\n")
        fh.write("./prepare.bin H.txt B.txt B.txt B.txt B.txt B.txt\n")
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
    # n -- number of spins to embed 2 qubit model into
    n = 100
    # lam -- coupling in front of (Z_1 + Z_2)
    lam = 1.0
    # gam -- coupling in front of tranverse field (X_1 + X_2)
    gam = 0.1
    # ======================================
    # Basic simulation parameters
    # ======================================
    # beta -- inverse temperature: positive float
    beta = 0.1
    # Tsteps -- equilibration steps: positive int
    Tsteps = 100000
    # steps -- QMC updates: positive int
    steps = 1000000
    # stepsPer... -- : # QMC updates per measurement: positive int
    stepsPerMeasurement = 10
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
    # l -- number of terms in random Pauli unitary: non-negative int
    # rotates Hamiltonian if l > 0 (see Appendix A, arXiv:2408.03924v1)
    l = None
    # ======================================
    # Set-up and execute simulation
    # ======================================
    high_sign_seeds = [3026438146, 2781301355, 4206712692, 2270171661, 2834120170, 1370102282, 2172515818,  185933287, 4109413259, 3629902865]
    main(nt, n, gam, lam, beta, tau, strnow, high_sign_seeds[0], l=l, Tsteps=Tsteps, steps=steps, stepsPerMeasurement=stepsPerMeasurement,save=save, restart=restart)
 # %%
