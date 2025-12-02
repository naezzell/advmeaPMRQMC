"""
This code was written in support of the experiments carried out in:
* Nic Ezzell, Lev Barash, Itay Hen, Exact and universal quantum Monte Carlo estimators for energy susceptibility and fidelity susceptibility, arXiv:2408.03924 (2024).
* Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, arXiv:2504.07295 (2025).

Description: Contains useful IO utilities for building and executing our QMC program.
"""
# %%
import re
import numpy as np

def make_no_stand_param_fstr(beta, tau, Tsteps=100000, steps=1000000,
stepsPerMeasurement=10, parity=0, save=True, restart=False):
    """
    Writes parameters.hpp without any standard observables.
    """
    param_str = make_param_file_header(beta, tau, Tsteps, steps, stepsPerMeasurement, parity)
    param_str += make_param_file_footer(save, restart)
    return param_str

def make_hdiag_susceptibility_param_fstr(beta, fidsus=True, Tsteps=100000, steps=1000000,
stepsPerMeasurement=10, parity=0, save=True, restart=False):
    """
    Makes parameters.hpp with Hdiag measurements relevant to compute
    fidelity suscepibility.
    """
    param_str = make_param_file_header(beta, 0.0, Tsteps, steps, stepsPerMeasurement, parity)
    param_str += """
//
// Below is the list of standard observables:
//

#define MEASURE_HDIAG                // <H_{diag}>      is measured when this line is not commented
#define MEASURE_HDIAG_EINT

"""
    if fidsus is True:
        param_str += f"#define MEASURE_HDIAG_FINT\n"
    param_str += make_param_file_footer(save, restart)

    return param_str

def make_hoffdiag_susceptibility_param_fstr(beta, fidsus=True, Tsteps=100000, steps=1000000,
stepsPerMeasurement=10, parity=0, save=True, restart=False):
    """
    Makes parameters.hpp with Hoffdiag measurements relevant to compute
    fidelity suscepibility.
    """
    param_str = make_param_file_header(beta, 0.0, Tsteps, steps, stepsPerMeasurement, parity)
    param_str += """
//
// Below is the list of standard observables:
//

#define MEASURE_HOFFDIAG                // <H_{diag}>      is measured when this line is not commented
#define MEASURE_HOFFDIAG_EINT

"""
    if fidsus is True:
        param_str += f"#define MEASURE_HOFFDIAG_FINT\n"
    param_str += make_param_file_footer(save, restart)

    return param_str

def make_all_stand_param_fstr(beta, tau, Tsteps=10000000, steps=1000000,
stepsPerMeasurement=10, parity=0, save=True, restart=False):
    """
    Makes parameters.hpp with all standard observables.
    """
    param_str = make_param_file_header(beta, tau, Tsteps, steps, stepsPerMeasurement, parity)
    param_str += """
//
// Below is the list of standard observables:
//

#define MEASURE_H                    // <H>             is measured when this line is not commented
#define MEASURE_H2                   // <H^2>           is measured when this line is not commented
#define MEASURE_HDIAG                // <H_{diag}>      is measured when this line is not commented
#define MEASURE_HDIAG2               // <H_{diag}^2>    is measured when this line is not commented
#define MEASURE_HOFFDIAG             // <H_{offdiag}>   is measured when this line is not commented
#define MEASURE_HOFFDIAG2            // <H_{offdiag}^2> is measured when this line is not commented
#define MEASURE_Z_MAGNETIZATION      // Z-magnetization is measured when this line is not commented
#define MEASURE_HDIAG_CORR
#define MEASURE_HDIAG_EINT
#define MEASURE_HDIAG_FINT
#define MEASURE_HOFFDIAG_CORR
#define MEASURE_HOFFDIAG_EINT
#define MEASURE_HOFFDIAG_FINT
"""
    param_str += make_param_file_footer(save, restart)

    return param_str

def parse_otxt_temp(temp_fname):
    """
    Parse temporary output file.
    """
    with open(temp_fname, 'r') as f:
        lines = f.readlines()
    # define regular expression parser for numbers
    #p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    p = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan'
    # get emergent quantities
    sign = re.findall(p, lines[2])[0]
    sign_std = re.findall(p, lines[3])[0]
    mean_q = re.findall(p, lines[4])[0]
    max_q = re.findall(p, lines[5])[0]
    emergent = [sign, sign_std, mean_q, max_q]
    otxt_array = []
    otxt_std = []
    # get O.txt observables
    for j in range(1, 7):
        idx = 6 + 3*j - 2
        otxt_array.append(re.findall(p, lines[idx])[0])
        idx += 1
        otxt_std.append(re.findall(p, lines[idx])[0])
    otxt_array = np.array([float(x) for x in otxt_array])
    otxt_std = np.array([float(x) for x in otxt_std])
    # get time simulation took
    idx = 6 + 3*6
    time = re.findall(p, lines[idx])[0]

    return emergent, otxt_array, otxt_std, time

def parse_susceptibility_temp(temp_fname, fidsus=True):
    """
    Parse temporary output file.
    """
    with open(temp_fname, 'r') as f:
        lines = f.readlines()
    # define regular expression parser for numbers
    #p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    p = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan'
    # get emergent quantities
    sign = re.findall(p, lines[2])[0]
    sign_std = re.findall(p, lines[3])[0]
    mean_q = re.findall(p, lines[4])[0]
    max_q = re.findall(p, lines[5])[0]
    emergent = [sign, sign_std, mean_q, max_q]
    obs_array = []
    obs_std = []
    # get <O> (or Hdiag, Hoffdiag), and 1 or 2 integrals thereof
    max_idx = 4 if fidsus == True else 3
    for j in range(1, max_idx):
        idx = 6 + 3*j - 2
        obs_array.append(re.findall(p, lines[idx])[0])
        idx += 1
        obs_std.append(re.findall(p, lines[idx])[0])
    # get time simulation took
    idx = 6 + 3*(max_idx - 1)
    time = re.findall(p, lines[idx])[0]

    return emergent, obs_array, obs_std, time

def parse_correlator_temp(temp_fname):
    """
    Parse temporary output file.
    """
    with open(temp_fname, 'r') as f:
        lines = f.readlines()
    # define regular expression parser for numbers
    #p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    p = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan'
    # get emergent quantities
    sign = re.findall(p, lines[2])[0]
    sign_std = re.findall(p, lines[3])[0]
    mean_q = re.findall(p, lines[4])[0]
    max_q = re.findall(p, lines[5])[0]
    emergent = [sign, sign_std, mean_q, max_q]
    obs_array = []
    obs_std = []
    # get <O> and <O(\tau)O>
    for j in range(1, 3):
        idx = 6 + 3*j - 2
        obs_array.append(re.findall(p, lines[idx])[0])
        idx += 1
        obs_std.append(re.findall(p, lines[idx])[0])
    # get time simulation took
    idx = 6 + 3*2
    time = re.findall(p, lines[idx])[0]

    return emergent, obs_array, obs_std, time

#
# helper functions only for this file
#
def make_param_file_header(beta, tau, Tsteps=10000000, steps=1000000,
stepsPerMeasurement=10, parity=0):
    """
    Writes parameters.hpp header.
    """
    param_str = """
//
// This program implements Permutation Matrix Representation Quantum Monte Carlo for arbitrary spin-1/2 Hamiltonians.
//
// This program is introduced in the paper:
// Lev Barash, Arman Babakhani, Itay Hen, A quantum Monte Carlo algorithm for arbitrary spin-1/2 Hamiltonians, Physical Review Research 6, 013281 (2024).
//
// Various advanced measurement capabilities were added as part of the
// work introduced in the papers:
// * Nic Ezzell, Lev Barash, Itay Hen, Exact and universal quantum Monte Carlo estimators for energy susceptibility and fidelity susceptibility, arXiv:2408.03924 (2024).
// * Nic Ezzell and Itay Hen, Advanced measurement techniques in quantum Monte Carlo: The permutation matrix representation approach, arXiv:2504.07295 (2025).
//
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//
// ExExFloat datatype and calculation of divided differences are described in the paper:
// L. Gupta, L. Barash, I. Hen, Calculating the divided differences of the exponential function by addition and removal of inputs, Computer Physics Communications 254, 107385 (2020)
//

//
// Below are the parameter values:
//
    """
    # add actual adjustable parameters
    param_str += f"""
#define Tsteps {Tsteps} // number of Monte-Carlo initial equilibration updates
#define steps {steps} // number of Monte-Carlo updates
#define stepsPerMeasurement {stepsPerMeasurement} // number of Monte-Carlo updates per measurement
#define beta {beta} // inverse temperature
#define tau {tau} //imaginary propogation time
#define parity_cond {parity} // controls parity subspace measurement 
"""
    return param_str

def make_param_file_footer(save=True, restart=False):
    # add final technical parameters, not usually needed to adjust
    param_str = """
//
// Below are the implementation parameters:
//

#define qmax     1000                // upper bound for the maximal length of the sequence of permutation operators
#define Nbins    100                 // number of bins for the error estimation via binning analysis
#define EXHAUSTIVE_CYCLE_SEARCH      // comment this line for a more restrictive cycle search
#define GAPS_GEOMETRIC_PARAMETER 0.8 // parameter of geometric distribution for the length of gaps in the cycle completion update
#define COMPOSITE_UPDATE_BREAK_PROBABILITY  0.9   // exit composite update at each step with this probability

// #define ABS_WEIGHTS                  // uncomment this line to employ absolute values of weights rather than real parts of weights
// #define EXACTLY_REPRODUCIBLE         // uncomment this to always employ the same RNG seeds and reproduce exactly the same results

//
// Uncomment or comment the macros below to enable or disable the ability to checkpoint and restart
//
"""
    if save == True:
        param_str += """
#define SAVE_COMPLETED_CALCULATION   // save detailed data to "qmc_data_*.dat" when calculaiton is completed
#define SAVE_UNFINISHED_CALCULATION  // save calculation state to the files "qmc_data_*.dat" prior to exiting when SIGTERM signal is detected
#define HURRY_ON_SIGTERM             // uncomment this line to break composite update on SIGTERM signal to speed up the process
    """
    if restart == True:
        param_str += """
#define RESUME_CALCULATION           // attempt to read data from "qmc_data_*.dat" to resume the previous calculation
        """
    
    return param_str
#%%
