
//
// This program is introduced in the paper:
// Lev Barash, Arman Babakhani, Itay Hen, A quantum Monte Carlo algorithm for arbitrary spin-1/2 Hamiltonians (2023).
//
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//

//
// Below are parameter values used for a simple speedy test (not good for real simulations)
//
#define Tsteps 1  // number of Monte-Carlo initial equilibration updates
#define steps 2500  // number of Monte-Carlo updates
#define stepsPerMeasurement 10 // number of Monte-Carlo updates per measurement
#define beta 0.1 // inverse temperature
#define tau 0.05 //imaginary propogation time

//
// Below are reasonable parameter values (commented out for speed)
//
    
//#define Tsteps 1000000  // number of Monte-Carlo initial equilibration updates
//#define steps 10000000  // number of Monte-Carlo updates
//#define stepsPerMeasurement 10 // number of Monte-Carlo updates per measurement
//#define beta 1.0 // inverse temperature
//#define tau 0.5 //imaginary propogation time


#define parity_cond 0 // controls parity subspace measurement (leave at 0 unless advanced TFIM useage requires +/- 1)

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

//
// Below are the implementation parameters:
//

#define qmax     1000                // upper bound for the maximal length of the sequence of permutation operators
#define Nbins    250                 // number of bins for the error estimation via binning analysis
#define EXHAUSTIVE_CYCLE_SEARCH      // comment this line for a more restrictive cycle search
#define GAPS_GEOMETRIC_PARAMETER 0.8 // parameter of geometric distribution for the length of gaps in the cycle completion update
#define COMPOSITE_UPDATE_BREAK_PROBABILITY  0.9   // exit composite update at each step with this probability

// #define ABS_WEIGHTS                  // uncomment this line to employ absolute values of weights rather than real parts of weights
// #define EXACTLY_REPRODUCIBLE         // uncomment this to always employ the same RNG seeds and reproduce exactly the same results

//
// Uncomment or comment the macros below to enable or disable the ability to checkpoint and restart
//
// #define SAVE_COMPLETED_CALCULATION   // save detailed data to "qmc_data_*.dat" when calculaiton is completed
// #define SAVE_UNFINISHED_CALCULATION  // save calculation state to the files "qmc_data_*.dat" prior to exiting when SIGTERM signal is detected
// #define RESUME_CALCULATION           // attempt to read data from "qmc_data_*.dat" to resume the previous calculation
// #define HURRY_ON_SIGTERM             // break composite update on SIGTERM signal to speed up saving data; this is usually not necessary
