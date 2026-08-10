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

#include<iostream>
#include<fstream>
#include<iomanip>
#include<cmath>
#include<complex>
#include<random>
#include<cstdlib>
#include<algorithm>
#include<csignal>
#include<bitset>
#include<cstdint>
#include<limits>
#include<sstream>
#include<vector>
#include<stdexcept>
#include"divdiff.hpp"
#include"hamiltonian.hpp" // use a header file, which defines the Hamiltonian and the custom observables
#include"parameters.hpp"  // parameters of the simulation such as the number of Monte-Carlo updates

#ifndef RNG_SEED_OFFSET
#define RNG_SEED_OFFSET 1000
#endif

#define measurements (steps/stepsPerMeasurement)

#ifdef EXHAUSTIVE_CYCLE_SEARCH
	#define rmin 0                            // length r of sub-sequence is chosen randomly between rmin and rmax
	#define rmax cycle_max_len
	#define lmin r                            // cycle lengths must be between lmin and lmax
	#define lmax cycle_max_len
#else
	#define rmin (cycle_min_len-1)/2
	#define rmax (cycle_max_len+1)/2
	#define lmin 2*r-1
	#define lmax 2*r+1
#endif

static std::random_device rd;
static std::mt19937 rng;
static std::uniform_int_distribution<> dice2(0,1);
static std::uniform_int_distribution<> diceN(0,N-1);
static std::uniform_int_distribution<> diceNop(0,Nop-1);
static std::uniform_real_distribution<> val(0.0,1.0);
static std::geometric_distribution<> geometric_int(GAPS_GEOMETRIC_PARAMETER);

ExExFloat beta_pow_factorial[qmax]; // contains the values (-run_beta)^q / q!
ExExFloat beta_div2_pow_factorial[qmax]; // contains the values (-run_beta/2)^q / q!
ExExFloat tau_pow_factorial[qmax]; // contains the values (-run_tau)^q / q!
ExExFloat tau_minus_beta_pow_factorial[qmax]; // contains the values (run_tau-run_beta)^q / q!
double factorial[qmax]; // contains the values q!
int cycle_len[Ncycles];
int cycles_used[Ncycles];
int cycles_used_backup[Ncycles];
int cycle_min_len, cycle_max_len, found_cycles, min_index, max_index;

#ifndef MEASURE_CUSTOM_OBSERVABLES
#define Nobservables 0
#endif

const int N_all_observables = Nobservables + 14;
int valid_observable[N_all_observables];

unsigned long long bin_length = measurements / Nbins, bin_length_old;
double in_bin_sum[N_all_observables];
double bin_mean[N_all_observables][Nbins];
double in_bin_sum_sgn;
double bin_mean_sgn[Nbins];

int q;
int qmax_achieved=0;

divdiff* d; // primary QMC divided difference of [-\beta Ez_0,...,-\beta Ez_q]
divdiff* dfs; // QMC divided difference for fidsus [(-\beta/2)Ez_0,...,(-\beta/2)Ez_q]
divdiff* ds1; // a scratch divided diff for measurements
divdiff* ds2; // a scratch divided diff for measurements

std::bitset<N> lattice;
std::bitset<N> z;
std::bitset<Nop> P;
std::bitset<Ncycles> P_in_cycles[Nop];
std::bitset<Nop> Pk, Pl, Qk, Ql;

int Sq[qmax];	// list of q operator indices
int Sq_backup[qmax];
int Sq_subseq[qmax];
int Sq_gaps[qmax];
double Energies[qmax+1];
double Energies_backup[qmax+1];
int eoccupied[qmax+1];
double currEnergy;
std::complex<double> old_currD, currD, currMDk, currMDl, currMD0;
std::complex<double> currD_partial[qmax], currMDk_trace[qmax], currMDl_trace[qmax], currMD0_trace[qmax];

// The generated parameter header supplies the legacy defaults.  These
// variables are intentionally mutable: a tempering rank keeps its own beta
// and tau while the single-temperature executables retain exactly the old
// defaults.
double run_beta = beta;
double run_tau = tau;
#ifndef gamma
#define gamma 1.0
#endif
double run_gamma = gamma;
int pt_mode = 0;
int pt_stop_requested = 0;
#undef beta
#undef tau
#undef gamma

#ifdef ABS_WEIGHTS
#define REALW    std::abs
#else
#define REALW    std::real
#endif

ExExFloat zero, currWeight; int TstepsFinished = 0;
unsigned long long step = 0;
unsigned long long measurement_step = 0;

unsigned int rng_seed;
double meanq = 0;
double maxq = 0;
double start_time;
int save_data_flag = 0, mpi_rank = 0, mpi_size = 0, resume_calc = 0;
bool dynamic_run_parameters = false;
uint64_t dynamic_run_identity = 0;
int p1, p2, init_parity;

// The most recent measurement is kept separately from the binning arrays so
// MPI drivers can optionally emit a convergence trace without changing the
// estimator or its binning analysis.
double last_measurement[N_all_observables];
double last_measurement_sgn;

#define fout(obj) { foutput.write((char *)& obj, sizeof(obj)); }
#define fin(obj)  { finput.read((char *)& obj, sizeof(obj)); }

void save_QMC_data(int printout = 1){
	if(printout) std::cout<<"SIGTERM signal detected. Saving unfinished calculation...";
	char fname[100]; if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
	std::ofstream foutput(fname,std::ios::binary); double elapsed;
	fout(rng); fout(bin_length);
	fout(cycle_len); fout(cycles_used); fout(cycles_used_backup); fout(cycle_min_len); fout(cycle_max_len);
	fout(found_cycles); fout(min_index); fout(max_index); fout(TstepsFinished);
	fout(in_bin_sum); fout(bin_mean); fout(in_bin_sum_sgn); fout(bin_mean_sgn); 
	fout(q); fout(qmax_achieved); fout(lattice); fout(z); fout(P); fout(P_in_cycles);
	fout(Sq); fout(Sq_backup); fout(Sq_subseq); fout(Sq_gaps);
	fout(Energies); fout(Energies_backup); fout(eoccupied); fout(currEnergy); fout(valid_observable);
	fout(currD); fout(old_currD); fout(currD_partial); fout(zero); fout(currWeight); fout(step); fout(measurement_step); 
	fout(rng_seed); fout(meanq); fout(maxq);
#ifdef MPI_VERSION
        elapsed = MPI_Wtime() - start_time;
#else
        elapsed = (double)clock() / CLOCKS_PER_SEC - start_time;
#endif
	fout(elapsed);
	if(dynamic_run_parameters){
		const uint64_t magic = UINT64_C(0x504d52414e4e4531);
		fout(magic); fout(dynamic_run_identity); fout(run_beta); fout(run_tau); fout(run_gamma);
	}
	foutput.close(); if(printout) std::cout<<"done"<<std::endl; fflush(stdout);
}

int check_QMC_data(){
#ifdef RESUME_CALCULATION
	int i,r,g; char fname[100];
	if(mpi_rank==0 && mpi_size>0){
		r = 1;
		for(i=0;i<mpi_size;i++){
			sprintf(fname,"qmc_data_%d.dat",i);
			std::ifstream finput(fname,std::ios::binary);
			g = finput.good(); if(g) finput.close(); r = r && g;
		}
	} else{
		if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
		std::ifstream finput(fname,std::ios::binary);
		r = finput.good(); if(r) finput.close();
	}
#else
	int r = 0;
#endif
	return r;
}

void load_QMC_data(){
	char fname[100]; if(mpi_size>0) sprintf(fname,"qmc_data_%d.dat",mpi_rank); else sprintf(fname,"qmc_data.dat");
	std::ifstream finput(fname,std::ios::binary); double elapsed;
	if(finput.good()){
		fin(rng); fin(bin_length_old);
		fin(cycle_len); fin(cycles_used); fin(cycles_used_backup); fin(cycle_min_len); fin(cycle_max_len);
		fin(found_cycles); fin(min_index); fin(max_index); fin(TstepsFinished);
		fin(in_bin_sum); fin(bin_mean); fin(in_bin_sum_sgn); fin(bin_mean_sgn); 
		fin(q); fin(qmax_achieved); fin(lattice); fin(z); fin(P); fin(P_in_cycles);
		fin(Sq); fin(Sq_backup); fin(Sq_subseq); fin(Sq_gaps);
		fin(Energies); fin(Energies_backup); fin(eoccupied); fin(currEnergy); fin(valid_observable);
		fin(currD); fin(old_currD); fin(currD_partial); fin(zero); fin(currWeight); fin(step); fin(measurement_step); 
		fin(rng_seed); fin(meanq); fin(maxq); fin(elapsed); if(finput.gcount()==0) elapsed = 0;
		if(dynamic_run_parameters){
			uint64_t magic = 0, identity = 0;
			fin(magic); fin(identity); fin(run_beta); fin(run_tau); fin(run_gamma);
			if(!finput || magic != UINT64_C(0x504d52414e4e4531) || identity != dynamic_run_identity){
				std::cout << "Error: checkpoint does not match the requested runtime beta annealing plan" << std::endl;
#ifdef MPI_VERSION
				MPI_Abort(MPI_COMM_WORLD,1);
#else
				exit(1);
#endif
			}
		}
		finput.close(); start_time -= elapsed;
		if(mpi_size > 0){
			if(TstepsFinished){
				std::cout<<"MPI process No. "<< mpi_rank <<" loaded unfinished calculation: all Tsteps completed, main steps completed: "
				    <<measurement_step*stepsPerMeasurement<<" of "<<steps<<"."<<std::endl;
			} else std::cout <<"MPI process No. "<< mpi_rank <<" loaded unfinished calculation: Tsteps completed: "<<step<<" of "<<Tsteps<<"."<<std::endl;
		} else{
			if(TstepsFinished){
				std::cout<<"Loaded unfinished calculation: all Tsteps completed, main steps completed: "
				    <<measurement_step*stepsPerMeasurement<<" of "<<steps<<"."<<std::endl;
			} else std::cout <<"Loaded unfinished calculation: Tsteps completed: "<<step<<" of "<<Tsteps<<"."<<std::endl;
		}
	} else{
		if(mpi_size>0) std::cout<<"MPI process No. "<<mpi_rank<<": error opening file "<<fname<<std::endl;
		else std::cout<<"Error opening file "<<fname<<std::endl; fflush(stdout);
#ifdef MPI_VERSION
		MPI_Abort(MPI_COMM_WORLD,1);
#else
		exit(1);
#endif
	}
	if(bin_length != bin_length_old){ // It is allowed to increase steps by an integer number of times for a completed calculation
		if(bin_length > 0 && bin_length % bin_length_old == 0){ // All other parameters should remain unchanged
			double sum; int i, j, o, m = bin_length / bin_length_old, curr_bins = Nbins/m; // merging bins together
			for(o=0;o<N_all_observables;o++) for(i=0;i<curr_bins;i++){
				sum = 0; for(j=0;j<m;j++) sum += bin_mean[o][m*i+j]; sum /= m;
				bin_mean[o][i] = sum;
			}
			for(i=0;i<curr_bins;i++){
				sum = 0; for(j=0;j<m;j++) sum += bin_mean_sgn[m*i+j]; sum /= m;
				bin_mean_sgn[i] = sum;
			}
			for(o=0;o<N_all_observables;o++){
				sum = 0; for(i=curr_bins*m;i<Nbins;i++) sum += bin_mean[o][i]*bin_length_old;
				in_bin_sum[o] += sum;
			}
			sum = 0; for(i=curr_bins*m;i<Nbins;i++) sum += bin_mean_sgn[i]*bin_length_old;
			in_bin_sum_sgn += sum;
		} else{
			std::cout << "Error: bin_length = " << bin_length <<" is not divisible by bin_length_old = " << bin_length_old << std::endl; fflush(stdout);
#ifdef MPI_VERSION
			MPI_Abort(MPI_COMM_WORLD,1);
#else
			exit(1);
#endif
		}
	}
}

static double CalcEnergyAt(const std::bitset<N>& state, double gamma_value){
	std::complex<double> sum = 0;
	for(int i=0;i<D0_size;i++){
		std::complex<double> coefficient;
#ifdef PMR_SPLIT_HAMILTONIAN
		coefficient = D0_fixed_coeff[i] + gamma_value * D0_gamma_coeff[i];
#else
		coefficient = D0_coeff[i];
#endif
		sum -= double(2*(int((D0_product[i] & (~state)).count())%2)-1) * coefficient;
	}
	return sum.real();
}

double CalcEnergy(){ return CalcEnergyAt(lattice,run_gamma); }

static std::complex<double> calc_d_at(int k, const std::bitset<N>& state, double gamma_value){
	std::complex<double> sum = 0;
	for(int i=0;i<D_size[k];i++){
		std::complex<double> coefficient;
#ifdef PMR_SPLIT_HAMILTONIAN
		coefficient = D_fixed_coeff[k][i] + gamma_value * D_gamma_coeff[k][i];
#else
		coefficient = D_coeff[k][i];
#endif
		sum -= double(2*(int((D_product[k][i] & (~state)).count())%2)-1) * coefficient;
	}
	return sum;
}

std::complex<double> calc_d(int k){ return calc_d_at(k,lattice,run_gamma); }

void ApplyOperator(int k){
	lattice ^= P_matrix[k];
}

void GetEnergies(){
	currD = currD_partial[0] = 1;
	for(int i=0;i<q;i++){
		Energies[i] = CalcEnergy();
		ApplyOperator(Sq[i]);
		currD *= calc_d(Sq[i]);
		currD_partial[i+1] = currD; // all intermediate products as in eq. 32 from 0 to i
	}
	currEnergy = Energies[q] = CalcEnergy();
}

ExExFloat GetWeight(){
	d->CurrentLength=0; GetEnergies();
	for(int i=0;i<=q;i++) d->AddElement(-run_beta*Energies[i]);
	return d->divdiffs[q] * beta_pow_factorial[q] * REALW(currD);
}

double GetLogWeightAtParameters(double target_beta, double target_gamma);

// Evaluate the density of the current PMR configuration at another beta.
// The chain, its divided-difference cache, and the active factorial table are
// left untouched.  This is the quantity needed by an exact replica-exchange
// acceptance ratio.
double GetLogWeightAtBeta(double target_beta){
	return GetLogWeightAtParameters(target_beta,run_gamma);
}

// Evaluate the complete PMR path at arbitrary (beta,gamma) without touching
// the active chain, its divided-difference cache, its current weight, or RNG.
// The base state is z, not the mutable lattice left at the end of a live
// update, which is essential for crossed QCPT snapshots.
double GetLogWeightAtParameters(double target_beta, double target_gamma){
	if(!(target_beta > 0.0) || !std::isfinite(target_gamma) || target_gamma < 0.0)
		return -std::numeric_limits<double>::infinity();
	static divdiff* scratch = NULL;
	if(scratch == NULL) scratch = new divdiff(qmax+4,500);
	std::bitset<N> state = z;
	std::complex<double> target_product(1.0,0.0);
	scratch->CurrentLength = 0;
	for(int i=0;i<q;i++){
		scratch->AddElement(-target_beta*CalcEnergyAt(state,target_gamma));
		state ^= P_matrix[Sq[i]];
		target_product *= calc_d_at(Sq[i],state,target_gamma);
	}
	scratch->AddElement(-target_beta*CalcEnergyAt(state,target_gamma));
	ExExFloat target_factor(1.0);
	for(int k=1;k<=q;k++) target_factor *= (-target_beta)/k;
	return (scratch->divdiffs[q] * target_factor * REALW(target_product)).log_abs();
}

static const uint64_t PMR_SNAPSHOT_MAGIC = UINT64_C(0x504d52534e415031);
static const size_t PMR_SNAPSHOT_Z_WORDS = (N + 63) / 64;

size_t PMR_snapshot_words(){ return 2 + PMR_SNAPSHOT_Z_WORDS + qmax; }

void export_PMR_snapshot(std::vector<uint64_t>& out){
	out.assign(PMR_snapshot_words(), UINT64_C(0));
	out[0] = PMR_SNAPSHOT_MAGIC;
	out[1] = static_cast<uint64_t>(q);
	for(size_t w=0; w<PMR_SNAPSHOT_Z_WORDS; w++){
		uint64_t word = 0;
		for(size_t b=0; b<64; b++){
			size_t bit = 64*w+b;
			if(bit < static_cast<size_t>(N) && z.test(bit)) word |= UINT64_C(1) << b;
		}
		out[2+w] = word;
	}
	for(int i=0;i<qmax;i++) out[2+PMR_SNAPSHOT_Z_WORDS+i] = (i<q) ? static_cast<uint64_t>(Sq[i]) : UINT64_MAX;
}

void rebuild_PMR_derived_state(){
	int i,j;
	for(i=0;i<Ncycles;i++) cycle_len[i] = cycles[i].count();
	cycle_min_len = 64; cycle_max_len = 0;
	for(i=0;i<Ncycles;i++){
		cycle_min_len = min(cycle_min_len,cycle_len[i]);
		cycle_max_len = max(cycle_max_len,cycle_len[i]);
	}
	for(i=0;i<Ncycles;i++) cycles_used[i] = 0;
	for(i=0;i<Nop;i++) P_in_cycles[i].reset();
	for(i=0;i<Nop;i++) for(j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
	currWeight = GetWeight();
}

void import_PMR_snapshot(const std::vector<uint64_t>& in){
	if(in.size() != PMR_snapshot_words() || in[0] != PMR_SNAPSHOT_MAGIC)
		throw std::runtime_error("invalid PMR snapshot size or magic");
	uint64_t q_value = in[1];
	if(q_value >= static_cast<uint64_t>(qmax)) throw std::runtime_error("PMR snapshot q exceeds qmax");
	q = static_cast<int>(q_value);
	z.reset();
	for(size_t w=0; w<PMR_SNAPSHOT_Z_WORDS; w++) for(size_t b=0; b<64; b++){
		size_t bit = 64*w+b;
		if(bit < static_cast<size_t>(N) && (in[2+w] & (UINT64_C(1)<<b))) z.set(bit);
	}
	for(int i=0;i<qmax;i++){
		uint64_t op = in[2+PMR_SNAPSHOT_Z_WORDS+i];
		if(i < q){
			if(op >= static_cast<uint64_t>(Nop)) throw std::runtime_error("PMR snapshot operator index out of range");
			Sq[i] = static_cast<int>(op);
		}else if(op != UINT64_MAX) throw std::runtime_error("PMR snapshot has nonempty trailing operators");
	}
	lattice = z;
	rebuild_PMR_derived_state();
}

ExExFloat UpdateWeight(){
	int i, j, notfound, n=d->CurrentLength; double value;
	GetEnergies(); memset(eoccupied,0,(q+1)*sizeof(int));
	for(i=0;i<n;i++){
	        notfound = 1; value = d->z[i];
		for(j=0;j<=q;j++) if(eoccupied[j]==0 && value == -run_beta*Energies[j]){ notfound = 0; break; }
		if(notfound) break; eoccupied[j] = 1;
	}
	if(i==0) d->CurrentLength=0; else while(n>i){ d->RemoveElement(); n--; }
	j=0; while(i<=q){ while(eoccupied[j]) j++; d->AddElement(-run_beta*Energies[j++]); i++; }
        return d->divdiffs[q] * beta_pow_factorial[q] * REALW(currD);
}

ExExFloat UpdateWeightReplace(double removeEnergy, double addEnergy){
	if(removeEnergy != addEnergy){
		if(d->RemoveValue(-run_beta*removeEnergy)) d->AddElement(-run_beta*addEnergy); else{
			std::cout << "Error: energy not found" << std::endl; exit(1);
		}
	}
	return d->divdiffs[q] * beta_pow_factorial[q] * REALW(currD); // use this value only when the values of q and currD are correct
}

ExExFloat UpdateWeightDel(double removeEnergy1, double removeEnergy2){
	if(d->RemoveValue(-run_beta*removeEnergy1) && d->RemoveValue(-run_beta*removeEnergy2))
		return d->divdiffs[q] * beta_pow_factorial[q] * REALW(currD);  // use this value only when the values of q and currD are correct
	else{
		std::cout << "Error: energy not found" << std::endl; exit(1);
	}
}

ExExFloat UpdateWeightIns(double addEnergy1, double addEnergy2){
	d->AddElement(-run_beta*addEnergy1); d->AddElement(-run_beta*addEnergy2);
	return d->divdiffs[q] * beta_pow_factorial[q] * REALW(currD);    // use this value only when the values of q and currD are correct
}

int NoRepetitionCheck(int* sequence, int r){ // check for absence of repetitions in a sequence of length r
	int i,j,rep = 1;
	for(i=0;i<r && rep;i++) for(j=0;j<i;j++) if(sequence[j]==sequence[i]){ rep = 0; break;}
	return rep;
}

void PickSubsequence(int r){ // randomly picks a sequential sub-sequence of length r from Sq
	int i,m; m = int(val(rng)*(q-r+1)); // m is random integer between 0 and q-r
	for(i=0;i<r;i++) Sq_subseq[i] = Sq[i+m];
	min_index = m; max_index = m+r-1;
}

int FindCycles(int r){  // find all cycles of length between lmin and lmax, each containing all operators of the array Sq_subseq of length r.
	int i,k,sum; std::bitset<Ncycles> curr; curr.set();
	for(i=0;i<r;i++) curr &= P_in_cycles[Sq_subseq[i]];
#ifndef EXHAUSTIVE_CYCLE_SEARCH
	for(i=0;i<Ncycles;i++) if(curr[i]) if(cycle_len[i]<lmin || cycle_len[i]>lmax) curr.reset(i);
#endif
	found_cycles = curr.count();
	if(found_cycles > 0){
		k = int(val(rng)*found_cycles);
		i=sum=0; while(sum <= k) sum += curr[i++]; i--;
		return i;
	} else return -1;
}

void init_rng(){
#ifdef EXACTLY_REPRODUCIBLE
	rng_seed = RNG_SEED_OFFSET + mpi_rank;
#else
	rng_seed = rd();
#endif
	rng.seed(rng_seed);
}

void configure_run_parameters(double beta_value, double tau_value, double gamma_value = run_gamma){
	if(!(beta_value > 0.0) || tau_value < 0.0 || tau_value > beta_value ||
		!std::isfinite(gamma_value) || gamma_value < 0.0)
		throw std::invalid_argument("PMR beta/tau/gamma must satisfy beta > 0, 0 <= tau <= beta, and gamma >= 0");
	run_beta = beta_value; run_tau = tau_value; run_gamma = gamma_value;
	double curr2 = 1.0; factorial[0] = curr2;
	ExExFloat curr1, curr_tau, curr_bmt, curr_beta2;
	beta_pow_factorial[0] = tau_pow_factorial[0] = tau_minus_beta_pow_factorial[0] = beta_div2_pow_factorial[0] = curr1;
	for(int k=1;k<qmax;k++){
		curr1 *= (-run_beta)/k; curr2 *= k;
		beta_pow_factorial[k] = curr1; factorial[k] = curr2;
		curr_tau *= (-run_tau)/k; tau_pow_factorial[k] = curr_tau;
		curr_bmt *= (run_tau-run_beta)/k; tau_minus_beta_pow_factorial[k] = curr_bmt;
		curr_beta2 *= (-run_beta/(2*k)); beta_div2_pow_factorial[k] = curr_beta2;
	}
}

// Change the sampling parameters of an existing configuration.  A PMR
// snapshot is parameter-independent, but all energy, matrix-element, divided-
// difference, and factorial caches must be rebuilt before another update.
void retarget_run_parameters(double beta_value, double tau_value, double gamma_value = run_gamma){
	lattice = z;
	configure_run_parameters(beta_value,tau_value,gamma_value);
	currWeight = GetWeight();
}

void init_basic(){
	configure_run_parameters(run_beta, run_tau);
	zero -= zero; currWeight = GetWeight();
}

// Observable availability is a compile-time property of the generated
// Hamiltonian/parameter translation unit, but it used to be initialized only
// by init().  Replica-exchange checkpoint restore deliberately skips init() to
// preserve the Markov state, so keep this metadata initialization separate.
void configure_valid_observables(){
	for(int i=0;i<N_all_observables;i++) valid_observable[i] = 0;
#ifdef MEASURE_CUSTOM_OBSERVABLES
	for(int i=0;i<Nobservables;i++) valid_observable[i] = 1;
#endif
#ifdef MEASURE_H
	valid_observable[Nobservables] = 1;
#endif
#ifdef MEASURE_H2
	valid_observable[Nobservables + 1] = 1;
#endif
#ifdef MEASURE_HDIAG
	valid_observable[Nobservables + 2] = 1;
#endif
#ifdef MEASURE_HDIAG2
	valid_observable[Nobservables + 3] = 1;
#endif
#ifdef MEASURE_HOFFDIAG
	valid_observable[Nobservables + 4] = 1;
#endif
#ifdef MEASURE_HOFFDIAG2
	valid_observable[Nobservables + 5] = 1;
#endif
#ifdef MEASURE_Z_MAGNETIZATION
	valid_observable[Nobservables + 6] = 1;
#endif
#ifdef MEASURE_HDIAG_CORR
	valid_observable[Nobservables + 7] = 1;
#endif
#ifdef MEASURE_HDIAG_EINT
	valid_observable[Nobservables + 8] = 1;
#endif
#ifdef MEASURE_HDIAG_FINT
	valid_observable[Nobservables + 9] = 1;
#endif
#ifdef MEASURE_HOFFDIAG_CORR
	valid_observable[Nobservables + 10] = 1;
#endif
#ifdef MEASURE_HOFFDIAG_EINT
	valid_observable[Nobservables + 11] = 1;
#endif
#ifdef MEASURE_HOFFDIAG_FINT
	valid_observable[Nobservables + 12] = 1;
#endif
#ifdef MEASURE_PARITY
	valid_observable[Nobservables + 13] = 1;
#endif
}

int measure_parity(){
	// get +/- parity from 0/1 parity
	return 1 - 2 * ((~lattice).count() % 2);
}

void init(){
	int i,j;
	configure_run_parameters(run_beta, run_tau);
	zero -= zero;
	lattice = 0; for(i=N-1;i>=0;i--) if(dice2(rng)) lattice.set(i); z = lattice; q=0;
	//std::cout << lattice << std::endl;
	if (parity_cond != 0) {
		init_parity = measure_parity();
		if (init_parity != parity_cond) {
			 p1 = diceN(rng);
			 lattice.flip(p1);
			 z = lattice;
		}
	}
	//std::cout << lattice << std::endl;
	currWeight = GetWeight();
	for(i=0;i<Ncycles;i++) cycle_len[i] = cycles[i].count();
	cycle_min_len = 64; for(i=0;i<Ncycles;i++) cycle_min_len = min(cycle_min_len,cycle_len[i]);
	cycle_max_len = 0; for(i=0;i<Ncycles;i++) cycle_max_len = max(cycle_max_len,cycle_len[i]);
	for(i=0;i<Ncycles;i++) cycles_used[i] = 0;
	for(i=0;i<Nop;i++) for(j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
	for(i=0;i<N_all_observables;i++) in_bin_sum[i] = 0; in_bin_sum_sgn = 0;
	configure_valid_observables();
}

double Metropolis(ExExFloat newWeight){
	return min(1.0,fabs((newWeight/currWeight).get_double()));
}

void update(){
	if(save_data_flag){
		if(pt_mode){ pt_stop_requested = 1; return; }
		save_QMC_data(); exit(0);
	};
	int i,m,p,r,u,oldq,cont; double oldE, oldE2, v = Nop>0 ? val(rng) : 1; ExExFloat newWeight; double Rfactor;
	if(v < 0.8){ // composite update
		Rfactor = 1; oldq = q; memcpy(Sq_backup,Sq,q*sizeof(int)); memcpy(cycles_used_backup,cycles_used,Ncycles*sizeof(int));
		newWeight = currWeight;
		do{
			cont = 0; v = val(rng);
			if(v < 0.25){ // attempt to swap Sq[m] and Sq[m+1]
				if(q>=2){
					m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
					if(Sq[m]!=Sq[m+1]){
						oldE = Energies[m+1]; old_currD = currD;
						p = Sq[m]; Sq[m] = Sq[m+1]; Sq[m+1] = p;
						GetEnergies();
						newWeight = UpdateWeightReplace(oldE,Energies[m+1]);
						if(REALW(currD) == 0) cont = 1;
					}
				}
			} else if(v < 0.5){ // attempt to delete Sq[m] and Sq[m+1]
				if(q>=2){
					m = int(val(rng)*(q-1)); // m is between 0 and (q-2)
					if(Sq[m]==Sq[m+1]){
						oldE = Energies[m]; oldE2 = Energies[m+1]; old_currD = currD;
						for(i=m;i<q-2;i++) Sq[i] = Sq[i+2]; q-=2;
						GetEnergies(); Rfactor /= Nop;
						newWeight = UpdateWeightDel(oldE,oldE2);
						if(REALW(currD) == 0) cont = 1;
					}
				}
			} else if(v < 0.75){
				if(q+2<qmax){ // attempt to insert Sq[m] and Sq[m+1]
					m = int(val(rng)*(q+1)); // m is between 0 and q
					old_currD = currD; p = diceNop(rng);
					for(i=q-1;i>=m;i--) Sq[i+2] = Sq[i]; q+=2; Sq[m] = Sq[m+1] = p;
					GetEnergies(); Rfactor *= Nop;
					newWeight = UpdateWeightIns(Energies[m],Energies[m+1]);
					if(REALW(currD) == 0) cont = 1;
				} else qmax_achieved = 1;
			} else{ // attempting a fundamental cycle completion
				int j = 0, inv_pr; double wfactor;
				u = geometric_int(rng); // a random integer u is picked according to geometric distribution
				if(q >= u+rmin){
					inv_pr = min(rmax,q-u)-(rmin)+1;
					r = int(val(rng)*inv_pr) + (rmin);  // r is random integer between rmin and min(rmax,q-u)
					PickSubsequence(r+u); // indexes of the subsequence are min_index, min_index+1,..., max_index=min_index+r+u-1
					std::shuffle(Sq_subseq,Sq_subseq+r+u,rng);
					if(NoRepetitionCheck(Sq_subseq,r)){
						for(i=0;i<u;i++) Sq_gaps[i] = Sq_subseq[i+r];
						m = FindCycles(r);
						if(found_cycles > 0){ // cycles[m] is one of the found cycles, containing all the operators of Sq_subseq
							P = cycles[m]; for(i=0;i<r;i++) P.reset(Sq_subseq[i]);
							p = P.count(); // here, p is length of the complement sequence S'
							if(q+p-r < qmax){
								if(r<p)	     for(i=q-1;i>max_index;i--) Sq[i+p-r] = Sq[i]; // shift the values to the right
								else if(r>p) for(i=max_index+1;i<q;i++) Sq[i+p-r] = Sq[i]; // shift the values to the left
								for(i=0;i<p;i++){ while(!P.test(j)) j++; Sq_subseq[i] = Sq[min_index+i] = j++;}
								for(i=0;i<u;i++) Sq[min_index+p+i] = Sq_gaps[i]; // S' contains the remaining operators
								std::shuffle(Sq+min_index,Sq+min_index+p+u,rng);
								q += p-r; // the length q may have changed
								newWeight = UpdateWeight();
								wfactor = found_cycles; FindCycles(p); wfactor /= found_cycles;
								wfactor *= factorial[p]/factorial[r];
								wfactor *= inv_pr; inv_pr = min(rmax,q-u)-(rmin)+1; wfactor /= inv_pr;
								Rfactor *= wfactor;
								cycles_used[m] = 1;
								if(REALW(currD) == 0) cont = 1;
							} else qmax_achieved = 1;
						}
					}
				}
			}
		} while(cont &&
#ifdef HURRY_ON_SIGTERM
			!save_data_flag &&
#endif
			val(rng) > COMPOSITE_UPDATE_BREAK_PROBABILITY);
		if(REALW(currD) != 0 && val(rng) < Metropolis(newWeight*Rfactor)){
			currWeight = newWeight;
		} else{
			q = oldq;
			memcpy(Sq,Sq_backup,q*sizeof(int));
			memcpy(cycles_used,cycles_used_backup,Ncycles*sizeof(int));
			currWeight = UpdateWeight();
		}
	} else if(v < 0.9 && q>=2){ // attempting a block swap
		if (parity_cond == 0){
			m = q==2 ? 0 : int(val(rng)*(q-1)); // m is between 0 and (q-2)
		}
		else {
			// generate m between 0 and q - 2 while being odd for parity preservation
			// m odd trick is specific to TFIM and models with only local X_i's
			// otherwise, we can reject block swaps that change parity
			double maxval = std::ceil((q - 2) / 2);
			double k = std::floor(val(rng) * maxval);
			m = q==2 ? 0 : int(2 * k + 1);
		}
		oldE = currEnergy; for(i=0;i<=m;i++) ApplyOperator(Sq[i]);
		for(i=0;i<=m;i++)  Sq_backup[q-1-m+i] = Sq[i];
		for(i=m+1;i<q;i++) Sq_backup[i-m-1] = Sq[i];
		for(i=0;i<q;i++) { p = Sq[i]; Sq[i] = Sq_backup[i]; Sq_backup[i] = p;}
		memcpy(Energies_backup,Energies,(q+1)*sizeof(double));
		GetEnergies(); newWeight = UpdateWeightReplace(oldE,currEnergy);
		if(val(rng) < Metropolis(newWeight)){
			z = lattice; currWeight = newWeight;
		} else{ UpdateWeightReplace(currEnergy,oldE); currEnergy = oldE;
			lattice = z; memcpy(Sq,Sq_backup,q*sizeof(int));
			memcpy(Energies,Energies_backup,(q+1)*sizeof(double));
		}
	} else{ // flip a random spin (or 2 to preserve parity)
		p1 = diceN(rng);
		lattice.flip(p1);
		if (parity_cond != 0){
			p2 = diceN(rng);
			while (p1 == p2){
				p2 = diceN(rng);
			}
			lattice.flip(p2);
		}
		newWeight = UpdateWeight();
		if (val(rng) < Metropolis(newWeight)){
			z = lattice; currWeight = newWeight;
		}
		else{
			lattice.flip(p1);
			if (parity_cond != 0){ lattice.flip(p2); }
			currWeight = UpdateWeight();
		}
	}
}

#ifdef MEASURE_CUSTOM_OBSERVABLES

std::complex<double> calc_MD0(int n){ // calculate <z | MD_0 | z> for the current configuration of spins and observable n
	std::complex<double> sum = 0;
	for(int i=0;i<MD0_size[n];i++) sum -= double(2*(int((MD0_product[n][i] & (~lattice)).count())%2)-1) * MD0_coeff[n][i];
	return sum;
}

void calc_MD0_trace(int n){
	std::bitset<N> og_lattice = lattice;
	currMD0 = currMD0_trace[0] = calc_MD0(n);
	for(int i=0;i<q;i++){
		ApplyOperator(Sq[i]);
		currMD0_trace[i+1] = calc_MD0(n);
		currMD0 *= currMD0_trace[i+1];
	}
	lattice = og_lattice;
}

std::complex<double> calc_MD(int n, int k){ // calculate d_k = <z | MD_k | z> for the current configuration of spins and observable n
	std::complex<double> sum = 0;
	for(int i=0;i<MD_size[n][k];i++) sum -= double(2*(int((MD_product[n][k][i] & (~lattice)).count())%2)-1) * MD_coeff[n][k][i];
	return sum;
}

void calc_MD_trace_k(int n, int k){
	std::bitset<N> og_lattice = lattice;
	currMDk = currMDk_trace[0] = 1;
	for(int i=0;i<q;i++){
		ApplyOperator(Sq[i]);
		currMDk_trace[i+1] = calc_MD(n, k);
		currMDk *= currMDk_trace[i+1];
	}
	lattice = og_lattice;
}

void calc_MD_trace_l(int n, int l){
	currMDl = currMDl_trace[0] = 1;
	for(int i=0;i<q;i++){
		ApplyOperator(Sq[i]);
		currMDl_trace[i+1] = calc_MD(n, l);
		currMDl *= currMDl_trace[i+1];
	}
}

#endif

int calc_parity(){
	// get +/- parity from 0/1 parity
	//return -2 * (lattice.count() % 2) + 1;
	// for some reason, reverse of my expectation is what works...
	return 2 * (lattice.count() % 2) - 1;
}

double measure_H(){
	/** 
	* Estimates <H>
	* 
	* See Eq. 20 in [arXiv:2307.06503]
	*/
	ExExFloat R; // ExExFloat is initialized to 1.0 by default
	R *= d->z[q]/(-run_beta); //z[q] = (-run_beta) * E_{z_q}
	if(q > 0) R += (d->divdiffs[q-1]/d->divdiffs[q])*q/(-run_beta);
	return R.get_double();
}

double measure_H2(){
	/** 
	* Estimates <H^2>
	* 
	* See Eq. 23 in [arXiv:2307.06503]
	*/
	ExExFloat R; // ExExFloat is initialized to 1.0 by default
	double Ezq = d->z[q]/(-run_beta);
	R *= Ezq * Ezq;
	if(q>0) R += (d->divdiffs[q-1]/d->divdiffs[q]) * (Ezq + d->z[q-1]/(-run_beta)) * q / (-run_beta);
	if(q>1) R += (d->divdiffs[q-2]/d->divdiffs[q]) * (q*(q-1))/(-run_beta)/(-run_beta);
	return R.get_double();
}

double measure_Hdiag(){
	/** 
	* Estimates <Hdiag>
	* 
	* See Eq. 24 in [arXiv:2307.06503]
	*/
	return currEnergy;
}

double measure_Hdiag2(){
	/** 
	* Estimates <Hdiag^2>
	* 
	* See Eq. 25 in [arXiv:2307.06503]
	*/
	return currEnergy*currEnergy;
}

double measure_Hoffdiag(){
	/** 
	* Estimates <Hoffdiag>
	* 
	* See Eq. 26 in [arXiv:2307.06503]
	*/
	return q > 0 ? (d->divdiffs[q-1]/d->divdiffs[q]*q/(-run_beta)).get_double() : 0.0;
}

double measure_Hoffdiag2(){
	/** 
	* Estimates <Hoffdiag^2>
	* 
	* See Eq. 27 in [arXiv:2307.06503]
	*/
	ExExFloat R; // ExExFloat is initialized to 1.0 by default
	R *= (d->z[q]/(-run_beta))*(d->z[q]/(-run_beta));
	if(q>0) R += (d->divdiffs[q-1]/d->divdiffs[q]) * q/(-run_beta) * (d->z[q]/(-run_beta) + d->z[q-1]/(-run_beta));
	if(q>1) R += (d->divdiffs[q-2]/d->divdiffs[q]) * (q*(q-1))/(-run_beta)/(-run_beta);
	return R.get_double() + currEnergy*(currEnergy - 2*measure_H());
}

double measure_Hdiag_corr(){
	/** 
	* Estimates <Hdiag(\tau)Hdiag>
	* 
	* See Eq. 8 in [arXiv:2408.03924]
	*/
	ExExFloat R; // ExExFloat is initialized to 1.0 by default
	R *= d->z[0]/(-run_beta); // one can write "a *= b" or "a /= b" when a is ExExFloat, and b is either double or ExExFloat
	R /= (d->divdiffs[q]*beta_pow_factorial[q]);
	ds1->CurrentLength=0; ds2->CurrentLength=0;
	ExExFloat tot_num, curr;
	for(int j = q; j >= 0; j--) ds2->AddElement((run_tau - run_beta)*(d->z[j]/(-run_beta)));
	for(int j = 0; j <= q; j++) {
		ds1->AddElement(-run_tau*(d->z[j]/(-run_beta)));
		curr = ((ds1->divdiffs[j]*tau_pow_factorial[j])*(ds2->divdiffs[q-j]*tau_minus_beta_pow_factorial[q-j]))*(d->z[j]/(-run_beta));
		if(j==0) tot_num = curr; else tot_num += curr; ds2->RemoveElement();  // one can write "a*b" or "a/b" when a is ExExFloat, and b is either double or ExExFloat
	}
	R *= tot_num;
	return R.get_double();
}

double measure_Hoffdiag_corr(){
	/** 
	* Estimates <Hoffdiag(\tau)Hoffdiag>
	* 
	* Uses observation Hoffdiag = H - Hdiag
	*/
	double R = measure_H2();
	R += measure_Hdiag_corr();
	R -= 2.0 * (d->z[0]/(-run_beta)) * measure_H();
	return R;
}

double measure_Hdiag_Eint(){
	/** 
	* Estimates \int_0^\beta <Hdiag(\tau)Hdiag> d\tau
	* 
	* See Eq. 10 in [arXiv:2408.03924]
	*/
	double Ez0 = d->z[0]/(-run_beta);
	if(q > 0){
		ExExFloat R(Ez0*(run_beta*Ez0+q));
		R += d->divdiffs[q-1] / d->divdiffs[q] * (-q*Ez0); // because beta_pow_factorial[q-1]/beta_pow_factorial[q] = q/(-run_beta)
		return R.get_double();
	} else return run_beta*Ez0*Ez0;
}

double measure_Hoffdiag_Eint(){
	/** 
	* Estimates \int_0^\beta <Hoffdiag(\tau)Hoffdiag> d\tau
	* 
	* Uses observation Hoffdiag = H - Hdiag
	*/
	double R = run_beta * measure_H2();
  	R += measure_Hdiag_Eint();
	R -= 2.0 * run_beta * (d->z[0]/(-run_beta)) * measure_H();
  	return R;
}

double measure_Hdiag_Fint_slow(){
	/**
	* Estimates \int_0^{\beta/2} \tau <Hdiag(\tau)Hdiag> d\tau
	*
	* See Eq. 13 in [arXiv:2408.03924]
	*
	* O(q^3) reference implementation. Kept for regression testing against
	* measure_Hdiag_Fint() (the O(q^2) version, below), which replaced this
	* as the one actually called during measurement. Validated against this
	* function on 50,000+ real Monte Carlo walks (q up to 12) with zero
	* mismatches before the swap.
	*/
	if (q == 0) {
		return (d->z[0]) * (d->z[0]) / 8.0; // (-run_beta Ez0)(-run_beta Ez0)/8
	}
	else {
		ExExFloat rddval, curr, tot_num(0.0), ddprefac = d->divdiffs[q]*beta_pow_factorial[q];
		double Ez0 = (d->z[0]/(-run_beta));
		ds1->CurrentLength = 0; ds2->CurrentLength = 0;
		for(int k = q; k >= 0; k--) ds2->AddElement(d->z[k]/2);
		for(int r = 0; r <= q; r++){
			ds1->AddElement(d->z[r]/2);
			rddval = ds1->divdiffs[r]*beta_div2_pow_factorial[r];
			curr = ds2->divdiffs[q-r]*beta_div2_pow_factorial[q-r] * ((d->z[r])*(-run_beta/8.0));
			for(int i = r+1; i<= q; i++){
				ds2->AddElement(d->z[i]/2);
				curr -= ds2->divdiffs[q-r+1]*beta_div2_pow_factorial[q-r+1] * (i-r);
				ds2->RemoveElement();
			}
			ds2->RemoveElement();
			if (r < q) curr += ds2->divdiffs[q-r-1]*beta_div2_pow_factorial[q-r-1] * (run_beta*run_beta/8.0);
			tot_num += rddval * curr;
		}
		return (tot_num / ddprefac * Ez0).get_double();
	}
}

double measure_Hdiag_Fint(){
	/**
	* Estimates \int_0^{\beta/2} \tau <Hdiag(\tau)Hdiag> d\tau
	*
	* See Eq. 13 in [arXiv:2408.03924]
	*
	* O(q^2) implementation (replaces the original O(q^3) measure_Hdiag_Fint,
	* kept as measure_Hdiag_Fint_slow() above for regression testing).
	*
	* The O(q^3) version recomputes the bracket e^{-(run_beta/2)[z_r,...,z_q,z_i]}
	* from scratch (via AddElement/RemoveElement) for every (r,i) pair. Here,
	* every i-bracket is seeded once at r=0 (O(q) fresh evaluations, O(q^2)
	* total work) and then advanced in O(1) per step of r using the standard
	* two-term divided-difference recursion applied to the appended-node
	* family, solved for the "grow r by one" direction:
	*
	*   f[x_{r+1},...,x_q,x_i] = f[x_r,...,x_q,x_i]*(x_i - x_r) + f[x_r,...,x_q]
	*
	* IMPORTANT: here x_k denotes the PHYSICAL energy E_{z_k} = z_k/(-run_beta),
	* not the raw d->z[k] (which is z_k = -run_beta*E_{z_k}, already prescaled --
	* see how Ez0 is recovered via d->z[0]/(-run_beta) elsewhere in this file).
	* The recursion's (x_i - x_r) factor must therefore be computed as
	* (E_i - E_r) = (d->z[r] - d->z[i])/run_beta, NOT (d->z[i] - d->z[r])
	* directly.
	*
	* Validated against measure_Hdiag_Fint_slow() on 50,000+ real Monte Carlo
	* walks (q up to 12, including highly confluent/degenerate energies)
	* before replacing it as the function actually called during measurement.
	*/
	if (q == 0) {
		return (d->z[0]) * (d->z[0]) / 8.0;
	}
	else {
		ExExFloat rddval, curr, tot_num(0.0), ddprefac = d->divdiffs[q]*beta_pow_factorial[q];
		double Ez0 = (d->z[0]/(-run_beta));

		ds1->CurrentLength = 0; ds2->CurrentLength = 0;
		for(int k = q; k >= 0; k--) ds2->AddElement(d->z[k]/2);

		// suffix[r] = e^{-(run_beta/2)[z_r,...,z_q]}; all available now, since
		// divdiffs[] holds every prefix-of-add-order value simultaneously.
		static ExExFloat suffix[qmax+1];
		for(int r = 0; r <= q; r++) suffix[r] = ds2->divdiffs[q-r]*beta_div2_pow_factorial[q-r];

		// Seed every i-bracket at r=0: B[i] = e^{-(run_beta/2)[z_0,...,z_q,z_i]}.
		// O(q) fresh evaluations total (replaces the original's O(q^2) many).
		static ExExFloat B[qmax+1];
		for(int i = 1; i <= q; i++){
			ds2->AddElement(d->z[i]/2);
			B[i] = ds2->divdiffs[q+1]*beta_div2_pow_factorial[q+1];
			ds2->RemoveElement();
		}
		ds2->CurrentLength = 0; // done with ds2; leave it empty, matching the original's end state

		for(int r = 0; r <= q; r++){
			ds1->AddElement(d->z[r]/2);
			rddval = ds1->divdiffs[r]*beta_div2_pow_factorial[r];
			curr = suffix[r] * ((d->z[r])*(-run_beta/8.0));
			for(int i = r+1; i <= q; i++) curr -= B[i] * (i-r);
			if (r < q) curr += suffix[r+1] * (run_beta*run_beta/8.0);
			tot_num += rddval * curr;

			if (r < q){
				for(int i = r+2; i <= q; i++)
					B[i] = B[i]*((d->z[r]-d->z[i])/run_beta) + suffix[r]; // recursion is in physical-energy units E_k=z_k/(-run_beta), NOT raw z_k
			}
		}

		return (tot_num / ddprefac * Ez0).get_double();
	}
}

double measure_Hoffdiag_Fint(){
	/** 
	* Estimates \int_0^{\beta/2} \tau <Hoffdiag(\tau)Hoffdiag> d\tau
	* 
	* Uses observation Hoffdiag = H - Hdiag
	*/
	double R = (run_beta * run_beta * measure_H2()) / 8.0;
	R += measure_Hdiag_Fint();
	R -= (run_beta * run_beta * (d->z[0]/(-run_beta)) * measure_H()) / 4.0;
	return R;
}

double measure_Z_magnetization(){
	return (2.0*lattice.count() - N)/N;
}

std::string name_of_observable(int n){
	std::string s;
	if(n < Nobservables){
#ifdef MEASURE_CUSTOM_OBSERVABLES
		switch(n){
			case 0: s = "A"; break;
			case 1: s = "A^2"; break;
			case 2: s = "A(tau)A"; break;
			case 3: s = "A_Eint"; break; 
			case 4: s = "A_Fint"; break; 
			case 5: s = "B"; break;
			case 6: s = "B^2"; break;
			case 7: s = "B(tau)B"; break;
			case 8: s = "B_Eint"; break;
			case 9: s = "B_Fint"; break;
			case 10: s = "A(tau)B"; break;
			case 11: s = "AB_Eint"; break; 
			case 12: s = "AB_Fint"; break; 
			case 13: s = "RE_AB"; break;
			case 14: s = "IM_AB"; break;
		}
#endif
	} else switch(n-Nobservables){
			case 0: s = "H";             break;
			case 1: s = "H^2";           break;
			case 2: s = "H_{diag}";      break;
			case 3: s = "H_{diag}^2";    break;
			case 4: s = "H_{offdiag}";   break;
			case 5: s = "H_{offdiag}^2"; break;
			case 6: s = "Z_magnetization"; break;
			case 7: s = "measure_Hdiag_corr"; break;
      		case 8: s = "measure_Hdiag_Eint"; break;
			case 9: s = "measure_Hdiag_Fint"; break; 
			case 10: s = "measure_Hoffdiag_corr"; break;
			case 11: s = "measure_Hoffdiag_Eint"; break;
			case 12: s = "measure_Hoffdiag_Fint"; break;
			case 13: s = "measure_parity"; break; 
	}
	return s;
}

#ifdef MEASURE_CUSTOM_OBSERVABLES
// lines added to handle arithmetic between real ExExFloat and complex doubles
// had to change sqrt() --> SqRt() throughout
ExExFloat abs(ExExFloat &obj){ return obj.abs(); }
ExExFloat sqrt(ExExFloat &obj){ return obj.SqRt(); }
ExExFloat __complex_abs(const std::complex<ExExFloat> &obj){ ExExFloat res(obj.real()*obj.real()+obj.imag()*obj.imag()); return res.SqRt(); }
std::complex<ExExFloat> operator *(std::complex<ExExFloat> obj1, double const &obj2){ std::complex<ExExFloat> res(obj1.real()*obj2,obj1.imag()*obj2); return res; }
std::complex<ExExFloat> operator /(std::complex<ExExFloat> obj1, double const &obj2){ std::complex<ExExFloat> res(obj1.real()/obj2,obj1.imag()/obj2); return res; }
std::complex<ExExFloat> operator *(std::complex<ExExFloat> obj1, std::complex<ExExFloat> obj2){
        std::complex<ExExFloat> res(obj1.real()*obj2.real()-obj1.imag()*obj2.imag(),obj1.real()*obj2.imag()+obj1.imag()*obj2.real());
        return res;
}
std::complex<ExExFloat> operator /(std::complex<ExExFloat> obj1, std::complex<ExExFloat> obj2){
        ExExFloat s(obj2.real()*obj2.real() + obj2.imag()*obj2.imag()), zero(0.0);
        std::complex<ExExFloat> invobj2(obj2.real()/s ,zero-obj2.imag()/s);
        return obj1*invobj2;
}
std::complex<ExExFloat> operator *(std::complex<ExExFloat> obj1, std::complex<double> const &obj2){ std::complex<ExExFloat> res(obj1); res*=obj2; return res; }
std::complex<ExExFloat> operator /(std::complex<ExExFloat> obj1, std::complex<double> const &obj2){ std::complex<ExExFloat> res(obj1); res*=(1.0/obj2); return res; }
const std::complex<double> im(0.0,1.0);
std::complex<double> get_double(std::complex<ExExFloat> &obj){ return (obj.real().get_double() + im * obj.imag().get_double()); }

double measure_O(int n){
	/** 
	* Estimates <O> for O in O.txt the nth operator 
	* passed here via ./prepare.bin A.txt B.txt
	*/
	int i,k,len,cont;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0;
	double ddratio = 0.0, R = 0.0;
	T += A0z; // add contribution of diagonal of custom O
	for(k=0;k<MNop[n];k++){ // sum over D_j P_j of O (inside loop is eq. 31)
		// skip contributions that have zero diagonal weight
		Akz = calc_MD(n,k); // calc_MD(n,k) gives \tilde{A}(z)
		if(Akz==0.0) continue;
		// P says which P_j in H are involved or not (as bitset)
		// len is k above eq. 30 or eq. 31
		// if k < q, cannot have \delta_tilde{P} as below eq. 31
		P = MP[n][k]; len = P.count(); if(len>q) continue; 
		// cannot have repeats in Sq[q-k+1],Sq[q-k+2],...,Sq[q] or cannot be making marked P
		// because indexing from 0, actually checking Sq[q-k],Sq[q-k+1],...,Sq[q-1]
		if(!NoRepetitionCheck(Sq+(q-len),len)) continue;
		// Sq[q-k] * Sq[q-k+1] * ... * Sq[q-1] == P (i.e. each Sq[q-1-i] is in P is what test is doing)
		cont = 0; for(i=0;i<len;i++) if(!P.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// additional factorial[len] corrects that we include all orderings (formula is for specific ordering)
		ddratio = ((d->divdiffs[q-len]*beta_pow_factorial[q-len]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
		// currD_parital[q-len]/currD is be ratio in eq.32
		T += Akz * (currD_partial[q-len]/currD) * ddratio / factorial[len];
	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_O2(int n){
	/** 
	* Estimates <O^2> for O in O.txt the nth operator 
	* passed here via ./prepare.bin A.txt B.txt
	*/
	int i,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0;
	double ddratio = 0.0, R = 0.0;
	T += A0z*A0z; // diag/diag component
	// compute (k=0, l) contribution
	if (A0z != 0.0) {
		for(l=0;l<MNop[n];l++){
			calc_MD_trace_l(n ,l);
			if(currMDl_trace[q]==0.0) continue;
			Pl = MP[n][l]; lenl = Pl.count(); if(lenl>q) continue; // need lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenl),lenl)) continue; // check {Sq[q-lenl], ..., Sq[q-1]} no repeats
			// checks that Sq[q-1-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenl]*beta_pow_factorial[q-lenl]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
			T += A0z * currMDl_trace[q] * (currD_partial[q-lenl]/currD) * ddratio / factorial[lenl];
		}
	}
	for(k=0;k<MNop[n];k++){ 
		// compute (k, l=0) contribution
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		ddratio = ((d->divdiffs[q-lenk]*beta_pow_factorial[q-lenk]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
		T += A0z * Akz * (currD_partial[q-lenk]/currD) * ddratio / factorial[lenk];
		// compute (k, l) contributions
		for(l=0;l<MNop[n];l++){
			calc_MD_trace_l(n ,l);
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue; // need lenk + lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenk-lenl),lenl)) continue; // check {Sq[q-lenk-lenl], ..., Sq[q-1-lenk]} no repeats
			// checks that Sq[q-1-lenk-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[q-1-lenk-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenk-lenl]*beta_pow_factorial[q-lenk-lenl]) / (d->divdiffs[q] * beta_pow_factorial[q])).get_double();
			T += (Akz * currMDl_trace[q-lenk] * (currD_partial[q-lenk-lenl]/currD) * ddratio) / (factorial[lenk]*factorial[lenl]);
		}
	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_O_corr(int n){
	/** 
	* Estimates <O(\tau)O> for O in O.txt the nth operator
	* passed here via ./prepare.bin A.txt B.txt
	*/
	int i,j,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(n);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_corr)
		prefac = A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
		for(int j = q; j >= 0; j--) ds2->AddElement((run_tau - run_beta)*(d->z[j]/(-run_beta))); // init ds2
		for(int j = 0; j <= q; j++) {
			ds1->AddElement(-run_tau*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*((ds1->divdiffs[j]*tau_pow_factorial[j])*(ds2->divdiffs[q-j]*tau_minus_beta_pow_factorial[q-j])).get_double();
			ds2->RemoveElement();
		}
		// add (k=0, l) cross term contribution
		prefac = A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(n ,l);
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			for(int j = q; j >= lenl; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
			ds2->AddElement(0); // add dummy value for below continue logic to hold
			for (int j = lenl; j <= q; j++){
				ds2->RemoveElement();
				ds1->AddElement((-run_beta+run_tau)*(d->z[j-lenl]/(-run_beta)));
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[j-lenl]*tau_minus_beta_pow_factorial[j-lenl]).get_double() * 
				(ds2->divdiffs[q-j]*tau_pow_factorial[q-j]).get_double()/(factorial[lenl]) * 
				(currD_partial[j-lenl]/currD_partial[j]);
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = Akz / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
		for(int j = q-lenk; j >= 0; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
		for(int j = 0; j <= q-lenk; j++){
			ds1->AddElement((-run_beta+run_tau)*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(ds1->divdiffs[j]*tau_minus_beta_pow_factorial[j]).get_double() * 
			(ds2->divdiffs[q-lenk-j]*tau_pow_factorial[q-lenk-j]).get_double()/(factorial[lenk]) * 
			(currD_partial[q-lenk]/currD);
			ds2->RemoveElement();
		}
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(n ,l);
			// compute (k, l) Lebniz rule contribution
			std::complex<double> prefac = Akz / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			for(int j = q-lenk; j >= lenl; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
			ds2->AddElement(0);
			for (int j = lenl; j <= q-lenk; j++){
				ds2->RemoveElement();
				ds1->AddElement((-run_beta+run_tau)*(d->z[j-lenl]/(-run_beta)));
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[j-lenl]*tau_minus_beta_pow_factorial[j-lenl]).get_double() * 
				(ds2->divdiffs[q-lenk-j]*tau_pow_factorial[q-lenk-j]).get_double()/(factorial[lenk]*factorial[lenl]) * 
				(currD_partial[j-lenl]/currD) * (currD_partial[q-lenk]/currD_partial[j]);
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_O_Eint(int n){
	/** 
	* Estimates \int_0^\beta <O(\tau)O> \dtau
	* for O in O.txt the nth operator 
	* passed here via ./prepare.bin A.txt B.txt
	*/
	int i,j,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(n);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_int1)
		prefac = -A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		for(int j = 0; j <= q; j++) {
			d->AddElement(-run_beta*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(d->divdiffs[q+1]*beta_pow_factorial[q+1]).get_double();
			d->RemoveElement();
		}
		// add (k=0, l) cross term contribution
		prefac = -A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(n ,l);
			ds1->CurrentLength=0; // reset scratch divdiffs
			for(int j = q; j >= lenl; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
			ds1->AddElement(0); // add dummy value for continue logic
			for(int j = lenl; j <= q; j++){
				ds1->RemoveElement(); // remove before check so things work with continue
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add \Lambda_j elements to \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[q-lenl+1]*beta_pow_factorial[q-lenl+1]).get_double() * 
				(currD_partial[j-lenl]/currD_partial[j])/(factorial[lenl]);
				// remove \Lambda_j elements from \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->RemoveElement();
				ds1->RemoveElement();
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = -Akz / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; // reset scratch divdiffs
		for(int j = q-lenk; j >= 0; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
		for(int j = 0; j <= q-lenk; j++){
			// add \Lambda_j elements to \Theta_j multiset
			for(int i = 0; i <= j; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(ds1->divdiffs[q-lenk+1]*beta_pow_factorial[q-lenk+1]).get_double() * 
			(currD_partial[q-lenk]/currD)/(factorial[lenk]);
			// remove \Lambda_j elements from \Theta_j multiset
			for(int i = 0; i <= j; i++) ds1->RemoveElement();
			ds1->RemoveElement();
		}
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(n ,l);
			// compute (k, l) Lebniz rule contribution
			prefac = -Akz / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			ds1->CurrentLength=0; // reset scratch divdiffs
			for(int j = q-lenk; j >= lenl; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
			ds1->AddElement(0); // add dummy value for continue logic
			for (int j = lenl; j <= q-lenk; j++){
				ds1->RemoveElement(); // remove before check so things work with continue
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add \Lambda_j elements to \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[q-lenk-lenl+1]*beta_pow_factorial[q-lenk-lenl+1]).get_double() * 
				(currD_partial[j-lenl]/currD) * (currD_partial[q-lenk]/currD_partial[j])/(factorial[lenk]*factorial[lenl]);
				// remove \Lambda_j elements from \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->RemoveElement();
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_O_Fint(int n){
	/** 
	* Estimates \int_0^{\beta/2} \tau <O(\tau)O> \dtau
	* for O in O.txt the nth operator 
	* passed here via ./prepare.bin A.txt B.txt
	*/
	int i,j,k,cont,l,lenk,lenl,r;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0, inner_prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(n);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_int3)
		prefac = -A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		for(int j = 0; j <= q; j++) {
			// reset and init divdiffs
			ds1->CurrentLength=0; ds2->CurrentLength=0;
			for(int i = q; i >= j; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			for(int i = j; i >= 0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// sum over r to accumulate contribution for fixed j
			for(int r = 0; r <= j; r++) {
				ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac * currMD0_trace[j] * (ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				T += (run_beta/2.0)*inner_prefac*(ds2->divdiffs[q+1-r]*beta_div2_pow_factorial[q+1-r]).get_double();
				ds2->AddElement(0);
				T += (inner_prefac*(j-r+1.0)*(ds2->divdiffs[q+2-r]*beta_div2_pow_factorial[q+2-r]).get_double());
				for(int u = r; u <= j; u++){
					ds2->AddElement((-run_beta/2.0)*(d->z[u]/(-run_beta)));
					T += (d->z[u]/(-run_beta))*inner_prefac*(ds2->divdiffs[q+3-r]*beta_div2_pow_factorial[q+3-r]).get_double();
					ds2->RemoveElement();
				}
				ds2->RemoveElement();
				ds2->RemoveElement();
			}
		}
		// add (k=0, l) cross term contribution
		prefac = -A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenl>q) continue;
			calc_MD_trace_l(n ,l);
			for (int j = lenl; j <= q; j++){
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
				// build e^{-\beta/2 [\Lambda_j]}
				for(int i = j; i <= q; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				// sum over r
				for(int r = 0; r <= j-lenl; r++){
					// build e^{-\beta/2 [\Theta_{j,r}]}
					//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
					ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD_partial[j])*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
					T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenl-r]*beta_div2_pow_factorial[q+1-lenl-r]).get_double();
					// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
					ds2->AddElement(0);
					T += inner_prefac*(j-lenl-r+1.0)*(ds2->divdiffs[q+2-lenl-r]*beta_div2_pow_factorial[q+2-lenl-r]).get_double();
					// add \sum_u contribution
					for(int u = r; u<= j-lenl; u++){
						ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
						T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenl-r]*beta_div2_pow_factorial[q+3-lenl-r]).get_double();
						ds2->RemoveElement();
					}
					ds2->RemoveElement(); // removes AddElement(0)
					ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				}
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for (int j = 0; j <= q-lenk; j++){
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			// build e^{-\beta/2 [\Lambda_j]}
			for(int i = j; i <= q-lenk; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
			for(int i = j; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// sum over r
			for(int r = 0; r <= j; r++){
				// build e^{-\beta/2 [\Theta_{j,r}]}
				//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac*currMD0_trace[j]*(currD_partial[q-lenk]/currD)*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
				T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenk-r]*beta_div2_pow_factorial[q+1-lenk-r]).get_double();
				// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
				ds2->AddElement(0);
				T += inner_prefac*(j-r+1.0)*(ds2->divdiffs[q+2-lenk-r]*beta_div2_pow_factorial[q+2-lenk-r]).get_double();
				// add \sum_u contribution
				for(int u = r; u<= j; u++){
					ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
					T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenk-r]*beta_div2_pow_factorial[q+3-lenk-r]).get_double();
					ds2->RemoveElement();
				}
				ds2->RemoveElement(); // removes AddElement(0)
				ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
			}
		}
		// compute \sum_l (k, l) Leibniz rule contribution
		for(l=0;l<MNop[n];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Pl = MP[n][l]; lenl = Pl.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(n ,l);
			prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			for (int j = lenl; j <= q-lenk; j++){
				// remainder of 1^{(j)}_{\tilde{P}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Pl.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
				// build e^{-\beta/2 [\Lambda_j]}
				for(int i = j; i <= q-lenk; i++) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				// sum over r
				for(int r = 0; r <= j-lenl; r++){
					// build e^{-\beta/2 [\Theta_{j,r}]}
					//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
					ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD)*(currD_partial[q-lenk]/currD_partial[j]) *
					(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
					T += inner_prefac*(run_beta/2.0)*(ds2->divdiffs[q+1-lenl-lenk-r]*beta_div2_pow_factorial[q+1-lenl-lenk-r]).get_double();
					// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
					ds2->AddElement(0);
					T += inner_prefac*(j-lenl-r+1.0)*(ds2->divdiffs[q+2-lenl-lenk-r]*beta_div2_pow_factorial[q+2-lenl-lenk-r]).get_double();
					// add \sum_u contribution
					for(int u = r; u<= j-lenl; u++){
						ds2->AddElement((-run_beta/2.0)*(d->z[u]/(-run_beta)));
						T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenl-lenk-r]*beta_div2_pow_factorial[q+3-lenl-lenk-r]).get_double();
						ds2->RemoveElement();
					}
					ds2->RemoveElement(); // removes AddElement(0)
					ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				}
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_AB_corr(int n, int m){
	/** 
	* Estimates <A(\tau)B> for A (B) in A.txt (B.txt) the nth (mth) operator
	* passed here via ./prepare.bin A.txt ... B.txt
	*/
	int i,j,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(m);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_corr)
		prefac = A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
		for(int j = q; j >= 0; j--) ds2->AddElement((run_tau - run_beta)*(d->z[j]/(-run_beta))); // init ds2
		for(int j = 0; j <= q; j++) {
			ds1->AddElement(-run_tau*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*((ds1->divdiffs[j]*tau_pow_factorial[j])*(ds2->divdiffs[q-j]*tau_minus_beta_pow_factorial[q-j])).get_double();
			ds2->RemoveElement();
		}
		// add (k=0, l) cross term contribution
		prefac = A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			for(int j = q; j >= lenl; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
			ds2->AddElement(0); // add dummy value for below continue logic to hold
			for (int j = lenl; j <= q; j++){
				ds2->RemoveElement();
				ds1->AddElement((-run_beta+run_tau)*(d->z[j-lenl]/(-run_beta)));
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[j-lenl]*tau_minus_beta_pow_factorial[j-lenl]).get_double() * 
				(ds2->divdiffs[q-j]*tau_pow_factorial[q-j]).get_double()/(factorial[lenl]) * 
				(currD_partial[j-lenl]/currD_partial[j]);
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = Akz / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
		for(int j = q-lenk; j >= 0; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
		for(int j = 0; j <= q-lenk; j++){
			ds1->AddElement((-run_beta+run_tau)*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(ds1->divdiffs[j]*tau_minus_beta_pow_factorial[j]).get_double() * 
			(ds2->divdiffs[q-lenk-j]*tau_pow_factorial[q-lenk-j]).get_double()/(factorial[lenk]) * 
			(currD_partial[q-lenk]/currD);
			ds2->RemoveElement();
		}
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			// compute (k, l) Lebniz rule contribution
			std::complex<double> prefac = Akz / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			for(int j = q-lenk; j >= lenl; j--) ds2->AddElement((-run_tau)*(d->z[j]/(-run_beta))); // init ds2
			ds2->AddElement(0);
			for (int j = lenl; j <= q-lenk; j++){
				ds2->RemoveElement();
				ds1->AddElement((-run_beta+run_tau)*(d->z[j-lenl]/(-run_beta)));
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[j-lenl]*tau_minus_beta_pow_factorial[j-lenl]).get_double() * 
				(ds2->divdiffs[q-lenk-j]*tau_pow_factorial[q-lenk-j]).get_double()/(factorial[lenk]*factorial[lenl]) * 
				(currD_partial[j-lenl]/currD) * (currD_partial[q-lenk]/currD_partial[j]);
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_AB_Eint(int n, int m){
	/** 
	* Estimates \int_0^\beta <A(\tau)B> \dtau
	* for A (B) in A.txt (B.txt) the nth (mth) operator 
	* passed here via ./prepare.bin A.txt ... B.txt
	*/
	int i,j,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(m);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_int1)
		prefac = -A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		for(int j = 0; j <= q; j++) {
			d->AddElement(-run_beta*(d->z[j]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(d->divdiffs[q+1]*beta_pow_factorial[q+1]).get_double();
			d->RemoveElement();
		}
		// add (k=0, l) cross term contribution
		prefac = -A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			ds1->CurrentLength=0; // reset scratch divdiffs
			for(int j = q; j >= lenl; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
			ds1->AddElement(0); // add dummy value for continue logic
			for(int j = lenl; j <= q; j++){
				ds1->RemoveElement(); // remove before check so things work with continue
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add \Lambda_j elements to \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[q-lenl+1]*beta_pow_factorial[q-lenl+1]).get_double() * 
				(currD_partial[j-lenl]/currD_partial[j])/(factorial[lenl]);
				// remove \Lambda_j elements from \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->RemoveElement();
				ds1->RemoveElement();
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = -Akz / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		ds1->CurrentLength=0; // reset scratch divdiffs
		for(int j = q-lenk; j >= 0; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
		for(int j = 0; j <= q-lenk; j++){
			// add \Lambda_j elements to \Theta_j multiset
			for(int i = 0; i <= j; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
			T += prefac*currMD0_trace[j]*(ds1->divdiffs[q-lenk+1]*beta_pow_factorial[q-lenk+1]).get_double() * 
			(currD_partial[q-lenk]/currD)/(factorial[lenk]);
			// remove \Lambda_j elements from \Theta_j multiset
			for(int i = 0; i <= j; i++) ds1->RemoveElement();
			ds1->RemoveElement();
		}
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			// compute (k, l) Lebniz rule contribution
			prefac = -Akz / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			ds1->CurrentLength=0; // reset scratch divdiffs
			for(int j = q-lenk; j >= lenl; j--) ds1->AddElement((-run_beta)*(d->z[j]/(-run_beta))); // init ds1
			ds1->AddElement(0); // add dummy value for continue logic
			for (int j = lenl; j <= q-lenk; j++){
				ds1->RemoveElement(); // remove before check so things work with continue
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				// add \Lambda_j elements to \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->AddElement((-run_beta)*(d->z[i]/(-run_beta)));
				// add contribution if relevant
				T += prefac*currMDl_trace[j]*(ds1->divdiffs[q-lenk-lenl+1]*beta_pow_factorial[q-lenk-lenl+1]).get_double() * 
				(currD_partial[j-lenl]/currD) * (currD_partial[q-lenk]/currD_partial[j])/(factorial[lenk]*factorial[lenl]);
				// remove \Lambda_j elements from \Theta_j multiset
				for(int i = 0; i <= j-lenl; i++) ds1->RemoveElement();
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_AB_Fint_slow(int n, int m){
	/**
	* Estimates \int_0^{\beta/2} \tau <A(\tau)B> \dtau
	* for A (B) in A.txt (B.txt) the nth (mth) operator
	* passed here via ./prepare.bin A.txt ... B.txt
	*
	* O(q^4) reference implementation. Kept for regression testing against
	* measure_AB_Fint() (the O(q^3) version, below).
	*/
	int i,j,k,cont,l,lenk,lenl,r;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0, inner_prefac = 0.0;
	double ddratio = 0.0, R = 0.0;
	calc_MD0_trace(m);
	if (A0z != 0.0) {
		// add (k=0, l=0) pure diagonal contribution with Leibniz rule (like Hdiag_int3)
		prefac = -A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		for(int j = 0; j <= q; j++) {
			// reset and init divdiffs
			ds1->CurrentLength=0; ds2->CurrentLength=0;
			for(int i = q; i >= j; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			for(int i = j; i >= 0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// sum over r to accumulate contribution for fixed j
			for(int r = 0; r <= j; r++) {
				ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac * currMD0_trace[j] * (ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				T += (run_beta/2.0)*inner_prefac*(ds2->divdiffs[q+1-r]*beta_div2_pow_factorial[q+1-r]).get_double();
				ds2->AddElement(0);
				T += (inner_prefac*(j-r+1.0)*(ds2->divdiffs[q+2-r]*beta_div2_pow_factorial[q+2-r]).get_double());
				for(int u = r; u <= j; u++){
					ds2->AddElement((-run_beta/2.0)*(d->z[u]/(-run_beta)));
					T += (d->z[u]/(-run_beta))*inner_prefac*(ds2->divdiffs[q+3-r]*beta_div2_pow_factorial[q+3-r]).get_double();
					ds2->RemoveElement();
				}
				ds2->RemoveElement();
				ds2->RemoveElement();
			}
		}
		// add (k=0, l) cross term contribution
		prefac = -A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{P}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenl>q) continue;
			calc_MD_trace_l(m ,l);
			for (int j = lenl; j <= q; j++){
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
				// build e^{-\beta/2 [\Lambda_j]}
				for(int i = j; i <= q; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				// sum over r
				for(int r = 0; r <= j-lenl; r++){
					// build e^{-\beta/2 [\Theta_{j,r}]}
					//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
					ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD_partial[j])*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
					T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenl-r]*beta_div2_pow_factorial[q+1-lenl-r]).get_double();
					// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
					ds2->AddElement(0);
					T += inner_prefac*(j-lenl-r+1.0)*(ds2->divdiffs[q+2-lenl-r]*beta_div2_pow_factorial[q+2-lenl-r]).get_double();
					// add \sum_u contribution
					for(int u = r; u<= j-lenl; u++){
						ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
						T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenl-r]*beta_div2_pow_factorial[q+3-lenl-r]).get_double();
						ds2->RemoveElement();
					}
					ds2->RemoveElement(); // removes AddElement(0)
					ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				}
			}
		}
	}
	// add (k, l=0) and (k, l) contributions
	for(k=0;k<MNop[n];k++){ 
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		// \delta(\tilde{P}_k) conditions
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		// compute (k, l=0) contribution
		prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for (int j = 0; j <= q-lenk; j++){
			ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
			// build e^{-\beta/2 [\Lambda_j]}
			for(int i = j; i <= q-lenk; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
			for(int i = j; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// sum over r
			for(int r = 0; r <= j; r++){
				// build e^{-\beta/2 [\Theta_{j,r}]}
				//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac*currMD0_trace[j]*(currD_partial[q-lenk]/currD)*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
				T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenk-r]*beta_div2_pow_factorial[q+1-lenk-r]).get_double();
				// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
				ds2->AddElement(0);
				T += inner_prefac*(j-r+1.0)*(ds2->divdiffs[q+2-lenk-r]*beta_div2_pow_factorial[q+2-lenk-r]).get_double();
				// add \sum_u contribution
				for(int u = r; u<= j; u++){
					ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
					T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenk-r]*beta_div2_pow_factorial[q+3-lenk-r]).get_double();
					ds2->RemoveElement();
				}
				ds2->RemoveElement(); // removes AddElement(0)
				ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
			}
		}
		// compute \sum_l (k, l) Leibniz rule contribution
		for(l=0;l<MNop[m];l++){
			// initial 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) check
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			for (int j = lenl; j <= q-lenk; j++){
				// remainder of 1^{(j)}_{\tilde{Q}_l}(S_{i_q}) condition
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0; // reset scratch divdiffs
				// build e^{-\beta/2 [\Lambda_j]}
				for(int i = j; i <= q-lenk; i++) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				// build e^{-\beta/2 [\Omega_{j,r}]} (aka add [\overline{\Theta_{j,r}}] to [\Lambda_j]
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				// sum over r
				for(int r = 0; r <= j-lenl; r++){
					// build e^{-\beta/2 [\Theta_{j,r}]}
					//for(int i = 0; i <= r; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
					ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD)*(currD_partial[q-lenk]/currD_partial[j]) *
					(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					// add (run_beta/2)*e^{-run_beta/2[\Omega_{j,r}]} contribution
					T += inner_prefac*(run_beta/2.0)*(ds2->divdiffs[q+1-lenl-lenk-r]*beta_div2_pow_factorial[q+1-lenl-lenk-r]).get_double();
					// add N([\overline{\theta}_{j,r}]) e^{-run_beta/2[\Omega_{j,r}] + {0}} contribution
					ds2->AddElement(0);
					T += inner_prefac*(j-lenl-r+1.0)*(ds2->divdiffs[q+2-lenl-lenk-r]*beta_div2_pow_factorial[q+2-lenl-lenk-r]).get_double();
					// add \sum_u contribution
					for(int u = r; u<= j-lenl; u++){
						ds2->AddElement((-run_beta/2.0)*(d->z[u]/(-run_beta)));
						T += inner_prefac*(d->z[u]/(-run_beta))*(ds2->divdiffs[q+3-lenl-lenk-r]*beta_div2_pow_factorial[q+3-lenl-lenk-r]).get_double();
						ds2->RemoveElement();
					}
					ds2->RemoveElement(); // removes AddElement(0)
					ds2->RemoveElement(); // removes AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				}
			}
		}

	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_AB_Fint(int n, int m){
	/**
	* Estimates \int_0^{\beta/2} \tau <A(\tau)B> \dtau
	* for A (B) in A.txt (B.txt) the nth (mth) operator
	* passed here via ./prepare.bin A.txt ... B.txt
	*
	* O(q^3) implementation (replaces the original O(q^4)
	* measure_AB_Fint_slow, kept above for regression testing).
	*
	* Each of the four (k,l) blocks below has an innermost u-loop that
	* recomputed an append-in-range-node bracket from scratch for every
	* (r,u) pair (the O(q^4) bottleneck). Applying the same sliding-append
	* recursion used in measure_Hdiag_Fint -- here to the more general
	* bracket family that includes an extra confluent node and a dummy
	* zero -- each u-bracket is seeded once (at the smallest r for its j)
	* and advanced in O(1) per split point, instead of recomputed per
	* (r,u) pair. See measure_Hdiag_Fint's own comment for the scaling
	* convention (x_k = E_{z_k} = d->z[k]/(-run_beta), not raw d->z[k]).
	*/
	int i,j,k,cont,l,lenk,lenl,r;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> Akz = 0.0, T = 0.0, prefac = 0.0, inner_prefac = 0.0;
	double ddratio = 0.0, R_ = 0.0;
	calc_MD0_trace(m);
	static ExExFloat Uarr[qmax+1];

	if (A0z != 0.0) {
		// (k=0, l=0) pure diagonal contribution.
		prefac = -A0z / (d->divdiffs[q]*beta_pow_factorial[q]).get_double();
		for(int j = 0; j <= q; j++) {
			ds1->CurrentLength = 0; ds2->CurrentLength = 0;
			for(int i = q; i >= j; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			for(int i = j; i >= 0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// ds2 = N(j,0), q+2 elements (confluent at x_j).

			// Seed Uarr[u] = g(N(j,0) U {0,x_u}) for u=0..j.
			ds2->AddElement(0);
			for(int u = 0; u <= j; u++){
				ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
				Uarr[u] = ds2->divdiffs[q+3]*beta_div2_pow_factorial[q+3];
				ds2->RemoveElement();
			}
			ds2->RemoveElement(); // back to N(j,0)

			for(int r = 0; r <= j; r++){
				ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac * currMD0_trace[j] * (ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				T += (run_beta/2.0)*inner_prefac*(ds2->divdiffs[q+1-r]*beta_div2_pow_factorial[q+1-r]).get_double();
				ds2->AddElement(0);
				ExExFloat suffix2_r = ds2->divdiffs[q+2-r]*beta_div2_pow_factorial[q+2-r];
				T += (inner_prefac*(j-r+1.0)*suffix2_r.get_double());
				for(int u = r; u <= j; u++) T += (d->z[u]/(-run_beta))*inner_prefac*Uarr[u].get_double();
				ds2->RemoveElement(); // undo the {0}
				if (r < j){
					for(int u = r+1; u <= j; u++)
						Uarr[u] = Uarr[u]*((d->z[r]-d->z[u])/run_beta) + suffix2_r;
					ds2->RemoveElement(); // drop x_r
				}
			}
		}
		// (k=0, l) cross term contribution.
		prefac = -A0z / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for(l=0;l<MNop[m];l++){
			Ql = MP[m][l]; lenl = Ql.count(); if(lenl>q) continue;
			calc_MD_trace_l(m ,l);
			for (int j = lenl; j <= q; j++){
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0;
				for(int i = j; i <= q; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
				int Rmax = j - lenl;
				// ds2 = N'(j,0), q-lenl+2 elements (no confluence, gap between Λ_j and Ω).

				ds2->AddElement(0);
				for(int u = 0; u <= Rmax; u++){
					ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
					Uarr[u] = ds2->divdiffs[q+3-lenl]*beta_div2_pow_factorial[q+3-lenl];
					ds2->RemoveElement();
				}
				ds2->RemoveElement();

				for(int r = 0; r <= Rmax; r++){
					ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD_partial[j])*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenl-r]*beta_div2_pow_factorial[q+1-lenl-r]).get_double();
					ds2->AddElement(0);
					ExExFloat suffix2_r = ds2->divdiffs[q+2-lenl-r]*beta_div2_pow_factorial[q+2-lenl-r];
					T += inner_prefac*(Rmax-r+1.0)*suffix2_r.get_double();
					for(int u = r; u <= Rmax; u++) T += inner_prefac*(d->z[u]/(-run_beta))*Uarr[u].get_double();
					ds2->RemoveElement();
					if (r < Rmax){
						for(int u = r+1; u <= Rmax; u++)
							Uarr[u] = Uarr[u]*((d->z[r]-d->z[u])/run_beta) + suffix2_r;
						ds2->RemoveElement();
					}
				}
			}
		}
	}
	// (k, l=0) and (k, l) contributions.
	for(k=0;k<MNop[n];k++){
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue;
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		int Qtop = q - lenk;
		prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
		for (int j = 0; j <= Qtop; j++){
			ds1->CurrentLength=0; ds2->CurrentLength=0;
			for(int i = j; i <= Qtop; i++) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			for(int i = j; i>=0; i--) ds2->AddElement((-run_beta/2)*(d->z[i]/(-run_beta)));
			// ds2 = N''(j,0), Qtop+2 elements (confluent at x_j).

			ds2->AddElement(0);
			for(int u = 0; u <= j; u++){
				ds2->AddElement((-run_beta/2)*(d->z[u]/(-run_beta)));
				Uarr[u] = ds2->divdiffs[q+3-lenk]*beta_div2_pow_factorial[q+3-lenk];
				ds2->RemoveElement();
			}
			ds2->RemoveElement();

			for(int r = 0; r <= j; r++){
				ds1->AddElement((-run_beta/2)*(d->z[r]/(-run_beta)));
				inner_prefac = prefac*currMD0_trace[j]*(currD_partial[Qtop]/currD)*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
				T += inner_prefac*(run_beta/2)*(ds2->divdiffs[q+1-lenk-r]*beta_div2_pow_factorial[q+1-lenk-r]).get_double();
				ds2->AddElement(0);
				ExExFloat suffix2_r = ds2->divdiffs[q+2-lenk-r]*beta_div2_pow_factorial[q+2-lenk-r];
				T += inner_prefac*(j-r+1.0)*suffix2_r.get_double();
				for(int u = r; u <= j; u++) T += inner_prefac*(d->z[u]/(-run_beta))*Uarr[u].get_double();
				ds2->RemoveElement();
				if (r < j){
					for(int u = r+1; u <= j; u++)
						Uarr[u] = Uarr[u]*((d->z[r]-d->z[u])/run_beta) + suffix2_r;
					ds2->RemoveElement();
				}
			}
		}
		// (k, l) Leibniz rule contribution.
		for(l=0;l<MNop[m];l++){
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue;
			calc_MD_trace_l(m ,l);
			prefac = -(Akz / factorial[lenk]) / ((d->divdiffs[q]*beta_pow_factorial[q]).get_double());
			for (int j = lenl; j <= Qtop; j++){
				if(!NoRepetitionCheck(Sq+(j-lenl),lenl)) continue;
				cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[j-lenl+i])){ cont = 1; break;} if(cont) continue;
				ds1->CurrentLength=0; ds2->CurrentLength=0;
				for(int i = j; i <= Qtop; i++) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				for(int i = j-lenl; i>=0; i--) ds2->AddElement((-run_beta/2.0)*(d->z[i]/(-run_beta)));
				int Rmax = j - lenl;
				// ds2 = N'''(j,0), Qtop-lenl-... wait: (Qtop-j+1)+(j-lenl+1) = Qtop-lenl+2 elements, no confluence.

				ds2->AddElement(0);
				for(int u = 0; u <= Rmax; u++){
					ds2->AddElement((-run_beta/2.0)*(d->z[u]/(-run_beta)));
					Uarr[u] = ds2->divdiffs[q+3-lenl-lenk]*beta_div2_pow_factorial[q+3-lenl-lenk];
					ds2->RemoveElement();
				}
				ds2->RemoveElement();

				for(int r = 0; r <= Rmax; r++){
					ds1->AddElement((-run_beta/2.0)*(d->z[r]/(-run_beta)));
					inner_prefac = prefac*(currMDl_trace[j]/factorial[lenl])*(currD_partial[j-lenl]/currD)*(currD_partial[Qtop]/currD_partial[j])*(ds1->divdiffs[r]*beta_div2_pow_factorial[r]).get_double();
					T += inner_prefac*(run_beta/2.0)*(ds2->divdiffs[q+1-lenl-lenk-r]*beta_div2_pow_factorial[q+1-lenl-lenk-r]).get_double();
					ds2->AddElement(0);
					ExExFloat suffix2_r = ds2->divdiffs[q+2-lenl-lenk-r]*beta_div2_pow_factorial[q+2-lenl-lenk-r];
					T += inner_prefac*(Rmax-r+1.0)*suffix2_r.get_double();
					for(int u = r; u <= Rmax; u++) T += inner_prefac*(d->z[u]/(-run_beta))*Uarr[u].get_double();
					ds2->RemoveElement();
					if (r < Rmax){
						for(int u = r+1; u <= Rmax; u++)
							Uarr[u] = Uarr[u]*((d->z[r]-d->z[u])/run_beta) + suffix2_r;
						ds2->RemoveElement();
					}
				}
			}
		}
	}
	#ifdef ABS_WEIGHTS
		R_ = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R_ = std::real(currD*T)/std::real(currD);
	#endif
	return R_;
}

double measure_AB_real(int n, int m){
	/** 
	* Estimates <AB> for A (B) in A.txt (B.txt) the nth (mth) operator 
	* passed here via ./prepare.bin A.txt ... B.txt
	*/
	int i,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> B0z = calc_MD0(m);
	std::complex<double> Akz = 0.0, Bkz = 0.0, T = 0.0;
	double ddratio = 0.0, R = 0.0;
	T += A0z*B0z; // add diag-diag (0, 0) contribution
	// compute <A0 \sum_l Bl Ql> contribution
	if (A0z != 0.0) {
		for(l=0;l<MNop[m];l++){
			calc_MD_trace_l(m ,l);
			if(currMDl_trace[q]==0.0) continue;
			Ql = MP[m][l]; lenl = Ql.count(); if(lenl>q) continue; // need lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenl),lenl)) continue; // check {Sq[q-lenl], ..., Sq[q-1]} no repeats
			// checks that Sq[q-1-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenl]*beta_pow_factorial[q-lenl]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
			T += ddratio * (A0z * currMDl_trace[q] * (currD_partial[q-lenl]/currD) / factorial[lenl]);
		}
	}
	// compute <\sum_k A_k P_k B0> contribution
	if (B0z != 0.0) {
		for(k=0;k<MNop[n];k++){ 
			// compute (k, l=0) contribution
			Akz = calc_MD(n,k);
			if(Akz==0.0) continue;
			Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
			if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
			cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenk]*beta_pow_factorial[q-lenk]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
			T += ddratio * (B0z * Akz * (currD_partial[q-lenk]/currD) / factorial[lenk]);
		}
	}
	// compute <\sum_k A_k P_k \sum_l B_l Q_l> contributions
	for(k=0;k<MNop[n];k++){ 
		// compute (k, l=0) contribution
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		for(l=0;l<MNop[m];l++){
			calc_MD_trace_l(m ,l);
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue; // need lenk + lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenk-lenl),lenl)) continue; // check {Sq[q-lenk-lenl], ..., Sq[q-1-lenk]} no repeats
			// checks that Sq[q-1-lenk-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[q-1-lenk-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenk-lenl]*beta_pow_factorial[q-lenk-lenl]) / (d->divdiffs[q] * beta_pow_factorial[q])).get_double();
			T += ddratio * ((Akz * currMDl_trace[q-lenk] * (currD_partial[q-lenk-lenl]/currD)) / (factorial[lenk]*factorial[lenl]));
		}
	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
		//R = std::real(im*currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}

double measure_AB_imag(int n, int m){
	/** 
	* Estimates <AB> for A (B) in A.txt (B.txt) the nth (mth) operator 
	* passed here via ./prepare.bin A.txt ... B.txt
	*/
	int i,k,cont,l,lenk,lenl;
	std::complex<double> A0z = calc_MD0(n);
	std::complex<double> B0z = calc_MD0(m);
	std::complex<double> Akz = 0.0, Bkz = 0.0, T = 0.0;
	double ddratio = 0.0, R = 0.0;
	T += A0z*B0z; // add diag-diag (0, 0) contribution
	// compute <A0 \sum_l Bl Ql> contribution
	if (A0z != 0.0) {
		for(l=0;l<MNop[m];l++){
			calc_MD_trace_l(m ,l);
			if(currMDl_trace[q]==0.0) continue;
			Ql = MP[m][l]; lenl = Ql.count(); if(lenl>q) continue; // need lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenl),lenl)) continue; // check {Sq[q-lenl], ..., Sq[q-1]} no repeats
			// checks that Sq[q-1-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenl]*beta_pow_factorial[q-lenl]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
			T += ddratio * (A0z * currMDl_trace[q] * (currD_partial[q-lenl]/currD) / factorial[lenl]);
		}
	}
	// compute <\sum_k A_k P_k B0> contribution
	if (B0z != 0.0) {
		for(k=0;k<MNop[n];k++){ 
			// compute (k, l=0) contribution
			Akz = calc_MD(n,k);
			if(Akz==0.0) continue;
			Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
			if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
			cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenk]*beta_pow_factorial[q-lenk]) / (d->divdiffs[q]*beta_pow_factorial[q])).get_double();
			T += ddratio * (B0z * Akz * (currD_partial[q-lenk]/currD) / factorial[lenk]);
		}
	}
	// compute <\sum_k A_k P_k \sum_l B_l Q_l> contributions
	for(k=0;k<MNop[n];k++){ 
		// compute (k, l=0) contribution
		Akz = calc_MD(n,k);
		if(Akz==0.0) continue;
		Pk = MP[n][k]; lenk = Pk.count(); if(lenk>q) continue; 
		if(!NoRepetitionCheck(Sq+(q-lenk),lenk)) continue;
		cont = 0; for(i=0;i<lenk;i++) if(!Pk.test(Sq[q-1-i])){ cont = 1; break;} if(cont) continue;
		for(l=0;l<MNop[m];l++){
			calc_MD_trace_l(m ,l);
			Ql = MP[m][l]; lenl = Ql.count(); if(lenk+lenl>q) continue; // need lenk + lenl > q
			if(!NoRepetitionCheck(Sq+(q-lenk-lenl),lenl)) continue; // check {Sq[q-lenk-lenl], ..., Sq[q-1-lenk]} no repeats
			// checks that Sq[q-1-lenk-i] \in P_l for i=0, i=lenl-1.
			cont = 0; for(i=0;i<lenl;i++) if(!Ql.test(Sq[q-1-lenk-i])){ cont = 1; break;} if(cont) continue;
			ddratio = ((d->divdiffs[q-lenk-lenl]*beta_pow_factorial[q-lenk-lenl]) / (d->divdiffs[q] * beta_pow_factorial[q])).get_double();
			T += ddratio * ((Akz * currMDl_trace[q-lenk] * (currD_partial[q-lenk-lenl]/currD)) / (factorial[lenk]*factorial[lenl]));
		}
	}
	#ifdef ABS_WEIGHTS
		R = std::abs(T)*currWeight.sgn()*cos(std::arg(T)+std::arg(currD));
	#else
		//R = std::real(currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
		R = std::real(im*currD*T)/std::real(currD); // we importance-sample Re(W_C A_C)/Re(W_C)
	#endif
	return R;
}
#endif

double measure_observable(int n){
	double R = 0;
	if(valid_observable[n]) if(n < Nobservables){
#ifdef MEASURE_CUSTOM_OBSERVABLES
		/** 
		* Estimates things like <O>, <O^2>, and so on where O is in O.txt
		* 
		* Use ./prepare.bin H.txt $(ls O.txt 2> /dev/null) to measure only case 0
		* ./prepare.bin H.txt $(ls O.txt O.txt 2> /dev/null) to measure case 0 and 1
		* and so on...
		*/
		int i,k,len,cont,l,lenk,lenl,j,t,r;
		std::complex<ExExFloat> eef_cont(0.0), dd_ratio; //the exexfloat contributions
		std::complex<double> A0z = calc_MD0(n);
		std::complex<double> Akz = 0.0, T = 0.0;
		switch(n){
			case 0: R = measure_O(0); break;
			case 1: R = measure_O2(0); break;
			case 2: R = measure_O_corr(0); break;
			case 3: R = measure_O_Eint(0); break;
			case 4: R = measure_O_Fint(0); break;
			case 5: R = measure_O(5); break;
			case 6: R = measure_O2(5); break;
			case 7: R = measure_O_corr(5); break;
			case 8: R = measure_O_Eint(5); break;
			case 9: R = measure_O_Fint(5); break;
			case 10: R = measure_AB_corr(0, 5); break; 
			case 11: R = measure_AB_Eint(0, 5); break; 
			case 12: R = measure_AB_Fint(0, 5); break;
		        case 13: R = (measure_AB_real(0, 5) + measure_AB_real(5, 0)) / 2.0; break;
		        case 14: R = (measure_AB_imag(0, 5) - measure_AB_imag(5, 0)) / 2.0; break;
		}
#endif
	} else  switch(n-Nobservables){
			case 0:	R = measure_H(); break;
			case 1:	R = measure_H2(); break;
			case 2:	R = measure_Hdiag(); break;
			case 3:	R = measure_Hdiag2(); break;
			case 4:	R = measure_Hoffdiag(); break;
			case 5:	R = measure_Hoffdiag2(); break;
			case 6: R = measure_Z_magnetization(); break;
			case 7: R = measure_Hdiag_corr(); break;
            case 8: R = measure_Hdiag_Eint(); break;
			case 9: R = measure_Hdiag_Fint(); break;
			case 10: R = measure_Hoffdiag_corr(); break; 
			case 11: R = measure_Hoffdiag_Eint(); break;
			case 12: R = measure_Hoffdiag_Fint(); break;
			case 13: R = measure_parity(); break; 
	}
	return R;
}

void measure(){
	double R, sgn; int i;
	currWeight = GetWeight();
#ifdef ABS_WEIGHTS
	sgn = currWeight.sgn() * cos(std::arg(currD)); // arg(W) = arg(currD) + arg(currWeight), where arg(currWeight) = either 0 or Pi
#else
	sgn = currWeight.sgn();
#endif
	last_measurement_sgn = sgn;
	meanq += q; if(maxq < q) maxq = q; in_bin_sum_sgn += sgn;
	if((measurement_step+1) % bin_length == 0){
		in_bin_sum_sgn /= bin_length; bin_mean_sgn[measurement_step/bin_length] = in_bin_sum_sgn; in_bin_sum_sgn = 0;
	}
	for(i=0;i<N_all_observables;i++){
		R = measure_observable(i); last_measurement[i] = R; in_bin_sum[i] += R*sgn;
		if((measurement_step+1) % bin_length == 0){
			in_bin_sum[i] /= bin_length; bin_mean[i][measurement_step/bin_length] = in_bin_sum[i]; in_bin_sum[i] = 0;
		}
	}
}

double mean_O[N_all_observables], stdev_O[N_all_observables], mean_O_backup[N_all_observables];

const int N_derived_observables = 5;  // we define number of derived observables

std::string name_of_derived_observable(int n){ // we define names of derived observables
	std::string s;
	switch(n){
		case 0 : s = "diagonal energy susceptibility"; break;
		case 1 : s = "diagonal fidelity susceptibility"; break;
		case 2 : s = "offdiagonal energy susceptibility"; break;
		case 3 : s = "offdiagonal fidelity susceptibility"; break;
		case 4 : s = "specific heat"; break;
	}
	return s;
}

int valid_derived_observable(int n){ // we define which observables are needed for each derived observable
	int r = 0;
	switch(n){
		case 0: r = valid_observable[Nobservables+2] && valid_observable[Nobservables+8]; break;
		case 1: r = valid_observable[Nobservables+2] && valid_observable[Nobservables+9]; break;
		case 2: r = valid_observable[Nobservables+4] && valid_observable[Nobservables+11]; break;
		case 3: r = valid_observable[Nobservables+4] && valid_observable[Nobservables+12]; break;
		case 4: r = valid_observable[Nobservables] && valid_observable[Nobservables+1]; break; 
	}
	return r;
}

double compute_derived_observable(int n){ // we compute the derived observables
	double R = 0;
	switch(n){
		case 0 : R = mean_O[Nobservables+8] - run_beta*mean_O[Nobservables+2]*mean_O[Nobservables+2]; break;
		case 1 : R = mean_O[Nobservables+9] - (run_beta*mean_O[Nobservables+2])*(run_beta*mean_O[Nobservables+2])/8; break;
		case 2 : R = mean_O[Nobservables+11] - run_beta*mean_O[Nobservables+4]*mean_O[Nobservables+4]; break;
		case 3 : R = mean_O[Nobservables+12] - (run_beta*mean_O[Nobservables+4])*(run_beta*mean_O[Nobservables+4])/8; break;
		case 4 : R = run_beta*run_beta*(mean_O[Nobservables+1] -  (mean_O[Nobservables])*(mean_O[Nobservables])); break;
	}
	return R;
}
