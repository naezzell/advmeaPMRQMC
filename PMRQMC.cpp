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

#include"mainqmc.hpp"
#include"beta_anneal.hpp"

double get_cpu_time(){ return (double)clock() / CLOCKS_PER_SEC;}
void signalHandler(int signum){	if(save_data_flag==0) save_data_flag = 1; }

struct FixedRunOptions {
	BetaAnnealOptions anneal;
	double target_beta = run_beta, target_tau = run_tau;
	bool target_beta_set = false, target_tau_set = false;
};

static bool parse_fixed_options(int argc, char** argv, FixedRunOptions& options){
	bool rank_set = false;
	for(int i=1;i<argc;i++){
		std::string arg(argv[i]);
		if(arg == "--target-beta" && i+1<argc){ options.target_beta=std::stod(argv[++i]); options.target_beta_set=true; }
		else if(arg == "--target-tau" && i+1<argc){ options.target_tau=std::stod(argv[++i]); options.target_tau_set=true; }
		else if(arg == "--beta-anneal") options.anneal.automatic=true;
		else if(arg == "--anneal-start-factor" && i+1<argc){ options.anneal.start_factor=std::stod(argv[++i]); options.anneal.start_factor_was_set=true; }
		else if(arg == "--anneal-interval" && i+1<argc){ options.anneal.interval=std::stoull(argv[++i]); options.anneal.interval_was_set=true; }
		else if(arg == "--beta-anneal-schedule" && i+1<argc) options.anneal.schedule_file=argv[++i];
		else if(!rank_set && !arg.empty() && arg[0]!='-'){ mpi_rank=std::stoi(arg); mpi_size=mpi_rank+1; rank_set=true; }
		else return false;
	}
	if(options.target_beta_set && !options.target_tau_set) options.target_tau=options.target_beta/2.0;
	return std::isfinite(options.target_beta) && options.target_beta>0.0 &&
		std::isfinite(options.target_tau) && options.target_tau>=0.0 && options.target_tau<=options.target_beta;
}

static uint64_t fixed_run_identity(const BetaAnnealPlan& plan, double beta_value, double tau_value){
	uint64_t result=beta_anneal_hash(plan), bits=0;
	std::memcpy(&bits,&beta_value,sizeof(bits)); result ^= bits+UINT64_C(0x9e3779b97f4a7c15)+(result<<6)+(result>>2);
	std::memcpy(&bits,&tau_value,sizeof(bits)); result ^= bits+UINT64_C(0x9e3779b97f4a7c15)+(result<<6)+(result>>2);
	return result;
}

int main(int argc, char* argv[]){
#ifdef SAVE_UNFINISHED_CALCULATION
	signal(SIGTERM,signalHandler);
#endif
	if(steps < Nbins*stepsPerMeasurement){
		std::cout << "Error: steps cannot be smaller than Nbins*stepsPerMeasurement." << std::endl;
		exit(1);
	}
	if(N == 0){
		std::cout << "Error: no particles found. At least one particle must be described by the Hamiltonian." << std::endl;
		exit(1);
	}
	start_time = get_cpu_time(); int i,j,k,o=0;
	FixedRunOptions options;
	try{
		if(!parse_fixed_options(argc,argv,options)) throw std::runtime_error("unknown or incomplete command-line option");
	}catch(const std::exception& error){ std::cerr << "Error: " << error.what() << std::endl; return 2; }
	if(mpi_size==0) if(const char* task_id_env = std::getenv("SLURM_ARRAY_TASK_ID")){
		mpi_rank = std::stoi(task_id_env); mpi_size = mpi_rank + 1;
	}
	BetaAnnealPlan anneal;
	try{ anneal=make_beta_anneal_plan(options.anneal,std::vector<double>(1,options.target_beta),Tsteps,N); }
	catch(const std::exception& error){ std::cerr << "Error: " << error.what() << std::endl; return 2; }
	dynamic_run_parameters = options.target_beta_set || options.target_tau_set || anneal.enabled;
	dynamic_run_identity = fixed_run_identity(anneal,options.target_beta,options.target_tau);
	divdiff_init(); divdiff dd(q+4,500); divdiff ddfs(q+4,500); divdiff dd1(q+4,500); divdiff dd2(q+4,500);
	d=&dd; dfs=&ddfs; ds1=&dd1; ds2=&dd2;
	if(dynamic_run_parameters){
		double initial_beta=anneal.enabled ? anneal.beta_at(0,0) : options.target_beta;
		configure_run_parameters(initial_beta,beta_anneal_tau(options.target_tau,initial_beta,options.target_beta));
	}
	init_rng();
	if(check_QMC_data()){
		load_QMC_data(); init_basic();
	} else{
		init();	std::cout << "RNG seed = " << rng_seed << std::endl;
	}
	std::cout << "Parameters: beta = " << run_beta << ", Tsteps = " << Tsteps << ", steps = " << steps << std::endl;
	if(TstepsFinished){
		if(step>0 && step<stepsPerMeasurement && measurement_step<measurements){
			for(;step<stepsPerMeasurement;step++) update(); measure(); measurement_step++;
		}
	} else{
		if(anneal.enabled){
			for(;step<Tsteps;step++){
				update(); uint64_t completed=static_cast<uint64_t>(step)+1;
				if(anneal.retarget_after(completed)){
					double next_beta=anneal.beta_at(completed,0);
					retarget_run_parameters(next_beta,beta_anneal_tau(options.target_tau,next_beta,options.target_beta));
				}
			}
			retarget_run_parameters(options.target_beta,options.target_tau);
		}else for(;step<Tsteps;step++) update();
		TstepsFinished = 1;
	}
	for(;measurement_step<measurements;measurement_step++){
		for(step=0;step<stepsPerMeasurement;step++) update(); measure();
	}
#ifdef SAVE_COMPLETED_CALCULATION
	save_QMC_data(0);
#endif
	double Rsum[N_all_observables] = {0}; double sgn_mean = 0;
	double over_bins_sum[N_all_observables] = {0}; double sgn_variance = 0;
	double over_bins_sum_cov[N_all_observables] = {0};
	for(i=0;i<Nbins;i++) sgn_mean += bin_mean_sgn[i]; sgn_mean /= Nbins;
	for(i=0;i<Nbins;i++) sgn_variance += (bin_mean_sgn[i] - sgn_mean)*(bin_mean_sgn[i] - sgn_mean); sgn_variance /= (Nbins*(Nbins-1));
	std::cout << std::setprecision(9);
	std::cout << "mean(sgn(W)) = " << sgn_mean << std::endl;
	std::cout << "std.dev.(sgn(W)) = " << sqrt(sgn_variance) << std::endl;
	if(qmax_achieved) std::cout << std::endl << "Warning: qmax = " << qmax << " was achieved. The results may be incorrect. The qmax parameter should be increased." << std::endl;
	for(i=0;i<Ncycles;i++) if(!cycles_used[i]) std::cout << "Warning: cycle No. " << i << " of length " << cycle_len[i] << " was not used" << std::endl;
	std::cout << "mean(q) = " << meanq / measurements << std::endl;
	std::cout << "max(q) = "<< maxq << std::endl;
	for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
		std::cout << "Observable #" << ++o << ": "<< name_of_observable(k) << std::endl;
		for(i=0;i<Nbins;i++) Rsum[k] += bin_mean[k][i]; Rsum[k] /= Nbins;
		for(i=0;i<Nbins;i++) over_bins_sum[k] += (bin_mean[k][i] - Rsum[k])*(bin_mean[k][i] - Rsum[k]); over_bins_sum[k] /= (Nbins*(Nbins-1));
		for(i=0;i<Nbins;i++) over_bins_sum_cov[k] += (bin_mean[k][i] - Rsum[k])*(bin_mean_sgn[i] - sgn_mean); over_bins_sum_cov[k] /= (Nbins*(Nbins-1));
		mean_O[k] = Rsum[k]/sgn_mean*(1 + sgn_variance/sgn_mean/sgn_mean) - over_bins_sum_cov[k]/sgn_mean/sgn_mean;
		stdev_O[k] = fabs(Rsum[k]/sgn_mean)*sqrt(over_bins_sum[k]/Rsum[k]/Rsum[k] + sgn_variance/sgn_mean/sgn_mean - 2*over_bins_sum_cov[k]/Rsum[k]/sgn_mean);
		std::cout << "mean(O) = " << mean_O[k] << std::endl;
		std::cout << "std.dev.(O) = " << stdev_O[k] << std::endl;
	}
	double mean_derived_O[N_derived_observables], stdev_derived_O[N_derived_observables], jackknife_O[N_derived_observables], jackknife_sum[N_derived_observables], sgn_meanJ, sgn_varianceJ;
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
		mean_derived_O[o] = compute_derived_observable(o); jackknife_sum[o] = 0;
	}
	for(j=0;j<Nbins;j++){
		sgn_meanJ = sgn_varianceJ = 0;
		for(i=0;i<Nbins;i++) if(i!=j) sgn_meanJ += bin_mean_sgn[i]; sgn_meanJ /= (Nbins-1);
		for(i=0;i<Nbins;i++) if(i!=j) sgn_varianceJ += (bin_mean_sgn[i] - sgn_meanJ)*(bin_mean_sgn[i] - sgn_meanJ); sgn_varianceJ /= ((Nbins-1)*(Nbins-2));
		for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
			Rsum[k] = over_bins_sum_cov[k] = 0;
			for(i=0;i<Nbins;i++) if(i!=j) Rsum[k] += bin_mean[k][i]; Rsum[k] /= (Nbins-1);
			for(i=0;i<Nbins;i++) if(i!=j) over_bins_sum_cov[k] += (bin_mean[k][i] - Rsum[k])*(bin_mean_sgn[i] - sgn_meanJ); over_bins_sum_cov[k] /= ((Nbins-1)*(Nbins-2));
			mean_O[k] = Rsum[k]/sgn_meanJ*(1 + sgn_varianceJ/sgn_meanJ/sgn_meanJ) - over_bins_sum_cov[k]/sgn_meanJ/sgn_meanJ;
		}
		for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
			jackknife_O[o] = compute_derived_observable(o);
			jackknife_sum[o] += (jackknife_O[o] - mean_derived_O[o])*(jackknife_O[o] - mean_derived_O[o]);
		}
	}
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)) stdev_derived_O[o] = sqrt(jackknife_sum[o]*(Nbins-1)/Nbins);
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
		std::cout << "Derived observable: " << name_of_derived_observable(o) << std::endl;
		std::cout << "mean(O) = " << mean_derived_O[o] << std::endl;
		std::cout << "std.dev.(O) = " << stdev_derived_O[o] << std::endl;
	}
	divdiff_clear_up();
	std::cout << "wall-clock cpu time = " << get_cpu_time()-start_time << " seconds" << std::endl;
	return 0;
}
