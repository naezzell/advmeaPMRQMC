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

#include<mpi.h>
#include"mainqmc.hpp"
#include"beta_anneal.hpp"

double elapsed_time;
double mean_derived_O[N_derived_observables], stdev_derived_O[N_derived_observables], jackknife_O[N_derived_observables], jackknife_sum[N_derived_observables], sgn_meanJ, sgn_varianceJ;
double sgn_mean, sgn_variance, sgn_stdev;
std::ofstream timeseries_file;
bool timeseries_enabled = false;
BetaAnnealPlan fixed_anneal_plan;
double fixed_target_beta = run_beta, fixed_target_tau = run_tau;
std::ofstream stream_timeseries_file;
bool stream_timeseries_enabled = false;

static void write_timeseries_header(){
	if(mpi_rank != 0) return;
	timeseries_file << "measurement,updates,beta,tau,sign";
	for(int k=0;k<N_all_observables;k++)
		timeseries_file << ",obs_" << k << ",signed_obs_" << k;
	timeseries_file << ",elapsed_seconds\n";
}

static void write_timeseries_row(unsigned long long measurement, unsigned long long updates,
		const double* observables, const double* signed_observables, double sign,
		double elapsed_seconds){
	if(mpi_rank != 0) return;
	timeseries_file << measurement << ',' << updates << ',' << std::setprecision(17)
		<< run_beta << ',' << run_tau << ',' << sign;
	for(int k=0;k<N_all_observables;k++)
		timeseries_file << ',' << observables[k] << ',' << signed_observables[k];
	timeseries_file << ',' << elapsed_seconds << '\n';
}

static std::string stream_timeseries_name(const std::string& prefix, int rank){
	return prefix + ".rank" + std::to_string(rank) + ".csv";
}

static bool file_has_content(const std::string& path){
	std::ifstream input(path.c_str(),std::ios::binary|std::ios::ate);
	return input.good() && input.tellg() > 0;
}

static void write_stream_timeseries_header(){
	stream_timeseries_file << "stream,rank,measurement,updates,beta,tau,global_z2_moves,sign";
	for(int k=0;k<N_all_observables;k++)
		stream_timeseries_file << ",obs_" << k << ",signed_obs_" << k;
	stream_timeseries_file << ",elapsed_seconds\n";
}

static void write_stream_timeseries_row(unsigned long long measurement, unsigned long long updates,
		double elapsed_seconds){
	stream_timeseries_file << mpi_rank << ',' << mpi_rank << ',' << measurement << ',' << updates
		<< ',' << std::setprecision(17) << run_beta << ',' << run_tau << ',' << global_z2_moves
		<< ',' << last_measurement_sgn;
	for(int k=0;k<N_all_observables;k++)
		stream_timeseries_file << ',' << last_measurement[k] << ',' << last_measurement[k]*last_measurement_sgn;
	stream_timeseries_file << ',' << elapsed_seconds << '\n';
}

void signalHandler(int signum){	if(save_data_flag==0) save_data_flag = 1; }

void compute(){
	if(!resume_calc) std::cout << "Starting calculation for MPI process No. " << mpi_rank << ", RNG seed = " << rng_seed << std::endl; fflush(stdout);
	if(TstepsFinished){
		if(step>0 && step<stepsPerMeasurement && measurement_step<measurements){
			for(;step<stepsPerMeasurement;step++) update(); measure(); measurement_step++;
		}
	} else{
		if(fixed_anneal_plan.enabled){
			for(;step<Tsteps;step++){
				update(); uint64_t completed=static_cast<uint64_t>(step)+1;
				if(fixed_anneal_plan.retarget_after(completed)){
					double next_beta=fixed_anneal_plan.beta_at(completed,0);
					retarget_run_parameters(next_beta,beta_anneal_tau(fixed_target_tau,next_beta,fixed_target_beta));
				}
			}
			retarget_run_parameters(fixed_target_beta,fixed_target_tau);
		}else for(;step<Tsteps;step++) update();
		TstepsFinished = 1;
	}
	for(;measurement_step<measurements;measurement_step++){
		for(step=0;step<stepsPerMeasurement;step++) update(); measure();
		if(stream_timeseries_enabled)
			write_stream_timeseries_row(measurement_step+1,
				static_cast<unsigned long long>(Tsteps) + (measurement_step+1)*stepsPerMeasurement,
				MPI_Wtime()-start_time);
		if(timeseries_enabled){
			double local_signed[N_all_observables], summed[N_all_observables];
			double summed_signed[N_all_observables], summed_sign;
			for(int k=0;k<N_all_observables;k++)
				local_signed[k] = last_measurement[k]*last_measurement_sgn;
			MPI_Reduce(last_measurement,summed,N_all_observables,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Reduce(local_signed,summed_signed,N_all_observables,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Reduce(&last_measurement_sgn,&summed_sign,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			if(mpi_rank==0){
				for(int k=0;k<N_all_observables;k++){
					summed[k] /= mpi_size;
					summed_signed[k] /= mpi_size;
				}
				write_timeseries_row(measurement_step+1,
					static_cast<unsigned long long>(Tsteps) + (measurement_step+1)*stepsPerMeasurement,
					summed,summed_signed,summed_sign/mpi_size,MPI_Wtime()-start_time);
			}
		}
	}
#ifdef SAVE_COMPLETED_CALCULATION
	save_QMC_data(0);
#endif
	meanq /= measurements;
	elapsed_time = MPI_Wtime()-start_time;
	std::cout << "Calculation completed for MPI process No. " << mpi_rank
	          << ", elapsed time = " << elapsed_time << " seconds" << std::endl; fflush(stdout);
}

void process_single_run(){
	double Rsum[N_all_observables] = {0}; sgn_mean = 0; int i,j,k,o;
	double over_bins_sum[N_all_observables] = {0}; sgn_variance = 0;
	double over_bins_sum_cov[N_all_observables] = {0};
	for(i=0;i<Nbins;i++) sgn_mean += bin_mean_sgn[i]; sgn_mean /= Nbins;
	for(i=0;i<Nbins;i++) sgn_variance += (bin_mean_sgn[i] - sgn_mean)*(bin_mean_sgn[i] - sgn_mean); sgn_variance /= (Nbins*(Nbins-1));
	for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
		for(i=0;i<Nbins;i++) Rsum[k] += bin_mean[k][i]; Rsum[k] /= Nbins;
		for(i=0;i<Nbins;i++) over_bins_sum[k] += (bin_mean[k][i] - Rsum[k])*(bin_mean[k][i] - Rsum[k]); over_bins_sum[k] /= (Nbins*(Nbins-1));
		for(i=0;i<Nbins;i++) over_bins_sum_cov[k] += (bin_mean[k][i] - Rsum[k])*(bin_mean_sgn[i] - sgn_mean); over_bins_sum_cov[k] /= (Nbins*(Nbins-1));
		mean_O_backup[k] = mean_O[k] = Rsum[k]/sgn_mean*(1 + sgn_variance/sgn_mean/sgn_mean) - over_bins_sum_cov[k]/sgn_mean/sgn_mean;
		stdev_O[k] = fabs(Rsum[k]/sgn_mean)*sqrt(over_bins_sum[k]/Rsum[k]/Rsum[k] + sgn_variance/sgn_mean/sgn_mean - 2*over_bins_sum_cov[k]/Rsum[k]/sgn_mean);
	}
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
	for(k=0;k<N_all_observables;k++) if(valid_observable[k]) mean_O[k] = mean_O_backup[k];
}

void printout_single_run(){
	std::cout << std::setprecision(9); int i,k,o=0;
	std::cout << "mean(sgn(W)) = " << sgn_mean << std::endl;
	std::cout << "std.dev.(sgn(W)) = " << sqrt(sgn_variance) << std::endl;
	if(qmax_achieved) std::cout << std::endl << "Warning: qmax = " << qmax << " was achieved. The results may be incorrect. The qmax parameter should be increased." << std::endl;
	for(i=0;i<Ncycles;i++) if(!cycles_used[i]) std::cout << "Warning: cycle No. " << i << " of length " << cycle_len[i] << " was not used" << std::endl;
	std::cout << "mean(q) = " << meanq << std::endl;
	std::cout << "max(q) = "<< maxq << std::endl;
	std::cout << "global Z2 moves = " << global_z2_moves << std::endl;
	for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
		std::cout << "Observable #" << ++o << ": "<< name_of_observable(k) << std::endl;
		std::cout << "mean(O) = " << mean_O[k] << std::endl;
		std::cout << "std.dev.(O) = " << stdev_O[k] << std::endl;
	}
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
		std::cout << "Derived observable: " << name_of_derived_observable(o) << std::endl;
		std::cout << "mean(O) = " << mean_derived_O[o] << std::endl;
		std::cout << "std.dev.(O) = " << stdev_derived_O[o] << std::endl;
	}
	std::cout << "Elapsed cpu time = " << elapsed_time << " seconds" << std::endl;
}

int    gathered_qmax_achieved;
double gathered_elapsed_time;
double gathered_meanq;
double gathered_maxq;
double gathered_bin_mean_sgn[Nbins];
double gathered_bin_mean[N_all_observables][Nbins];

int main(int argc, char* argv[]){
#ifdef SAVE_UNFINISHED_CALCULATION
	signal(SIGTERM,signalHandler);
#endif
	int i, j, k, o=0; divdiff_init();
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	std::string timeseries_prefix, stream_timeseries_prefix;
	BetaAnnealOptions anneal_options;
	bool target_beta_set=false, target_tau_set=false;
	for(int arg=1;arg<argc;arg++){
		if(std::string(argv[arg]) == "--timeseries-prefix" && arg+1<argc) timeseries_prefix = argv[++arg];
		else if(std::string(argv[arg]) == "--stream-timeseries-prefix" && arg+1<argc) stream_timeseries_prefix = argv[++arg];
		else if(std::string(argv[arg]) == "--target-beta" && arg+1<argc){ fixed_target_beta=std::atof(argv[++arg]); target_beta_set=true; }
		else if(std::string(argv[arg]) == "--target-tau" && arg+1<argc){ fixed_target_tau=std::atof(argv[++arg]); target_tau_set=true; }
		else if(std::string(argv[arg]) == "--beta-anneal") anneal_options.automatic=true;
		else if(std::string(argv[arg]) == "--anneal-start-factor" && arg+1<argc){ anneal_options.start_factor=std::stod(argv[++arg]); anneal_options.start_factor_was_set=true; }
		else if(std::string(argv[arg]) == "--anneal-interval" && arg+1<argc){ anneal_options.interval=std::strtoull(argv[++arg],NULL,10); anneal_options.interval_was_set=true; }
		else if(std::string(argv[arg]) == "--beta-anneal-schedule" && arg+1<argc) anneal_options.schedule_file=argv[++arg];
		else{
			if(mpi_rank==0) std::cerr << "Unknown or incomplete option: " << argv[arg] << std::endl;
			MPI_Finalize(); return 2;
		}
	}
	if(target_beta_set && !target_tau_set) fixed_target_tau=fixed_target_beta/2.0;
	try{
		if(!std::isfinite(fixed_target_beta) || !(fixed_target_beta>0.0) || !std::isfinite(fixed_target_tau) || fixed_target_tau<0.0 || fixed_target_tau>fixed_target_beta)
			throw std::runtime_error("target beta/tau must satisfy beta > 0 and 0 <= tau <= beta");
		fixed_anneal_plan=make_beta_anneal_plan(anneal_options,std::vector<double>(1,fixed_target_beta),Tsteps,N);
	}catch(const std::exception& error){ if(mpi_rank==0) std::cerr << "Error: " << error.what() << std::endl; MPI_Finalize(); return 2; }
	dynamic_run_parameters=target_beta_set || target_tau_set || fixed_anneal_plan.enabled;
	if(dynamic_run_parameters){
		dynamic_run_identity=beta_anneal_hash(fixed_anneal_plan);
		uint64_t bits=0; std::memcpy(&bits,&fixed_target_beta,sizeof(bits)); dynamic_run_identity^=bits;
		std::memcpy(&bits,&fixed_target_tau,sizeof(bits)); dynamic_run_identity^=bits+UINT64_C(0x9e3779b97f4a7c15);
	}
	if(steps < Nbins*stepsPerMeasurement){
		std::cout << "Error: steps cannot be smaller than Nbins*stepsPerMeasurement." << std::endl;
		MPI_Finalize(); exit(1);
	}
	if(N == 0){
		std::cout << "Error: no particles found. At least one particle must be described by the Hamiltonian." << std::endl;
		MPI_Finalize(); exit(1);
	}
	if(mpi_rank == 0) resume_calc = check_QMC_data();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&resume_calc,1,MPI_INT,0,MPI_COMM_WORLD); init_rng();
	divdiff dd(q+4,500); divdiff ddfs(q+4,500); divdiff dd1(q+4,500); divdiff dd2(q+4,500); 
	d=&dd; dfs=&ddfs; ds1=&dd1; ds2=&dd2;
	if(dynamic_run_parameters){
		double initial_beta=fixed_anneal_plan.enabled ? fixed_anneal_plan.beta_at(0,0) : fixed_target_beta;
		configure_run_parameters(initial_beta,beta_anneal_tau(fixed_target_tau,initial_beta,fixed_target_beta));
	}
	start_time = MPI_Wtime();
	if(resume_calc){ load_QMC_data(); init_basic(); } else init();
	if(!timeseries_prefix.empty()){
		timeseries_enabled = true;
		if(mpi_rank==0){
			bool append = resume_calc && file_has_content(timeseries_prefix);
			timeseries_file.open(timeseries_prefix.c_str(),append ? std::ios::app : std::ios::out);
			if(!timeseries_file){ std::cerr << "Cannot open timeseries output: " << timeseries_prefix << std::endl; MPI_Abort(MPI_COMM_WORLD,2); }
			if(!append) write_timeseries_header();
		}
	}
	if(!stream_timeseries_prefix.empty()){
		stream_timeseries_enabled = true;
		std::string name = stream_timeseries_name(stream_timeseries_prefix,mpi_rank);
		bool append = resume_calc && file_has_content(name);
		stream_timeseries_file.open(name.c_str(),append ? std::ios::app : std::ios::out);
		if(!stream_timeseries_file){ std::cerr << "Cannot open stream timeseries output: " << name << std::endl; MPI_Abort(MPI_COMM_WORLD,2); }
		if(!append) write_stream_timeseries_header();
	}
	MPI_Barrier(MPI_COMM_WORLD);
	compute(); process_single_run();
	MPI_Barrier(MPI_COMM_WORLD);
	if(mpi_rank == 0){
		std::cout << std::endl;
		std::cout << "Parameters: beta = " << run_beta << ", Tsteps = " << Tsteps << ", steps = " << steps << std::endl << std::endl;
		std::cout << "Number of MPI processes: " << mpi_size << std::endl;
		std::cout << std::endl << "Output of the MPI process No. 0:" << std::endl << std::endl;
		printout_single_run();
		std::cout << std::endl;
	}
	if(mpi_size>4){
		if(mpi_rank == 0) std::cout << "Testing thermalization" << std::endl << std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		double* gathered_mean = new double[mpi_size]; double gathered_stdev, mean_mean, std_mean; sgn_stdev = sqrt(sgn_variance);
		MPI_Gather(&sgn_mean,1,MPI_DOUBLE,gathered_mean,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Reduce(&sgn_stdev,&gathered_stdev,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		if(mpi_rank == 0){
			mean_mean = std_mean = 0; gathered_stdev /= mpi_size;
			for(i=0;i<mpi_size;i++) mean_mean += gathered_mean[i]; mean_mean /= mpi_size;
			for(i=0;i<mpi_size;i++) std_mean += (gathered_mean[i] - mean_mean)*(gathered_mean[i] - mean_mean);
			std_mean /= (mpi_size - 1); std_mean = sqrt(std_mean);
			// std::cout << "mean of std.dev.(sgn(W)) = " << gathered_stdev << ", std.dev. of mean(sgn(W)) = " << std_mean;
			// if(gathered_stdev >= 0.7 * std_mean) std::cout << ": test passed" << std::endl; else std::cout << ": test failed" << std::endl;
		}
		for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
			MPI_Gather(&mean_O[k],1,MPI_DOUBLE,gathered_mean,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Reduce(&stdev_O[k],&gathered_stdev,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			if(mpi_rank == 0){
				std::cout << "Observable #" << ++o << ": "<< name_of_observable(k);
				mean_mean = std_mean = 0; gathered_stdev /= mpi_size;
				for(i=0;i<mpi_size;i++) mean_mean += gathered_mean[i]; mean_mean /= mpi_size;
				for(i=0;i<mpi_size;i++) std_mean += (gathered_mean[i] - mean_mean)*(gathered_mean[i] - mean_mean);
				std_mean /= (mpi_size - 1); std_mean = sqrt(std_mean);
				std::cout << ", mean of std.dev.(O) = " << gathered_stdev << ", std.dev. of mean(O) = " << std_mean;
				if(gathered_stdev >= 0.7 * std_mean) std::cout << ": test passed" << std::endl; else std::cout << ": test failed" << std::endl;
			}
		}
		for(k=0;k<N_derived_observables;k++) if(valid_derived_observable(k)){
			MPI_Gather(&mean_derived_O[k],1,MPI_DOUBLE,gathered_mean,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Reduce(&stdev_derived_O[k],&gathered_stdev,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			if(mpi_rank == 0){
				std::cout << "Derived observable: "<< name_of_derived_observable(k);
				mean_mean = std_mean = 0; gathered_stdev /= mpi_size;
				for(i=0;i<mpi_size;i++) mean_mean += gathered_mean[i]; mean_mean /= mpi_size;
				for(i=0;i<mpi_size;i++) std_mean += (gathered_mean[i] - mean_mean)*(gathered_mean[i] - mean_mean);
				std_mean /= (mpi_size - 1); std_mean = sqrt(std_mean);
				std::cout << ", mean of std.dev.(O) = " << gathered_stdev << ", std.dev. of mean(O) = " << std_mean;
				if(gathered_stdev >= 0.7 * std_mean) std::cout << ": test passed" << std::endl; else std::cout << ": test failed" << std::endl;
			}
		}
		delete[] gathered_mean; if(mpi_rank == 0) std::cout << std::endl;
	}
	if(mpi_rank == 0) std::cout << "Collecting statistics and finalizing the calculation" << std::endl << std::endl;
	if(mpi_rank==0 && timeseries_enabled) timeseries_file.close();
	if(stream_timeseries_enabled) stream_timeseries_file.close();
	double Rsum[N_all_observables] = {0}; sgn_mean = 0; o = 0;
	double over_bins_sum[N_all_observables] = {0}; sgn_variance = 0;
	double over_bins_sum_cov[N_all_observables] = {0};
	MPI_Barrier(MPI_COMM_WORLD);
	if(mpi_rank == 0) std::cout << "Total number of MC updates = " << steps*(unsigned long long)mpi_size << std::endl;
	MPI_Reduce(&elapsed_time,&gathered_elapsed_time,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&meanq,&gathered_meanq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); gathered_meanq /= mpi_size;
	if(mpi_rank == 0) std::cout << "Total mean(q) = " << gathered_meanq << std::endl;
	MPI_Reduce(&maxq,&gathered_maxq,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(mpi_rank == 0) std::cout << "Total max(q) = " << gathered_maxq << std::endl;
	MPI_Reduce(&qmax_achieved,&gathered_qmax_achieved,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
	if(mpi_rank == 0 && gathered_qmax_achieved) std::cout << "Warning: qmax = " << qmax << " was achieved by at least one of the MPI processes. The results may be incorrect. The qmax parameter should be increased." << std::endl;
	MPI_Reduce(bin_mean_sgn,gathered_bin_mean_sgn,Nbins,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(mpi_rank == 0){
		for(i=0;i<Nbins;i++) gathered_bin_mean_sgn[i] /= mpi_size;
		for(i=0;i<Nbins;i++) sgn_mean += gathered_bin_mean_sgn[i]; sgn_mean /= Nbins;
		for(i=0;i<Nbins;i++) sgn_variance += (gathered_bin_mean_sgn[i] - sgn_mean)*(gathered_bin_mean_sgn[i] - sgn_mean); sgn_variance /= (Nbins*(Nbins-1));
		std::cout << std::setprecision(9);
		std::cout << "Total mean(sgn(W)) = " << sgn_mean << std::endl;
		std::cout << "Total std.dev.(sgn(W)) = " << sqrt(sgn_variance) << std::endl;
	}
	for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
		MPI_Reduce(bin_mean[k],gathered_bin_mean[k],Nbins,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		if(mpi_rank == 0){
			std::cout << "Total of observable #" << ++o << ": "<< name_of_observable(k) << std::endl;
			for(i=0;i<Nbins;i++) gathered_bin_mean[k][i] /= mpi_size;
			for(i=0;i<Nbins;i++) Rsum[k] += gathered_bin_mean[k][i]; Rsum[k] /= Nbins;
			for(i=0;i<Nbins;i++) over_bins_sum[k] += (gathered_bin_mean[k][i] - Rsum[k])*(gathered_bin_mean[k][i] - Rsum[k]); over_bins_sum[k] /= (Nbins*(Nbins-1));
			for(i=0;i<Nbins;i++) over_bins_sum_cov[k] += (gathered_bin_mean[k][i] - Rsum[k])*(gathered_bin_mean_sgn[i] - sgn_mean); over_bins_sum_cov[k] /= (Nbins*(Nbins-1));
			mean_O[k] = Rsum[k]/sgn_mean*(1 + sgn_variance/sgn_mean/sgn_mean) - over_bins_sum_cov[k]/sgn_mean/sgn_mean;
			stdev_O[k] = fabs(Rsum[k]/sgn_mean)*sqrt(over_bins_sum[k]/Rsum[k]/Rsum[k] + sgn_variance/sgn_mean/sgn_mean - 2*over_bins_sum_cov[k]/Rsum[k]/sgn_mean);
			std::cout << "Total mean(O) = " << mean_O[k] << std::endl;
			std::cout << "Total std.dev.(O) = " << stdev_O[k] << std::endl;
		}
	}
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o) && mpi_rank == 0){
		mean_derived_O[o] = compute_derived_observable(o); jackknife_sum[o] = 0;
	}
	if(mpi_rank == 0) for(j=0;j<Nbins;j++){
		sgn_meanJ = sgn_varianceJ = 0;
		for(i=0;i<Nbins;i++) if(i!=j) sgn_meanJ += gathered_bin_mean_sgn[i]; sgn_meanJ /= (Nbins-1);
		for(i=0;i<Nbins;i++) if(i!=j) sgn_varianceJ += (gathered_bin_mean_sgn[i] - sgn_meanJ)*(gathered_bin_mean_sgn[i] - sgn_meanJ); sgn_varianceJ /= ((Nbins-1)*(Nbins-2));
		for(k=0;k<N_all_observables;k++) if(valid_observable[k]){
			Rsum[k] = over_bins_sum_cov[k] = 0;
			for(i=0;i<Nbins;i++) if(i!=j) Rsum[k] += gathered_bin_mean[k][i]; Rsum[k] /= (Nbins-1);
			for(i=0;i<Nbins;i++) if(i!=j) over_bins_sum_cov[k] += (gathered_bin_mean[k][i] - Rsum[k])*(gathered_bin_mean_sgn[i] - sgn_meanJ); over_bins_sum_cov[k] /= ((Nbins-1)*(Nbins-2));
			mean_O[k] = Rsum[k]/sgn_meanJ*(1 + sgn_varianceJ/sgn_meanJ/sgn_meanJ) - over_bins_sum_cov[k]/sgn_meanJ/sgn_meanJ;
		}
		for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
			jackknife_O[o] = compute_derived_observable(o);
			jackknife_sum[o] += (jackknife_O[o] - mean_derived_O[o])*(jackknife_O[o] - mean_derived_O[o]);
		}
	}
	for(o=0;o<N_derived_observables;o++) if(valid_derived_observable(o) && mpi_rank == 0){
		stdev_derived_O[o] = sqrt(jackknife_sum[o]*(Nbins-1)/Nbins);
		std::cout << "Total of derived observable: " << name_of_derived_observable(o) << std::endl;
		std::cout << "Total mean(O) = " << mean_derived_O[o] << std::endl;
		std::cout << "Total std.dev.(O) = " << stdev_derived_O[o] << std::endl;
	}
	if(mpi_rank == 0){
		std::cout << "Total elapsed cpu time = " << gathered_elapsed_time << " seconds" << std::endl;
		elapsed_time = MPI_Wtime()-start_time;
		std::cout << std::endl << "Wall-clock time = " << elapsed_time << " seconds" << std::endl;
	}
	MPI_Finalize();
	divdiff_clear_up();
	return 0;
}
