// Beta-only PMR replica exchange.  Ranks own fixed temperature slots; PMR
// configurations, rather than beta values, move between neighboring slots.
#include <mpi.h>
#include "mainqmc.hpp"
#include "pt_schedule.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static const uint32_t PT_CHECKPOINT_VERSION = 3;
static const char PT_CHECKPOINT_MAGIC[8] = {'P','M','R','P','T','C','K','1'};

struct PTCheckpointHeader {
	char magic[8];
	uint32_t version;
	uint64_t schedule_hash;
	int32_t world_rank;
	int32_t ladder;
	int32_t temperature;
	uint64_t completed_updates;
	uint32_t exchange_parity;
	uint64_t trajectory_id;
	uint64_t rng_size;
	uint64_t snapshot_words;
	uint64_t flow_words;
	uint64_t edge_words;
};

struct PTOptions {
	std::string schedule;
	std::string output_prefix;
	std::string timeseries_prefix;
	int updates_per_exchange = 10;
	int independent_ladders = 0;
	int checkpoint_every = 0;
	bool resume = false;
};

static void pt_signal_handler(int){ save_data_flag = 1; }

static void usage(const char* program){
	if(mpi_rank == 0) std::cerr
		<< "Usage: " << program << " --schedule FILE [--updates-per-exchange N]\\n"
		<< "       [--independent-ladders N] [--output-prefix PREFIX] [--timeseries-prefix FILE]\\n"
		<< "       [--checkpoint-every N] [--resume]\\n";
}

static bool parse_options(int argc, char** argv, PTOptions& options){
	for(int i=1;i<argc;i++){
		std::string arg(argv[i]);
		if(arg == "--schedule" && i+1<argc) options.schedule = argv[++i];
		else if(arg == "--updates-per-exchange" && i+1<argc) options.updates_per_exchange = std::atoi(argv[++i]);
		else if(arg == "--independent-ladders" && i+1<argc) options.independent_ladders = std::atoi(argv[++i]);
		else if(arg == "--checkpoint-every" && i+1<argc) options.checkpoint_every = std::atoi(argv[++i]);
		else if(arg == "--output-prefix" && i+1<argc) options.output_prefix = argv[++i];
		else if(arg == "--timeseries-prefix" && i+1<argc) options.timeseries_prefix = argv[++i];
		else if(arg == "--resume") options.resume = true;
		else return false;
	}
	if(options.output_prefix.empty()) options.output_prefix = "pmrqmc_pt";
	return !options.schedule.empty() && options.updates_per_exchange > 0 && options.checkpoint_every >= 0;
}

static void write_timeseries_header(std::ofstream& output){
	output << "temperature,beta,tau,measurement,updates,sign";
	for(int k=0;k<N_all_observables;k++)
		output << ",obs_" << k << ",signed_obs_" << k;
	output << ",elapsed_seconds\n";
}

static void write_timeseries_row(std::ofstream& output, int temperature, double beta_value, double tau_value,
		unsigned long long measurement, uint64_t updates, const double* observables,
		const double* signed_observables, double sign, double elapsed_seconds){
	output << temperature << ',' << std::setprecision(17) << beta_value << ',' << tau_value << ','
		<< measurement << ',' << updates << ',' << sign;
	for(int k=0;k<N_all_observables;k++)
		output << ',' << observables[k] << ',' << signed_observables[k];
	output << ',' << elapsed_seconds << '\n';
}

static std::string checkpoint_name(const std::string& prefix, int rank){
	return prefix + ".rank" + std::to_string(rank) + ".ptckpt";
}

static bool checkpoint_exists(const std::string& name){
	std::ifstream input(name.c_str(),std::ios::binary);
	return input.good();
}

static bool checkpoint_set_exists(const std::string& prefix, int size){
	for(int rank=0;rank<size;rank++) if(!checkpoint_exists(checkpoint_name(prefix,rank))) return false;
	return true;
}

static void write_checkpoint(const std::string& prefix, uint64_t schedule_hash, int world_rank,
		int ladder, int temperature, uint64_t completed_updates, uint32_t exchange_parity,
		uint64_t trajectory_id, const std::vector<uint64_t>& flow, uint64_t endpoint_visits,
		uint64_t round_trips, int endpoint_state, int endpoint_origin, int endpoint_seen_opposite,
		double crossed_weight_seconds,
		const std::vector<uint64_t>& attempts, const std::vector<uint64_t>& accepts){
	std::vector<uint64_t> snapshot; export_PMR_snapshot(snapshot);
	std::ostringstream rng_text; rng_text << rng;
	std::string rng_state = rng_text.str();
	PTCheckpointHeader header;
	std::copy(PT_CHECKPOINT_MAGIC,PT_CHECKPOINT_MAGIC+8,header.magic);
	header.version = PT_CHECKPOINT_VERSION; header.schedule_hash = schedule_hash;
	header.world_rank = world_rank; header.ladder = ladder; header.temperature = temperature;
	header.completed_updates = completed_updates; header.exchange_parity = exchange_parity;
	header.trajectory_id = trajectory_id; header.rng_size = rng_state.size();
	header.snapshot_words = snapshot.size();
	header.flow_words = flow.size(); header.edge_words = attempts.size();
	std::string final_name = checkpoint_name(prefix,world_rank);
	std::string temporary_name = final_name + ".tmp";
	std::ofstream output(temporary_name.c_str(),std::ios::binary|std::ios::trunc);
	if(!output) throw std::runtime_error("cannot open checkpoint temporary file: " + temporary_name);
	output.write(reinterpret_cast<const char*>(&header),sizeof(header));
	output.write(rng_state.data(),static_cast<std::streamsize>(rng_state.size()));
	output.write(reinterpret_cast<const char*>(in_bin_sum),sizeof(in_bin_sum));
	output.write(reinterpret_cast<const char*>(bin_mean),sizeof(bin_mean));
	output.write(reinterpret_cast<const char*>(&in_bin_sum_sgn),sizeof(in_bin_sum_sgn));
	output.write(reinterpret_cast<const char*>(bin_mean_sgn),sizeof(bin_mean_sgn));
	output.write(reinterpret_cast<const char*>(&meanq),sizeof(meanq));
	output.write(reinterpret_cast<const char*>(&maxq),sizeof(maxq));
	output.write(reinterpret_cast<const char*>(&qmax_achieved),sizeof(qmax_achieved));
	output.write(reinterpret_cast<const char*>(&measurement_step),sizeof(measurement_step));
	output.write(reinterpret_cast<const char*>(&endpoint_visits),sizeof(endpoint_visits));
	output.write(reinterpret_cast<const char*>(&round_trips),sizeof(round_trips));
	output.write(reinterpret_cast<const char*>(&endpoint_state),sizeof(endpoint_state));
	output.write(reinterpret_cast<const char*>(&endpoint_origin),sizeof(endpoint_origin));
	output.write(reinterpret_cast<const char*>(&endpoint_seen_opposite),sizeof(endpoint_seen_opposite));
	output.write(reinterpret_cast<const char*>(&crossed_weight_seconds),sizeof(crossed_weight_seconds));
	output.write(reinterpret_cast<const char*>(flow.data()),static_cast<std::streamsize>(flow.size()*sizeof(uint64_t)));
	output.write(reinterpret_cast<const char*>(attempts.data()),static_cast<std::streamsize>(attempts.size()*sizeof(uint64_t)));
	output.write(reinterpret_cast<const char*>(accepts.data()),static_cast<std::streamsize>(accepts.size()*sizeof(uint64_t)));
	output.write(reinterpret_cast<const char*>(snapshot.data()),static_cast<std::streamsize>(snapshot.size()*sizeof(uint64_t)));
	output.close();
	if(!output) throw std::runtime_error("failed while writing checkpoint: " + temporary_name);
	if(std::rename(temporary_name.c_str(),final_name.c_str()) != 0)
		throw std::runtime_error("cannot atomically publish checkpoint: " + final_name);
}

static uint64_t load_checkpoint(const std::string& prefix, uint64_t schedule_hash, int world_rank,
		int ladder, int temperature, uint64_t& trajectory_id, uint32_t& exchange_parity,
		std::vector<uint64_t>& flow, uint64_t& endpoint_visits, uint64_t& round_trips,
		int& endpoint_state, int& endpoint_origin, int& endpoint_seen_opposite,
		double& crossed_weight_seconds, std::vector<uint64_t>& attempts,
		std::vector<uint64_t>& accepts){
	std::string name = checkpoint_name(prefix,world_rank);
	std::ifstream input(name.c_str(),std::ios::binary);
	if(!input) throw std::runtime_error("cannot open checkpoint: " + name);
	PTCheckpointHeader header; input.read(reinterpret_cast<char*>(&header),sizeof(header));
	if(input.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
		std::memcmp(header.magic,PT_CHECKPOINT_MAGIC,8) != 0 || header.version != PT_CHECKPOINT_VERSION ||
		header.schedule_hash != schedule_hash || header.world_rank != world_rank ||
		header.ladder != ladder || header.temperature != temperature ||
		header.snapshot_words != PMR_snapshot_words() || header.flow_words != flow.size() ||
		header.edge_words != attempts.size())
		throw std::runtime_error("checkpoint is partial, incompatible, or belongs to another layout: " + name);
	std::string rng_state(header.rng_size,' ');
	input.read(&rng_state[0],static_cast<std::streamsize>(header.rng_size));
	input.read(reinterpret_cast<char*>(in_bin_sum),sizeof(in_bin_sum));
	input.read(reinterpret_cast<char*>(bin_mean),sizeof(bin_mean));
	input.read(reinterpret_cast<char*>(&in_bin_sum_sgn),sizeof(in_bin_sum_sgn));
	input.read(reinterpret_cast<char*>(bin_mean_sgn),sizeof(bin_mean_sgn));
	input.read(reinterpret_cast<char*>(&meanq),sizeof(meanq));
	input.read(reinterpret_cast<char*>(&maxq),sizeof(maxq));
	input.read(reinterpret_cast<char*>(&qmax_achieved),sizeof(qmax_achieved));
	input.read(reinterpret_cast<char*>(&measurement_step),sizeof(measurement_step));
	input.read(reinterpret_cast<char*>(&endpoint_visits),sizeof(endpoint_visits));
	input.read(reinterpret_cast<char*>(&round_trips),sizeof(round_trips));
	input.read(reinterpret_cast<char*>(&endpoint_state),sizeof(endpoint_state));
	input.read(reinterpret_cast<char*>(&endpoint_origin),sizeof(endpoint_origin));
	input.read(reinterpret_cast<char*>(&endpoint_seen_opposite),sizeof(endpoint_seen_opposite));
	input.read(reinterpret_cast<char*>(&crossed_weight_seconds),sizeof(crossed_weight_seconds));
	input.read(reinterpret_cast<char*>(flow.data()),static_cast<std::streamsize>(flow.size()*sizeof(uint64_t)));
	input.read(reinterpret_cast<char*>(attempts.data()),static_cast<std::streamsize>(attempts.size()*sizeof(uint64_t)));
	input.read(reinterpret_cast<char*>(accepts.data()),static_cast<std::streamsize>(accepts.size()*sizeof(uint64_t)));
	std::vector<uint64_t> snapshot(header.snapshot_words);
	input.read(reinterpret_cast<char*>(snapshot.data()),static_cast<std::streamsize>(snapshot.size()*sizeof(uint64_t)));
	if(!input) throw std::runtime_error("truncated checkpoint: " + name);
	std::istringstream rng_text(rng_state);
	if(!(rng_text >> rng)) throw std::runtime_error("invalid RNG state in checkpoint: " + name);
	import_PMR_snapshot(snapshot);
	trajectory_id = header.trajectory_id; exchange_parity = header.exchange_parity;
	return header.completed_updates;
}

static double pt_mean_derived[N_derived_observables], pt_stdev_derived[N_derived_observables];
static double pt_sgn_mean, pt_sgn_variance;

static void process_pt_run(){
	double Rsum[N_all_observables] = {0};
	double over_bins_sum[N_all_observables] = {0};
	double over_bins_sum_cov[N_all_observables] = {0};
	pt_sgn_mean = pt_sgn_variance = 0.0;
	for(int i=0;i<Nbins;i++) pt_sgn_mean += bin_mean_sgn[i];
	pt_sgn_mean /= Nbins;
	for(int i=0;i<Nbins;i++) pt_sgn_variance += (bin_mean_sgn[i]-pt_sgn_mean)*(bin_mean_sgn[i]-pt_sgn_mean);
	pt_sgn_variance /= (Nbins*(Nbins-1));
	for(int k=0;k<N_all_observables;k++) if(valid_observable[k]){
		for(int i=0;i<Nbins;i++) Rsum[k] += bin_mean[k][i];
		Rsum[k] /= Nbins;
		for(int i=0;i<Nbins;i++){
			over_bins_sum[k] += (bin_mean[k][i]-Rsum[k])*(bin_mean[k][i]-Rsum[k]);
			over_bins_sum_cov[k] += (bin_mean[k][i]-Rsum[k])*(bin_mean_sgn[i]-pt_sgn_mean);
		}
		over_bins_sum[k] /= (Nbins*(Nbins-1));
		over_bins_sum_cov[k] /= (Nbins*(Nbins-1));
		mean_O[k] = Rsum[k]/pt_sgn_mean*(1+pt_sgn_variance/(pt_sgn_mean*pt_sgn_mean)) - over_bins_sum_cov[k]/(pt_sgn_mean*pt_sgn_mean);
		stdev_O[k] = std::fabs(Rsum[k]/pt_sgn_mean)*std::sqrt(over_bins_sum[k]/(Rsum[k]*Rsum[k]) + pt_sgn_variance/(pt_sgn_mean*pt_sgn_mean) - 2*over_bins_sum_cov[k]/(Rsum[k]*pt_sgn_mean));
	}
	for(int o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)) pt_mean_derived[o] = compute_derived_observable(o);
	for(int o=0;o<N_derived_observables;o++) if(valid_derived_observable(o)){
		double jackknife_sum = 0.0;
		for(int excluded=0;excluded<Nbins;excluded++){
			double sign_mean = 0.0, sign_variance = 0.0;
			for(int i=0;i<Nbins;i++) if(i!=excluded) sign_mean += bin_mean_sgn[i];
			sign_mean /= (Nbins-1);
			for(int i=0;i<Nbins;i++) if(i!=excluded) sign_variance += (bin_mean_sgn[i]-sign_mean)*(bin_mean_sgn[i]-sign_mean);
			sign_variance /= ((Nbins-1)*(Nbins-2));
			for(int k=0;k<N_all_observables;k++) if(valid_observable[k]){
				Rsum[k] = over_bins_sum_cov[k] = 0.0;
				for(int i=0;i<Nbins;i++) if(i!=excluded) Rsum[k] += bin_mean[k][i];
				Rsum[k] /= (Nbins-1);
				for(int i=0;i<Nbins;i++) if(i!=excluded) over_bins_sum_cov[k] += (bin_mean[k][i]-Rsum[k])*(bin_mean_sgn[i]-sign_mean);
				over_bins_sum_cov[k] /= ((Nbins-1)*(Nbins-2));
				mean_O[k] = Rsum[k]/sign_mean*(1+sign_variance/(sign_mean*sign_mean)) - over_bins_sum_cov[k]/(sign_mean*sign_mean);
			}
			jackknife_sum += (compute_derived_observable(o)-pt_mean_derived[o])*(compute_derived_observable(o)-pt_mean_derived[o]);
		}
		pt_stdev_derived[o] = std::sqrt(jackknife_sum*(Nbins-1)/Nbins);
	}
}

static void exchange_edge(MPI_Comm ladder_comm, int ladder_rank, int temperature,
		const PMRTemperatures& schedule, int edge, uint32_t parity,
		std::vector<uint64_t>& current, std::vector<uint64_t>& received,
		uint64_t& trajectory_id, int& endpoint_state, uint64_t& attempts, uint64_t& accepted,
		double& crossed_weight_seconds, int& endpoint_origin, int& endpoint_seen_opposite){
	if(edge % 2 != static_cast<int>(parity % 2)) return;
	int partner = -1;
	if(temperature == edge || temperature == edge+1) partner = (temperature == edge) ? edge+1 : edge;
	if(partner < 0) return;
	int partner_rank = partner;
	// Serialize the configuration at the exchange point.  QMC updates mutate
	// the live PMR state, so the snapshot retained from the previous exchange
	// no longer represents the state whose weights are evaluated below.
	export_PMR_snapshot(current);
	double local_beta = schedule.beta[temperature];
	double other_beta = schedule.beta[partner];
	double log_local = GetLogWeightAtBeta(local_beta);
	double log_other = GetLogWeightAtBeta(other_beta);
	double logs[2] = {log_local,log_other}, peer_logs[2] = {0.0,0.0};
	std::vector<uint64_t> peer(current.size());
	double weight_start = MPI_Wtime();
	MPI_Sendrecv(logs,2,MPI_DOUBLE,partner_rank,200+edge,peer_logs,2,MPI_DOUBLE,partner_rank,200+edge,ladder_comm,MPI_STATUS_IGNORE);
	MPI_Sendrecv(current.data(),static_cast<int>(current.size()),MPI_UINT64_T,partner_rank,300+edge,
		peer.data(),static_cast<int>(peer.size()),MPI_UINT64_T,partner_rank,300+edge,ladder_comm,MPI_STATUS_IGNORE);
	crossed_weight_seconds += MPI_Wtime()-weight_start;
	// logs = {log pi_i(x), log pi_j(x)} and the peer sends
	// {log pi_j(y), log pi_i(y)}.
	double log_ratio = replica_exchange_log_ratio(
		logs[0], logs[1], peer_logs[0], peer_logs[1]);
	if(!std::isfinite(log_ratio)) log_ratio = (log_ratio > 0.0) ? 0.0 : -std::numeric_limits<double>::infinity();
	int proposal = 0, peer_proposal = 0;
	if(temperature == edge){
		attempts++;
		double probability = log_ratio >= 0.0 ? 1.0 : std::exp(log_ratio);
		proposal = val(rng) < probability;
	}
	MPI_Sendrecv(&proposal,1,MPI_INT,partner_rank,400+edge,&peer_proposal,1,MPI_INT,partner_rank,400+edge,ladder_comm,MPI_STATUS_IGNORE);
	int do_swap = temperature == edge ? proposal : peer_proposal;
	if(do_swap){
		if(temperature == edge) accepted++;
		current.swap(peer); import_PMR_snapshot(current);
		uint64_t peer_trajectory = 0;
		MPI_Sendrecv(&trajectory_id,1,MPI_UINT64_T,partner_rank,500+edge,&peer_trajectory,1,MPI_UINT64_T,partner_rank,500+edge,ladder_comm,MPI_STATUS_IGNORE);
		trajectory_id = peer_trajectory;
		int flow_state[3] = {endpoint_state, endpoint_origin, endpoint_seen_opposite};
		int peer_flow_state[3] = {-1,-1,0};
		MPI_Sendrecv(flow_state,3,MPI_INT,partner_rank,600+edge,peer_flow_state,3,MPI_INT,partner_rank,600+edge,ladder_comm,MPI_STATUS_IGNORE);
		endpoint_state = peer_flow_state[0]; endpoint_origin = peer_flow_state[1]; endpoint_seen_opposite = peer_flow_state[2];
	}else{
		// Keep the communication collective even for a rejected proposal.
		uint64_t ignored = 0;
		MPI_Sendrecv(&ignored,1,MPI_UINT64_T,partner_rank,500+edge,&ignored,1,MPI_UINT64_T,partner_rank,500+edge,ladder_comm,MPI_STATUS_IGNORE);
	}
}

static void write_csv_outputs(const PTOptions& options, const PMRTemperatures& schedule,
		const std::vector<double>& summaries, int world_size, int ntemperatures){
	std::ofstream observables((options.output_prefix+"_observables.csv").c_str());
	observables << "temperature,beta,tau,kind,name,mean,stdev\n";
	const size_t summary_size = 2 + 2*N_all_observables + 2*N_derived_observables;
	for(int temperature=0;temperature<ntemperatures;temperature++){
		const double* s = &summaries[temperature*summary_size];
		observables << temperature << ',' << std::setprecision(17) << schedule.beta[temperature] << ',' << schedule.tau[temperature] << ",sign,sign," << s[0] << ',' << s[1] << "\n";
		for(int k=0;k<N_all_observables;k++) if(valid_observable[k]){
			observables << temperature << ',' << schedule.beta[temperature] << ',' << schedule.tau[temperature] << ",observable,\"" << name_of_observable(k) << "\"," << s[2+2*k] << ',' << s[2+2*k+1] << "\n";
		}
		for(int k=0;k<N_derived_observables;k++) if(valid_derived_observable(k)){
			const size_t offset = 2+2*N_all_observables;
			observables << temperature << ',' << schedule.beta[temperature] << ',' << schedule.tau[temperature] << ",derived,\"" << name_of_derived_observable(k) << "\"," << s[offset+2*k] << ',' << s[offset+2*k+1] << "\n";
		}
	}
}

int main(int argc, char** argv){
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank); MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	start_time = MPI_Wtime();
	pt_mode = 1; std::signal(SIGTERM,pt_signal_handler); std::signal(SIGINT,pt_signal_handler);
	PTOptions options;
	if(!parse_options(argc,argv,options)){ usage(argv[0]); MPI_Finalize(); return 2; }
	PMRTemperatures schedule;
	try{ schedule = read_pt_schedule(options.schedule); }
	catch(const std::exception& error){ if(mpi_rank==0) std::cerr << "Error: " << error.what() << '\n'; MPI_Finalize(); return 2; }
	PMRTemperanceLayout layout;
	try{ layout = infer_pt_layout(mpi_size,static_cast<int>(schedule.beta.size()),options.independent_ladders); }
	catch(const std::exception& error){ if(mpi_rank==0) std::cerr << "Error: " << error.what() << '\n'; MPI_Finalize(); return 2; }
	int ntemperatures = layout.temperatures;
	int ladder = mpi_rank/ntemperatures, temperature = mpi_rank%ntemperatures;
	MPI_Comm ladder_comm, temperature_comm;
	MPI_Comm_split(MPI_COMM_WORLD,ladder,temperature,&ladder_comm);
	MPI_Comm_split(MPI_COMM_WORLD,temperature,ladder,&temperature_comm);
	int ladder_rank; MPI_Comm_rank(ladder_comm,&ladder_rank);
	std::ofstream timeseries_file;
	if(!options.timeseries_prefix.empty() && mpi_rank==0){
		timeseries_file.open(options.timeseries_prefix.c_str());
		if(!timeseries_file){ std::cerr << "Cannot open timeseries output: " << options.timeseries_prefix << '\n'; MPI_Abort(MPI_COMM_WORLD,2); }
		write_timeseries_header(timeseries_file);
	}
	uint64_t schedule_hash = pt_schedule_hash(schedule);
	uint64_t trajectory_id = temperature; uint32_t exchange_parity = 0; uint64_t completed_updates = 0;
	std::vector<uint64_t> current, received;
	std::vector<uint64_t> flow(static_cast<size_t>(ntemperatures)*ntemperatures,0);
	uint64_t endpoint_visits = 0, round_trips = 0; int endpoint_state = -1, endpoint_origin = -1, endpoint_seen_opposite = 0;
	std::vector<uint64_t> attempts(std::max(0,ntemperatures-1),0), accepts(std::max(0,ntemperatures-1),0);
	double crossed_weight_seconds = 0.0;
	try{
		if(steps < Nbins*stepsPerMeasurement) throw std::runtime_error("steps must be at least Nbins*stepsPerMeasurement");
		if(N == 0) throw std::runtime_error("Hamiltonian contains no particles");
		divdiff_init();
		divdiff dd(q+4,500), ddfs(q+4,500), dd1(q+4,500), dd2(q+4,500);
		d=&dd; dfs=&ddfs; ds1=&dd1; ds2=&dd2;
		configure_run_parameters(schedule.beta[temperature],schedule.tau[temperature]);
		init_rng();
		bool resume = false;
		if(options.resume){
			int has = (mpi_rank==0 && checkpoint_set_exists(options.output_prefix,mpi_size)) ? 1 : 0;
			MPI_Bcast(&has,1,MPI_INT,0,MPI_COMM_WORLD);
			if(!has) throw std::runtime_error("--resume requested but a complete checkpoint set was not found");
			resume = true;
		}
		if(resume) completed_updates = load_checkpoint(options.output_prefix,schedule_hash,mpi_rank,ladder,temperature,trajectory_id,exchange_parity,flow,endpoint_visits,round_trips,endpoint_state,endpoint_origin,endpoint_seen_opposite,crossed_weight_seconds,attempts,accepts);
		else init();
		export_PMR_snapshot(current); received.resize(current.size());
		const uint64_t total_updates = static_cast<uint64_t>(Tsteps) + static_cast<uint64_t>(steps);
		const uint64_t exchange_interval = static_cast<uint64_t>(options.updates_per_exchange);
		for(uint64_t update_number=completed_updates; update_number<total_updates; update_number++){
			step = update_number;
			update();
			if(update_number >= static_cast<uint64_t>(Tsteps) &&
				((update_number-static_cast<uint64_t>(Tsteps)+1) % static_cast<uint64_t>(stepsPerMeasurement) == 0)){
				measure();
				if(!options.timeseries_prefix.empty()){
					std::vector<double> local_signed(N_all_observables,0.0);
					std::vector<double> reduced(N_all_observables,0.0), reduced_signed(N_all_observables,0.0);
					double reduced_sign = 0.0;
					for(int k=0;k<N_all_observables;k++)
						local_signed[k] = last_measurement[k]*last_measurement_sgn;
					MPI_Reduce(last_measurement,reduced.data(),N_all_observables,MPI_DOUBLE,MPI_SUM,0,temperature_comm);
					MPI_Reduce(local_signed.data(),reduced_signed.data(),N_all_observables,MPI_DOUBLE,MPI_SUM,0,temperature_comm);
					MPI_Reduce(&last_measurement_sgn,&reduced_sign,1,MPI_DOUBLE,MPI_SUM,0,temperature_comm);
					// Only the first independent ladder writes samples.  The ladder
					// communicator then collects one reduced sample per temperature.
					std::vector<double> ladder_sample(2*N_all_observables+1,0.0);
					if(ladder==0){
						for(int k=0;k<N_all_observables;k++){
							ladder_sample[k] = reduced[k]/layout.independent_ladders;
							ladder_sample[N_all_observables+k] = reduced_signed[k]/layout.independent_ladders;
						}
						ladder_sample[2*N_all_observables] = reduced_sign/layout.independent_ladders;
					}
					std::vector<double> all_samples;
					if(ladder_rank==0) all_samples.resize(static_cast<size_t>(ntemperatures)*(2*N_all_observables+1),0.0);
					MPI_Gather(ladder_sample.data(),2*N_all_observables+1,MPI_DOUBLE,
						ladder_rank==0 ? all_samples.data() : NULL,2*N_all_observables+1,MPI_DOUBLE,0,ladder_comm);
					if(mpi_rank==0){
						for(int t=0;t<ntemperatures;t++){
							const double* sample = &all_samples[static_cast<size_t>(t)*(2*N_all_observables+1)];
							write_timeseries_row(timeseries_file,t,schedule.beta[t],schedule.tau[t],measurement_step+1,
								update_number+1,sample,sample+N_all_observables,
								sample[2*N_all_observables],MPI_Wtime()-start_time);
						}
					}
				}
				measurement_step++;
			}
			if((update_number+1) % exchange_interval == 0 || update_number+1 == total_updates){
				exchange_parity = static_cast<uint32_t>(((update_number+1)/exchange_interval - 1) % 2);
				for(int edge=0;edge<ntemperatures-1;edge++) exchange_edge(ladder_comm,ladder_rank,temperature,schedule,edge,exchange_parity,current,received,trajectory_id,endpoint_state,attempts[edge],accepts[edge],crossed_weight_seconds,endpoint_origin,endpoint_seen_opposite);
				flow[static_cast<size_t>(trajectory_id)*ntemperatures+temperature]++;
				if(temperature==0 || temperature==ntemperatures-1){
					endpoint_visits++;
					if(record_pt_endpoint(temperature,endpoint_origin,endpoint_seen_opposite))
						round_trips++;
					endpoint_state = temperature;
				}
				int local_stop = (save_data_flag || pt_stop_requested) ? 1 : 0, global_stop = 0;
				MPI_Allreduce(&local_stop,&global_stop,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
				if(options.checkpoint_every > 0 && (update_number+1) % static_cast<uint64_t>(options.checkpoint_every) == 0) global_stop = 2;
				if(global_stop){
					write_checkpoint(options.output_prefix,schedule_hash,mpi_rank,ladder,temperature,update_number+1,exchange_parity,trajectory_id,flow,endpoint_visits,round_trips,endpoint_state,endpoint_origin,endpoint_seen_opposite,crossed_weight_seconds,attempts,accepts);
					MPI_Barrier(MPI_COMM_WORLD);
					if(global_stop == 1){ MPI_Comm_free(&ladder_comm); MPI_Comm_free(&temperature_comm); divdiff_clear_up(); MPI_Finalize(); return 0; }
				}
			}
		}
		export_PMR_snapshot(current);
		// Aggregate bins only among replicas occupying the same temperature.
		std::vector<double> reduced_bins(static_cast<size_t>(N_all_observables)*Nbins,0.0), reduced_sign(Nbins,0.0);
		MPI_Reduce(&bin_mean[0][0],reduced_bins.data(),static_cast<int>(reduced_bins.size()),MPI_DOUBLE,MPI_SUM,0,temperature_comm);
		MPI_Reduce(bin_mean_sgn,reduced_sign.data(),Nbins,MPI_DOUBLE,MPI_SUM,0,temperature_comm);
		std::vector<double> summary(2+2*N_all_observables+2*N_derived_observables,0.0);
		if(ladder==0){
			for(int k=0;k<N_all_observables;k++) for(int b=0;b<Nbins;b++) bin_mean[k][b]=reduced_bins[static_cast<size_t>(k)*Nbins+b]/layout.independent_ladders;
			for(int b=0;b<Nbins;b++) bin_mean_sgn[b]=reduced_sign[b]/layout.independent_ladders;
			process_pt_run();
			summary[0]=pt_sgn_mean; summary[1]=std::sqrt(pt_sgn_variance);
			for(int k=0;k<N_all_observables;k++){ summary[2+2*k]=mean_O[k]; summary[2+2*k+1]=stdev_O[k]; }
			for(int k=0;k<N_derived_observables;k++){ const size_t offset=2+2*N_all_observables; summary[offset+2*k]=pt_mean_derived[k]; summary[offset+2*k+1]=pt_stdev_derived[k]; }
		}
		std::vector<double> all_summaries;
		if(mpi_rank==0) all_summaries.resize(static_cast<size_t>(mpi_size)*(summary.size()),0.0);
		MPI_Gather(summary.data(),static_cast<int>(summary.size()),MPI_DOUBLE,mpi_rank==0?all_summaries.data():NULL,static_cast<int>(summary.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
		std::vector<uint64_t> total_flow;
		if(mpi_rank==0) total_flow.resize(flow.size(),0);
		MPI_Reduce(flow.data(),mpi_rank==0?total_flow.data():NULL,static_cast<int>(flow.size()),MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
		std::vector<uint64_t> all_attempts, all_accepts;
		if(mpi_rank==0){ all_attempts.resize(attempts.size()); all_accepts.resize(accepts.size()); }
		MPI_Reduce(attempts.data(),mpi_rank==0?all_attempts.data():NULL,static_cast<int>(attempts.size()),MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(accepts.data(),mpi_rank==0?all_accepts.data():NULL,static_cast<int>(accepts.size()),MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
		uint64_t total_endpoint_visits = 0, total_round_trips = 0;
		double total_crossed_weight_seconds = 0.0, total_meanq = 0.0, total_maxq = 0.0;
		int any_qmax_achieved = 0;
		MPI_Reduce(&endpoint_visits,&total_endpoint_visits,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&round_trips,&total_round_trips,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&crossed_weight_seconds,&total_crossed_weight_seconds,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&meanq,&total_meanq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&maxq,&total_maxq,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
		MPI_Reduce(&qmax_achieved,&any_qmax_achieved,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
		if(mpi_rank==0){
			std::vector<double> temperature_summaries(static_cast<size_t>(ntemperatures)*summary.size(),0.0);
			for(int t=0;t<ntemperatures;t++) std::copy(all_summaries.begin()+static_cast<size_t>(t)*summary.size(),all_summaries.begin()+(static_cast<size_t>(t)+1)*summary.size(),temperature_summaries.begin()+static_cast<size_t>(t)*summary.size());
			write_csv_outputs(options,schedule,temperature_summaries,mpi_size,ntemperatures);
			std::ofstream swaps((options.output_prefix+"_swaps.csv").c_str()); swaps << "edge,attempts,accepted,acceptance\n";
			for(size_t edge=0;edge<all_attempts.size();edge++) swaps << edge << ',' << all_attempts[edge] << ',' << all_accepts[edge] << ',' << (all_attempts[edge]?static_cast<double>(all_accepts[edge])/all_attempts[edge]:0.0) << '\n';
			std::ofstream flow_file((options.output_prefix+"_flow.csv").c_str());
			flow_file << "trajectory,temperature,visits,endpoint_visits,round_trips,mean_q,max_q,qmax_achieved,crossed_weight_seconds\n";
			for(int tr=0;tr<ntemperatures;tr++) for(int t=0;t<ntemperatures;t++)
				flow_file << tr << ',' << t << ',' << total_flow[static_cast<size_t>(tr)*ntemperatures+t] << ','
					<< total_endpoint_visits << ',' << total_round_trips << ',' << total_meanq/(mpi_size*measurements) << ','
					<< total_maxq << ',' << any_qmax_achieved << ',' << total_crossed_weight_seconds << '\n';
			if(any_qmax_achieved)
				std::cerr << "Warning: qmax = " << qmax << " was reached by at least one PT rank; increase qmax.\n";
			std::cout << "Parallel tempering completed: " << ntemperatures << " temperatures, " << layout.independent_ladders << " independent ladder(s)\n";
			if(!options.timeseries_prefix.empty()) timeseries_file.close();
		}
		MPI_Comm_free(&ladder_comm); MPI_Comm_free(&temperature_comm); divdiff_clear_up(); MPI_Finalize(); return 0;
	}catch(const std::exception& error){
		std::cerr << "MPI rank " << mpi_rank << " error: " << error.what() << '\n';
		MPI_Abort(MPI_COMM_WORLD,3);
	}
	return 3;
}
