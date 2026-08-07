#ifndef PMRQMC_PT_SCHEDULE_HPP
#define PMRQMC_PT_SCHEDULE_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct PMRTemperatures {
	std::vector<double> beta;
	std::vector<double> tau;
};

// Log of the Metropolis ratio for exchanging configurations x and y between
// neighboring beta slots i and j:
//
//   [pi_i(y) pi_j(x)] / [pi_i(x) pi_j(y)].
//
// Keeping this four-weight identity outside the MPI code makes the ordering
// explicit and independently testable.
inline double replica_exchange_log_ratio(double log_x_at_i, double log_x_at_j,
		double log_y_at_j, double log_y_at_i){
	return log_x_at_j + log_y_at_i - log_x_at_i - log_y_at_j;
}

// Update endpoint history for one configuration trajectory.  The caller only
// invokes this at the hot or cold endpoint.  Return true exactly on the first
// return to the trajectory's origin after it has visited the opposite end.
inline bool record_pt_endpoint(int endpoint, int& origin, int& seen_opposite){
	if(origin < 0){
		origin = endpoint;
		seen_opposite = 0;
		return false;
	}
	if(endpoint != origin){
		seen_opposite = 1;
		return false;
	}
	if(seen_opposite){
		seen_opposite = 0;
		return true;
	}
	return false;
}

inline PMRTemperatures read_pt_schedule(const std::string& filename){
	std::ifstream input(filename.c_str());
	if(!input) throw std::runtime_error("cannot open tempering schedule: " + filename);
	PMRTemperatures schedule;
	std::string line;
	unsigned long line_number = 0;
	while(std::getline(input,line)){
		line_number++;
		std::string::size_type comment = line.find('#');
		if(comment != std::string::npos) line.erase(comment);
		std::istringstream row(line);
		double beta_value, tau_value;
		if(!(row >> beta_value)) continue;
		if(row >> tau_value){
			double extra;
			if(row >> extra) throw std::runtime_error("too many columns on schedule line " + std::to_string(line_number));
		}else{
			row.clear(); row >> std::ws;
			if(!row.eof()) throw std::runtime_error("schedule row must contain beta and optional tau on line " + std::to_string(line_number));
			tau_value = beta_value/2.0;
		}
		if(!std::isfinite(beta_value) || !(beta_value > 0.0)) throw std::runtime_error("schedule beta must be positive on line " + std::to_string(line_number));
		if(!std::isfinite(tau_value) || tau_value < 0.0 || tau_value > beta_value)
			throw std::runtime_error("schedule tau must satisfy 0 <= tau <= beta on line " + std::to_string(line_number));
		if(!schedule.beta.empty() && beta_value <= schedule.beta.back())
			throw std::runtime_error("schedule beta values must be strictly increasing on line " + std::to_string(line_number));
		schedule.beta.push_back(beta_value);
		schedule.tau.push_back(tau_value);
	}
	if(schedule.beta.empty()) throw std::runtime_error("tempering schedule is empty");
	return schedule;
}

inline uint64_t pt_schedule_hash(const PMRTemperatures& schedule){
	// FNV-1a over the IEEE-754 bytes makes checkpoint validation independent of
	// locale and of the textual spelling used in the schedule file.
	uint64_t hash = UINT64_C(1469598103934665603);
	auto add = [&hash](const void* ptr, size_t bytes){
		const unsigned char* data = static_cast<const unsigned char*>(ptr);
		for(size_t i=0;i<bytes;i++){ hash ^= data[i]; hash *= UINT64_C(1099511628211); }
	};
	uint64_t count = schedule.beta.size(); add(&count,sizeof(count));
	for(size_t i=0;i<schedule.beta.size();i++){
		add(&schedule.beta[i],sizeof(double)); add(&schedule.tau[i],sizeof(double));
	}
	return hash;
}

struct PMRTemperanceLayout {
	int temperatures;
	int independent_ladders;
};

inline PMRTemperanceLayout infer_pt_layout(int mpi_size, int temperatures, int requested_ladders = 0){
	if(temperatures < 1) throw std::runtime_error("tempering schedule must contain at least one temperature");
	if(mpi_size < temperatures || mpi_size % temperatures != 0)
		throw std::runtime_error("MPI size must be a positive multiple of the number of temperatures");
	int inferred = mpi_size / temperatures;
	if(requested_ladders != 0 && requested_ladders != inferred)
		throw std::runtime_error("independent-ladders does not match MPI size / schedule length");
	return {temperatures, inferred};
}

#endif
