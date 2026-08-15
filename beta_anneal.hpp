#ifndef PMRQMC_BETA_ANNEAL_HPP
#define PMRQMC_BETA_ANNEAL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct BetaAnnealOptions {
	bool automatic = false;
	std::string schedule_file;
	double start_factor = 0.001;
	bool start_factor_was_set = false;
	uint64_t interval = 0;
	bool interval_was_set = false;
};

struct BetaAnnealPlan {
	bool enabled = false;
	uint64_t updates = 0;
	uint64_t interval = 1;
	std::vector<uint64_t> waypoint_updates;
	std::vector<std::vector<double> > waypoint_betas;

	double beta_at(uint64_t completed_updates, size_t slot) const {
		if(!enabled || waypoint_updates.empty()) throw std::logic_error("beta_at called on a disabled annealing plan");
		if(slot >= waypoint_betas[0].size()) throw std::out_of_range("annealing slot is out of range");
		if(completed_updates >= updates) return waypoint_betas.back()[slot];
		std::vector<uint64_t>::const_iterator upper = std::upper_bound(
			waypoint_updates.begin(),waypoint_updates.end(),completed_updates);
		if(upper == waypoint_updates.begin()) return waypoint_betas.front()[slot];
		size_t right = static_cast<size_t>(upper-waypoint_updates.begin());
		if(right >= waypoint_updates.size()) return waypoint_betas.back()[slot];
		size_t left = right-1;
		double fraction = static_cast<double>(completed_updates-waypoint_updates[left]) /
			static_cast<double>(waypoint_updates[right]-waypoint_updates[left]);
		return waypoint_betas[left][slot] + fraction*(waypoint_betas[right][slot]-waypoint_betas[left][slot]);
	}

	bool retarget_after(uint64_t completed_updates) const {
		return enabled && (completed_updates == updates || completed_updates % interval == 0);
	}
};

inline bool beta_anneal_close(double left, double right){
	return std::fabs(left-right) <= 1e-12*std::max(1.0,std::max(std::fabs(left),std::fabs(right)));
}

inline double beta_anneal_tau(double target_tau, double beta_value, double target_beta){
	if(!std::isfinite(target_tau) || !std::isfinite(beta_value) ||
	   !std::isfinite(target_beta) || !(target_beta > 0.0))
		throw std::runtime_error("cannot scale tau with non-finite parameters or non-positive target beta");
	return target_tau*beta_value/target_beta;
}

inline BetaAnnealPlan make_beta_anneal_plan(const BetaAnnealOptions& options,
		const std::vector<double>& targets, uint64_t updates, uint64_t default_interval){
	BetaAnnealPlan plan;
	if(!options.automatic && options.schedule_file.empty()){
		if(options.start_factor_was_set || options.interval_was_set)
			throw std::runtime_error("anneal factor/interval requires --beta-anneal or --beta-anneal-schedule");
		return plan;
	}
	if(options.automatic && !options.schedule_file.empty())
		throw std::runtime_error("--beta-anneal and --beta-anneal-schedule are mutually exclusive");
	if(targets.empty()) throw std::runtime_error("beta annealing requires at least one target beta");
	if(updates == 0) throw std::runtime_error("beta annealing requires Tsteps > 0");
	plan.enabled = true;
	plan.updates = updates;
	plan.interval = options.interval_was_set ? options.interval : default_interval;
	if(plan.interval == 0) throw std::runtime_error("anneal interval must be positive");
	if(updates <= plan.interval) throw std::runtime_error("beta annealing requires at least two beta plateaus (Tsteps > anneal interval)");

	if(options.automatic){
		if(!std::isfinite(options.start_factor) || !(options.start_factor > 0.0) || options.start_factor > 1.0)
			throw std::runtime_error("anneal start factor must satisfy 0 < factor <= 1");
		plan.waypoint_updates.push_back(0);
		plan.waypoint_updates.push_back(updates);
		std::vector<double> start, finish;
		for(size_t slot=0;slot<targets.size();slot++){
			if(!std::isfinite(targets[slot]) || !(targets[slot] > 0.0))
				throw std::runtime_error("annealing target beta must be finite and positive");
			start.push_back(options.start_factor*targets[slot]);
			finish.push_back(targets[slot]);
		}
		plan.waypoint_betas.push_back(start);
		plan.waypoint_betas.push_back(finish);
		return plan;
	}

	std::ifstream input(options.schedule_file.c_str());
	if(!input) throw std::runtime_error("cannot open beta annealing schedule: " + options.schedule_file);
	std::string line; unsigned long line_number = 0;
	while(std::getline(input,line)){
		line_number++;
		std::string::size_type comment = line.find('#');
		if(comment != std::string::npos) line.erase(comment);
		std::istringstream row(line); uint64_t coordinate;
		if(!(row >> coordinate)) continue;
		std::vector<double> values(targets.size());
		for(size_t slot=0;slot<targets.size();slot++)
			if(!(row >> values[slot])) throw std::runtime_error("wrong beta column count on annealing schedule line " + std::to_string(line_number));
		double extra;
		if(row >> extra) throw std::runtime_error("wrong beta column count on annealing schedule line " + std::to_string(line_number));
		for(size_t slot=0;slot<values.size();slot++)
			if(!std::isfinite(values[slot]) || !(values[slot] > 0.0))
				throw std::runtime_error("annealing beta must be finite and positive on line " + std::to_string(line_number));
		if(!plan.waypoint_updates.empty()){
			if(coordinate <= plan.waypoint_updates.back())
				throw std::runtime_error("annealing update coordinates must be strictly increasing");
			for(size_t slot=0;slot<values.size();slot++) if(values[slot] < plan.waypoint_betas.back()[slot])
				throw std::runtime_error("annealing beta columns must be nondecreasing");
		}
		plan.waypoint_updates.push_back(coordinate);
		plan.waypoint_betas.push_back(values);
	}
	if(plan.waypoint_updates.size() < 2) throw std::runtime_error("annealing schedule must contain at least two rows");
	if(plan.waypoint_updates.front() != 0 || plan.waypoint_updates.back() != updates)
		throw std::runtime_error("annealing schedule must begin at update 0 and end at Tsteps");
	for(size_t slot=0;slot<targets.size();slot++) if(!beta_anneal_close(plan.waypoint_betas.back()[slot],targets[slot]))
		throw std::runtime_error("final annealing beta does not match its target beta");
	return plan;
}

inline uint64_t beta_anneal_hash(const BetaAnnealPlan& plan){
	if(!plan.enabled) return UINT64_C(0);
	uint64_t hash = UINT64_C(1469598103934665603);
	auto add = [&hash](const void* pointer, size_t bytes){
		const unsigned char* data = static_cast<const unsigned char*>(pointer);
		for(size_t index=0;index<bytes;index++){ hash ^= data[index]; hash *= UINT64_C(1099511628211); }
	};
	add(&plan.updates,sizeof(plan.updates)); add(&plan.interval,sizeof(plan.interval));
	uint64_t rows = plan.waypoint_updates.size(); add(&rows,sizeof(rows));
	for(size_t row=0;row<plan.waypoint_updates.size();row++){
		add(&plan.waypoint_updates[row],sizeof(uint64_t));
		for(size_t slot=0;slot<plan.waypoint_betas[row].size();slot++) add(&plan.waypoint_betas[row][slot],sizeof(double));
	}
	return hash;
}

inline uint64_t combine_schedule_anneal_hash(uint64_t schedule_hash, const BetaAnnealPlan& plan){
	if(!plan.enabled) return schedule_hash;
	uint64_t anneal = beta_anneal_hash(plan);
	return schedule_hash ^ (anneal + UINT64_C(0x9e3779b97f4a7c15) + (schedule_hash<<6) + (schedule_hash>>2));
}

#endif
