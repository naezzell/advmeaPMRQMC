#include "../divdiff.hpp"
#include "../pt_schedule.hpp"
#include "../beta_anneal.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>

int main(int argc, char** argv){
	divdiff_init();
	ExExFloat one(1.0), negative(-2.5), zero(0.0), extreme(1.0);
	for(int i=0;i<4000;i++) extreme *= 2.0;
	assert(std::fabs(one.log_abs()) < 1e-14);
	assert(std::fabs(negative.log_abs()-std::log(2.5)) < 1e-14);
	assert(std::isinf(zero.log_abs()) && zero.log_abs() < 0.0);
	assert(std::fabs(extreme.log_abs()-4000.0*std::log(2.0)) < 1e-10);

	// Current assignment: pi_i(x) pi_j(y) = exp(-2).  Crossed assignment:
	// pi_i(y) pi_j(x) = exp(-5), hence log acceptance ratio = -3.
	assert(std::fabs(replica_exchange_log_ratio(-1.0,-4.0,-1.0,-1.0)+3.0) < 1e-14);
	// Reversing the exchange negates the log ratio (detailed-balance identity).
	assert(std::fabs(replica_exchange_log_ratio(-1.0,-1.0,-4.0,-1.0)-3.0) < 1e-14);
	assert(replica_exchange_log_ratio(2.0,2.0,-7.0,-7.0) == 0.0);
	int origin = -1, seen_opposite = 0;
	assert(!record_pt_endpoint(0,origin,seen_opposite));

	BetaAnnealOptions automatic;
	automatic.automatic = true;
	automatic.interval = 10;
	automatic.interval_was_set = true;
	BetaAnnealPlan linear = make_beta_anneal_plan(automatic,std::vector<double>{1.0,4.0},100,7);
	assert(linear.enabled && linear.interval == 10 && linear.retarget_after(20));
	assert(std::fabs(linear.beta_at(0,0)-0.001) < 1e-14);
	assert(std::fabs(linear.beta_at(50,1)-2.002) < 1e-14);
	assert(linear.beta_at(100,1)==4.0 && beta_anneal_hash(linear)!=0);
	assert(!record_pt_endpoint(0,origin,seen_opposite));
	assert(!record_pt_endpoint(4,origin,seen_opposite));
	assert(record_pt_endpoint(0,origin,seen_opposite));
	assert(!record_pt_endpoint(0,origin,seen_opposite));

	if(argc > 1){
		std::ofstream schedule_file(argv[1]);
		schedule_file << "# beta tau\n0.1\n0.2 0.1\n1.0 0.7\n";
		schedule_file.close();
		PMRTemperatures schedule = read_pt_schedule(argv[1]);
		assert(schedule.beta.size()==3 && schedule.tau.size()==3);
		assert(schedule.tau[0]==0.05 && schedule.tau[2]==0.7);
		assert(infer_pt_layout(6,3).independent_ladders==2);
		std::string invalid_name = std::string(argv[1]) + ".invalid";
		std::ofstream invalid_file(invalid_name.c_str());
		invalid_file << "0.2 0.1\n0.1 0.05\n";
		invalid_file.close();
		bool rejected = false;
		try{ read_pt_schedule(invalid_name); } catch(const std::runtime_error&){ rejected = true; }
		assert(rejected);
		std::string qcpt_name = std::string(argv[1]) + ".qcpt";
		std::ofstream qcpt_file(qcpt_name.c_str());
		qcpt_file << "0.25 1.5\n0.7 0.75 0.5\n1.6 0.25\n";
		qcpt_file.close();
		PMRTemperatures qcpt = read_qcpt_schedule(qcpt_name);
		assert(qcpt.beta.size()==3 && qcpt.gamma.size()==3 && qcpt.tau.size()==3);
		assert(qcpt.gamma[0]==1.5 && qcpt.tau[0]==0.125 && qcpt.tau[1]==0.5);
		std::string duplicate_name = std::string(argv[1]) + ".duplicate";
		std::ofstream duplicate_file(duplicate_name.c_str());
		duplicate_file << "0.2 0.1\n0.2 0.1\n";
		duplicate_file.close();
		bool duplicate_rejected = false;
		try{ read_qcpt_schedule(duplicate_name); } catch(const std::runtime_error&){ duplicate_rejected = true; }
		assert(duplicate_rejected);

		std::string anneal_name = std::string(argv[1]) + ".anneal";
		std::ofstream anneal_file(anneal_name.c_str());
		anneal_file << "# completed beta0 beta1\n0 0.01 0.02\n50 0.2 0.8\n100 1 4\n";
		anneal_file.close();
		BetaAnnealOptions custom; custom.schedule_file = anneal_name;
		BetaAnnealPlan parsed = make_beta_anneal_plan(custom,std::vector<double>{1.0,4.0},100,10);
		assert(std::fabs(parsed.beta_at(25,0)-0.105) < 1e-14);
		assert(std::fabs(parsed.beta_at(75,1)-2.4) < 1e-14);
		std::string reheating_name = std::string(argv[1]) + ".reheating";
		std::ofstream reheating_file(reheating_name.c_str());
		reheating_file << "0 0.2\n50 0.1\n100 1\n";
		reheating_file.close();
		custom.schedule_file = reheating_name;
		bool reheating_rejected = false;
		try{ make_beta_anneal_plan(custom,std::vector<double>{1.0},100,10); }
		catch(const std::runtime_error&){ reheating_rejected = true; }
		assert(reheating_rejected);
	}
	divdiff_clear_up();
	return 0;
}
