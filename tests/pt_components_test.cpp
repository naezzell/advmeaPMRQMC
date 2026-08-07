#include "../divdiff.hpp"
#include "../pt_schedule.hpp"

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
	}
	divdiff_clear_up();
	return 0;
}
