#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mainqmc.hpp"

static bool same_double(double left, double right){
    return left == right;
}

static bool same_weight(ExExFloat& left, ExExFloat& right){
    return left.get_double() == right.get_double() && left.sgn() == right.sgn();
}

static std::string rng_text(){
    std::ostringstream stream;
    stream << rng;
    return stream.str();
}

int main(){
    mpi_rank = 0; mpi_size = 0;
    divdiff_init();
    divdiff dd(qmax + 4,500), ddfs(qmax + 4,500), dd1(qmax + 4,500), dd2(qmax + 4,500);
    d = &dd; dfs = &ddfs; ds1 = &dd1; ds2 = &dd2;
    configure_run_parameters(0.7,0.35,0.75);
    configure_valid_observables();
    init_rng();
    init();
    for(int i=0;i<300;i++) update();

    std::vector<uint64_t> snapshot_before, snapshot_after;
    export_PMR_snapshot(snapshot_before);
    const std::string rng_before = rng_text();
    const std::bitset<N> lattice_before = lattice, z_before = z;
    const std::bitset<Nop> P_before = P;
    const int q_before = q, qmax_before = qmax_achieved;
    const double beta_before = run_beta, tau_before = run_tau, gamma_before = run_gamma;
    const std::complex<double> currD_before = currD, old_currD_before = old_currD;
    ExExFloat currWeight_before = currWeight;
    std::vector<int> sequence_before(Sq,Sq+qmax), backup_before(Sq_backup,Sq_backup+qmax);
    std::vector<double> energies_before(Energies,Energies+qmax+1);
    std::vector<std::complex<double> > partial_before(currD_partial,currD_partial+qmax);
    std::vector<std::complex<double> > md_before(currMDk_trace,currMDk_trace+qmax);
    std::vector<std::complex<double> > ml_before(currMDl_trace,currMDl_trace+qmax);
    std::vector<std::complex<double> > m0_before(currMD0_trace,currMD0_trace+qmax);
    const int d_length_before = d->CurrentLength;
    std::vector<double> d_z_before(d_length_before), d_div_before(d_length_before+extralen+1);
    for(int i=0;i<d_length_before;i++) d_z_before[i] = d->z[i];
    for(int i=0;i<d_length_before+extralen+1;i++) d_div_before[i] = d->divdiffs[i].get_double();

    const double target = GetLogWeightAtParameters(1.6,0.25);

    export_PMR_snapshot(snapshot_after);
    bool unchanged = snapshot_before == snapshot_after && rng_before == rng_text();
    unchanged = unchanged && lattice == lattice_before && z == z_before && P == P_before;
    unchanged = unchanged && q == q_before && qmax_achieved == qmax_before;
    unchanged = unchanged && same_double(run_beta,beta_before) && same_double(run_tau,tau_before) && same_double(run_gamma,gamma_before);
    unchanged = unchanged && currD == currD_before && old_currD == old_currD_before && same_weight(currWeight,currWeight_before);
    unchanged = unchanged && std::equal(sequence_before.begin(),sequence_before.end(),Sq);
    unchanged = unchanged && std::equal(backup_before.begin(),backup_before.end(),Sq_backup);
    unchanged = unchanged && std::equal(energies_before.begin(),energies_before.end(),Energies);
    unchanged = unchanged && std::equal(partial_before.begin(),partial_before.end(),currD_partial);
    unchanged = unchanged && std::equal(md_before.begin(),md_before.end(),currMDk_trace);
    unchanged = unchanged && std::equal(ml_before.begin(),ml_before.end(),currMDl_trace);
    unchanged = unchanged && std::equal(m0_before.begin(),m0_before.end(),currMD0_trace);
    unchanged = unchanged && d->CurrentLength == d_length_before;
    for(int i=0;i<d_length_before;i++) unchanged = unchanged && d->z[i] == d_z_before[i];
    for(int i=0;i<d_length_before+extralen+1;i++) unchanged = unchanged && d->divdiffs[i].get_double() == d_div_before[i];
    if(!std::isfinite(target) || !unchanged){
        std::cerr << "target-weight immutability failure" << std::endl;
        return 1;
    }
    std::cout << "target-weight immutability: OK\n";
    return 0;
}
