#include <bitset>
#include <complex>
#include <iostream>

#include "hamiltonian.hpp"

int main(){
    double fixed_z = 0.0, gamma_z = 0.0, fixed_zz = 0.0, gamma_zz = 0.0;
    for(int i=0;i<D0_size;i++){
        if(D0_product[i] == std::bitset<N>("1")){ fixed_z += D0_fixed_coeff[i].real(); gamma_z += D0_gamma_coeff[i].real(); }
        if(D0_product[i] == std::bitset<N>("11")){ fixed_zz += D0_fixed_coeff[i].real(); gamma_zz += D0_gamma_coeff[i].real(); }
    }
    std::cout << fixed_z << ' ' << gamma_z << ' ' << fixed_zz << ' ' << gamma_zz << '\n';
    return 0;
}
