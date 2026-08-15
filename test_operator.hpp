#include <bitset>
#include <random>

// template <size_t NBITS>
// double calculate_Oi(const std::bitset<NBITS> &l) {
//   return static_cast<double>(l.to_ullong());
// }

// template <size_t NBITS>
// double calculate_Oi(const std::bitset<NBITS>& l) {
//     static std::array<double, NBITS> coeffs = [] {
//       std::mt19937 gen(0);
//       std::uniform_real_distribution<double> dist(-1, 1);
//       std::array<double, NBITS> ret{};
//       for (size_t ii = 0; ii < NBITS; ++ii) {
//         ret[ii] = dist(gen);
//       }
//       return ret;
//     }();
//     double O_val = 0.0;
    
//     for (size_t i = 0; i < NBITS; ++i) {
//         // Map bit 0 -> spin -1.0, bit 1 -> spin +1.0
//         // l.test(i) is safer and faster than l[i] for std::bitset
//         double spin = l.test(i) ? 1.0 : -1.0; 
        
//         O_val += coeffs[i] * spin;
//     }
    
//     return O_val;
// }

// ============================================================================
// On-the-fly Néel Order Parameter: O_Neel^z
// Complexity: O(N)
// ============================================================================
template <size_t LX, size_t LY>
double calculate_ONeel(const std::bitset<LX * LY>& l) {
    constexpr double NBITS = static_cast<double>(LX * LY);
    double M_stag = 0.0;

    for (size_t y = 0; y < LY; ++y) {
        for (size_t x = 0; x < LX; ++x) {
            size_t idx = y * LX + x;
            
            // Map bit 0 -> spin -1.0, bit 1 -> spin +1.0 (Pauli Z eigenvalue)
            double spin = l.test(idx) ? 1.0 : -1.0; 
            double eta = ((x + y) % 2 == 0) ? 1.0 : -1.0;

            M_stag += eta * spin;
        }
    }

    // O_Neel = (3 / 4N^2) * (M_stag)^2
    constexpr double prefactor = 3.0 / (4.0 * NBITS * NBITS);
    return prefactor * (M_stag * M_stag);
}

// ============================================================================
// On-the-fly VBS Order Parameter: O_VBS^z
// Complexity: O(N)
// ============================================================================
template <size_t LX, size_t LY>
double calculate_OVBS(const std::bitset<LX * LY>& l) {
    constexpr double NBITS = static_cast<double>(LX * LY);
    double Dx = 0.0;
    double Dy = 0.0;

    for (size_t y = 0; y < LY; ++y) {
        for (size_t x = 0; x < LX; ++x) {
            size_t idx       = y * LX + x;
            size_t idx_right = y * LX + ((x + 1) % LX);  // PBC right
            size_t idx_up    = ((y + 1) % LY) * LX + x;  // PBC up

            // Get Z eigenvalues (+1.0 or -1.0)
            double s_i     = l.test(idx) ? 1.0 : -1.0;
            double s_right = l.test(idx_right) ? 1.0 : -1.0;
            double s_up    = l.test(idx_up) ? 1.0 : -1.0;

            // Z_i * Z_j correlations
            double bond_x = s_i * s_right;
            double bond_y = s_i * s_up;

            double sign_x = (x % 2 == 0) ? 1.0 : -1.0;
            double sign_y = (y % 2 == 0) ? 1.0 : -1.0;

            Dx += sign_x * bond_x;
            Dy += sign_y * bond_y;
        }
    }

    // O_VBS = (9 / 16N^2) * (Dx^2 + Dy^2)
    constexpr double prefactor = 9.0 / (16.0 * NBITS * NBITS);
    return prefactor * (Dx * Dx + Dy * Dy);
}

inline double calculate_Oi(const std::bitset<16> &l) {
    return calculate_ONeel<4, 4>(l);
}
