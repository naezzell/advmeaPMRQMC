#include <bitset>

template <size_t NBITS>
double calculate_Oi(const std::bitset<NBITS> &l) {
  return static_cast<double>(l.to_ullong());
}
