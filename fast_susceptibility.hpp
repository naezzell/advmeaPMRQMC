#pragma once
// #include "DivDiff.hpp"
#include "DivDiffExp.hpp"
// #include "ExExFloat.hpp"
#include "Toeplitz.hpp"
#include <ranges>
#include <vector>
#include <iostream>

constexpr size_t factorial_max = 100;
static std::array<double, factorial_max> factorial_arr = [] {
  std::array<double, factorial_max> ret;
  ret[0] = 1;
  for (size_t i = 1; i < factorial_max; ++i) {
    ret[i] = ret[i - 1] / static_cast<double>(i);
  }
  return ret;
}();

inline double OneOverNFac(size_t n) {
  if (n >= factorial_max) {
    std::cout << "factorial called on " << n << "\n";
    throw std::runtime_error("increase factorial max");
  }
  return factorial_arr[n];
}

template<size_t k>
std::array<double, k> Ikj(std::span<const double> E, // Hamiltonian energies only
                          std::span<const double> dd_beta_div2,
                          std::span<const double> beta_div2_pow_fac,
                          size_t j,
                          double beta_div2) {
  auto gfunc = [](const auto &elem) -> double { return get_double(elem); };

  DivDiffExp<Toeplitz<k>> dd2{gfunc};
  dd2.add_element(E[j] * beta_div2);
  for (const auto Ei : E | std::views::drop(j)) {
    Toeplitz<k> t{Ei * beta_div2};
    t.set(1, beta_div2);
    dd2.add_element(t);
  }

  std::array<double, k> inner1{};
  for (size_t i = 0; i < k; ++i) {
    inner1[i] = dd_beta_div2[j] * beta_div2_pow_fac[j];
    inner1[i] *= dd2.back().get(i) * beta_div2_pow_fac[dd2.size() - 1];
  }

  for (size_t r : std::views::iota(0uz, j) | std::views::reverse) {
    dd2.add_element(E[r] * beta_div2);
    for (size_t i = 0; i < k; ++i) {
      inner1[i] += dd_beta_div2[r] * beta_div2_pow_fac[r] *
                   dd2.back().get(i) * beta_div2_pow_fac[dd2.size() - 1];
    }
  }

  // for (size_t i = 0; i < k; ++i) {
  //   double sign = (i % 2 == 0) ? -1.0 : 1.0;
  //   inner1[i] *= sign / OneOverNFac(i);
  //   inner1[i] *= sign;
  // }

  return inner1;
}

template <size_t k>
std::array<double, k> Mk(std::span<const double> E, // Hamiltonian energies
                         std::span<const double> O, // Custom operator eigenvalues O(z_i)
                         std::span<const double> dd_beta,
                         std::span<const double> beta_pow_fac,
                         std::span<const double> beta_div2_pow_fac,
                         double beta_) {
  size_t q = E.size();

  std::vector<double> dd_beta_2;
  dd_beta_2.reserve(q);
  auto gfunc = [](const ExExFloat &elem) -> double {
    return elem.get_double();
  };

  // 1. Divided differences at -beta/2 use Hamiltonian energies E
  DivDiffExp<ExExFloat> dd1(gfunc);
  for (const auto Ei : E) {
    dd1.add_element(Ei * -beta_ / 2);
    dd_beta_2.push_back(dd1.back().get_double());
  }

  std::array<double, k> ret{};
  for (size_t j = 0; j < q; ++j) {
    // 2. Ikj receives Hamiltonian energies E
    auto this_j = Ikj<k>(E, dd_beta_2, beta_div2_pow_fac, j, -beta_ / 2);
    
    // 3. Multiply slice integral weight by observable value O(z_j)
    for (auto &&[ret_j, this_ji] : std::views::zip(ret, this_j)) {
      ret_j += O[j] * this_ji;
    }
  }

  double ddef_full = dd_beta.back() * beta_pow_fac[dd_beta.size() - 1];

  // 4. Prefactor uses O(z_0) for the initial operator O(0)
  double prefactor = O[0] / ddef_full;
  for (auto &ret_j : ret) {
    ret_j *= prefactor;
  }

  return ret;
}
