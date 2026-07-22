#pragma once
#include "DivDiff.hpp"
#include "DivDiffExp.hpp"
// #include "ExExFloat.hpp"
#include "Toeplitz.hpp"
#include <ranges>
#include <vector>
#include <iostream>

constexpr size_t factorial_max = 10'000;
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

template <size_t k>
std::array<double, k> Ikj(const std::vector<double> &x, size_t j, double a) {
  auto dfunc = [](int, auto mu) -> decltype(auto) { return exp(mu); };
  auto gfunc = [](const auto &elem) -> double { return get_double(elem); };
  DivDiff<Toeplitz<k>> dd1{dfunc, gfunc};
  std::vector<double> divdiffs;
  for (const auto xi : x) {
    dd1.add_element(xi * a);
    divdiffs.push_back(dd1.back().get(0));
  }

  DivDiff<Toeplitz<k>> dd2{dfunc, gfunc};
  dd2.add_element(x[j] * a);
  for (const auto xi : x | std::views::drop(j)) {
    Toeplitz<k> t{xi * a};
    t.set(1, a);
    dd2.add_element(t);
  }

  std::array<double, k> inner1{};
  for (size_t i = 0; i < k; ++i) {
    inner1[i] =
        divdiffs[j] * OneOverNFac(j) * std::pow(a, static_cast<double>(j));
    inner1[i] *= dd2.back().get(i) * OneOverNFac(dd2.size() - 1) *
                 std::pow(a, dd2.size() - 1);
  }

  for (size_t r : std::views::iota(0uz, j) | std::views::reverse) {
    dd2.add_element(x[r] * a);
    for (size_t i = 0; i < k; ++i) {
      inner1[i] += divdiffs[r] * OneOverNFac(r) * std::pow(a, r) *
                   dd2.back().get(i) * OneOverNFac(dd2.size() - 1) *
                   std::pow(a, dd2.size() - 1);
    }
  }

  for (size_t i = 0; i < k; ++i) {
    double sign = (i % 2 == 0) ? -1.0 : 1.0;
    inner1[i] *= sign / OneOverNFac(i);
  }

  return inner1;
}

template <size_t k>
std::array<double, k> Ikj(const std::vector<double> &x,
                          const std::vector<double> divdiffs, size_t j,
                          double a) {
  auto gfunc = [](const auto &elem) -> double { return get_double(elem); };

  DivDiffExp<Toeplitz<k>> dd2{gfunc};
  dd2.add_element(x[j] * a);
  for (const auto xi : x | std::views::drop(j)) {
    Toeplitz<k> t{xi * a};
    t.set(1, a);
    dd2.add_element(t);
  }

  std::array<double, k> inner1{};
  for (size_t i = 0; i < k; ++i) {
    inner1[i] =
        divdiffs[j] * OneOverNFac(j) * std::pow(a, static_cast<double>(j));
    inner1[i] *= dd2.back().get(i) * OneOverNFac(dd2.size() - 1) *
                 std::pow(a, dd2.size() - 1);
  }

  for (size_t r : std::views::iota(0uz, j) | std::views::reverse) {
    dd2.add_element(x[r] * a);
    for (size_t i = 0; i < k; ++i) {
      inner1[i] += divdiffs[r] * OneOverNFac(r) * std::pow(a, r) *
                   dd2.back().get(i) * OneOverNFac(dd2.size() - 1) *
                   std::pow(a, dd2.size() - 1);
    }
  }

  for (size_t i = 0; i < k; ++i) {
    double sign = (i % 2 == 0) ? -1.0 : 1.0;
    inner1[i] *= sign / OneOverNFac(i);
  }

  return inner1;
}

template <size_t k>
std::array<double, k> Mk(const std::vector<double> &x, double beta_) {
  size_t q = x.size();
  // std::cout << q << "\n";

  std::vector<double> divdiffs;
  divdiffs.reserve(x.size());
  auto gfunc = [](const ExExFloat &elem) -> double {
    return elem.get_double();
  };

  DivDiffExp<ExExFloat> dd1(gfunc);
  for (const auto xi : x) {
    dd1.add_element(xi * -beta_ / 2);
    divdiffs.push_back(dd1.back().get_double());
  }

  std::array<double, k> ret{};
  for (size_t j = 0; j < q; ++j) {
    auto this_j = Ikj<k>(x, divdiffs, j, -beta_ / 2);
    for (auto &&[ret_j, this_ji] : std::views::zip(ret, this_j)) {
      ret_j += x[j] * this_ji;
    }
  }
  DivDiffExp<double> ddef;
  for (const auto &xi : x) {
    ddef.add_element(xi * -beta_);
  }
  double ddef_full = ddef.back() * OneOverNFac(ddef.size() - 1) *
                     std::pow(-beta_, ddef.size() - 1);
  double prefactor = x[0] / ddef_full;
  for (auto &ret_j : ret) {
    ret_j *= prefactor;
  }

  return ret;
}

