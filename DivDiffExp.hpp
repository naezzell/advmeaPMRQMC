#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T> class DivDiffExp {
public:
  using d_type = T;
  using GetDoubleFunc = std::function<double(const d_type &element)>;

private:
  GetDoubleFunc get_double;

  int extralen = 30;
  int s = 1;
  size_t current_max_len = 10;
  double mu = 0.0;
  d_type expmu;

  std::vector<d_type> z; // Changed from double to d_type
  std::vector<d_type> h;
  std::vector<d_type> divdiff;
  std::vector<std::vector<d_type>> ddd;

  double get_mean(const std::vector<d_type> &vec) const {
    if (vec.empty())
      return 0.0;
    double sum = 0.0;
    for (const auto &val : vec) {
      sum += get_double(val);
    }
    return sum / static_cast<double>(vec.size());
  }

  double get_max_abs_diff(const std::vector<d_type> &vec) const {
    if (vec.empty())
      return 0.0;
    double min_val = get_double(vec[0]);
    double max_val = min_val;
    for (size_t i = 1; i < vec.size(); ++i) {
      double val = get_double(vec[i]);
      if (val < min_val)
        min_val = val;
      if (val > max_val)
        max_val = val;
    }
    return std::abs(max_val - min_val);
  }

  bool s_changed(const d_type &znew) const {
    return std::abs(get_double(znew) - mu) / 3.5 > s;
  }

  void add_all(size_t target_len, int force_s = 0) {
    if (target_len >= current_max_len) {
      current_max_len = std::max(current_max_len * 2, target_len);
    }

    std::vector<d_type> z_copy(z.begin(), z.begin() + target_len);
    int new_s =
        (force_s == 0)
            ? static_cast<int>(std::ceil(get_max_abs_diff(z_copy) / 3.5))
            : force_s;
    double new_mu = get_mean(z_copy);

    z.clear(); // Reset state, capacity remains safely intact

    add_element_force(z_copy[0], new_s, new_mu);
    for (size_t i = 1; i < target_len; ++i) {
      add_element_force(z_copy[i], 0, 0.0);
    }
  }

  void add_element_force(d_type znew, int force_s = 0, double force_mu = 0.0) {
    size_t n = z.size();
    size_t NN = current_max_len + extralen;

    if (h.size() <= NN)
      h.resize(NN + 1);
    if (divdiff.size() <= n)
      divdiff.resize(current_max_len + 1);

    z.push_back(znew);

    if (n == 0) {
      s = (force_s == 0) ? 1 : force_s;
      mu = (force_mu == 0.0) ? get_double(z[0]) : force_mu;

      using std::exp;
      expmu = d_type(exp(mu)); // Ensure safe initialization from scalar

      if (ddd.size() < static_cast<size_t>(s))
        ddd.resize(s);
      for (auto &row : ddd)
        row.resize(current_max_len + 1);

      h[0] = d_type(1.0);
      for (size_t k = 1; k <= NN; k++)
        h[k] = h[k - 1] / static_cast<double>(s);

      if (get_double(z[0]) != mu) {
        for (size_t k = NN; k > 0; k--)
          h[k - 1] += h[k] * (z[0] - d_type(mu)) / static_cast<double>(k);
      }

      d_type curr = expmu * h[0];
      for (int k = 0; k < s - 1; k++) {
        ddd[k][0] = curr;
        curr *= h[0];
      }
      divdiff[0] = expmu;
    } else if (s_changed(znew) || n >= current_max_len) {
      // Trigger a complete rebuild of the stored points
      add_all(n + 1, force_s);
    } else {
      for (size_t k = NN; k > n; k--) {
        h[k - 1] += h[k] * (z[n] - d_type(mu)) / static_cast<double>(k);
      }
      d_type curr = expmu * h[n];

      for (size_t k = n; k >= 1; k--) {
        h[k - 1] =
            (h[k - 1] * static_cast<double>(n) + h[k] * (z[n] - z[n - k])) /
            static_cast<double>(n - k + 1);
      }

      for (int k = 0; k < s - 1; k++) {
        ddd[k][n] = curr;
        curr = ddd[k][0] * h[n];
        for (size_t j = 1; j <= n; j++) {
          curr += ddd[k][j] * h[n - j];
        }
      }
      divdiff[n] = curr;
    }
  }

public:
  // Default constructor: Safe for types implicitly convertible to double (like
  // double, float)
  DivDiffExp() {
    if constexpr (std::is_convertible_v<d_type, double>) {
      get_double = [](const d_type &val) { return static_cast<double>(val); };
    } else {
      get_double = [](const d_type &) -> double {
        throw std::runtime_error("DivDiffExp: A custom GetDoubleFunc callback "
                                 "must be provided for non-scalar types!");
      };
    }
  }

  // Explicit constructor: Accepts projection callback for dual numbers,
  // matrices, etc.
  explicit DivDiffExp(GetDoubleFunc double_func)
      : get_double(std::move(double_func)) {}

  void add_element(d_type znew) { add_element_force(znew, 0, 0.0); }

  void remove_element() {
    if (!z.empty()) {
      size_t n = z.size() - 1;
      size_t NN = current_max_len + extralen;
      for (size_t k = 1; k <= n; k++) {
        h[k - 1] = (h[k - 1] * static_cast<double>(n - k + 1) -
                    h[k] * (z[n] - z[n - k])) /
                   static_cast<double>(n);
      }
      for (size_t k = n + 1; k <= NN; k++) {
        h[k - 1] -= h[k] * (z[n] - d_type(mu)) / static_cast<double>(k);
      }
      z.pop_back();
    }
  }

  void print_state() const {
    std::cout << "Length=" << z.size() << "\n";

    std::cout << "z={";
    for (size_t i = 0; i < z.size(); ++i) {
      std::cout << z[i] << (i < z.size() - 1 ? "," : "};\n");
    }

    std::cout << "DivDiffsRel={";
    for (size_t i = 0; i < z.size(); ++i) {
      std::cout << divdiff[i] << (i < z.size() - 1 ? "," : "};\n");
    }
  }

  [[nodiscard]] d_type back() const { return divdiff[z.size() - 1]; }
  [[nodiscard]] size_t size() const { return z.size(); }
};
