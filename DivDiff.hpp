#pragma once
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>

template <typename T> class DivDiff {
public:
  using d_type = T;
  using DerivativeFunc = std::function<double(int k, double mu)>;
  using GetDoubleFunc = std::function<double(const d_type &element)>;

private:
  DerivativeFunc get_derivative;
  GetDoubleFunc get_double;

  int extralen{30};
  double mu{0.0};
  size_t current_max_len{10};
  double mu_drift_threshold{2.0};
  std::vector<d_type> z;
  std::vector<d_type> h;
  std::vector<d_type> divdiff;

  auto get_mean(const std::vector<d_type> &vec) const -> double {
    if (vec.empty())
      return 0.0;
    d_type sum = std::accumulate(vec.begin(), vec.end(), d_type(0.0));
    return get_double(sum) / static_cast<double>(vec.size());
  }

  auto mu_drifted(d_type znew) const -> double {
    d_type theoretical_mean =
        (std::accumulate(z.begin(), z.end(), d_type(0.0)) + znew) /
        (z.size() + 1);
    return std::abs(get_double(theoretical_mean - d_type(mu))) >
           mu_drift_threshold;
  }

  void add_all(size_t target_len,
               std::optional<double> force_mu = std::nullopt) {
    if (target_len >= current_max_len) {
      current_max_len = std::max(current_max_len * 2, target_len);
    }

    std::vector<d_type> z_copy(z.begin(), z.begin() + target_len);
    double new_mu = force_mu.value_or(get_mean(z_copy));

    z.clear();

    add_element_force(z_copy[0], new_mu, true);
    for (size_t i = 1; i < target_len; ++i) {
      add_element_force(z_copy[i], std::nullopt, true);
    }
  }

  void add_element_force(d_type znew,
                         std::optional<double> force_mu = std::nullopt,
                         bool is_rebuilding = false) {
    size_t n = z.size();
    size_t NN = current_max_len + extralen;

    if (h.size() <= NN)
      h.resize(NN + 1, 0.0);
    if (divdiff.size() <= n)
      divdiff.resize(current_max_len + 1, 0.0);

    if (n == 0) {
      z.push_back(znew);
      double z0double = get_double(z[0]);
      mu = force_mu.value_or(z0double);

      for (size_t k = 0; k <= NN; k++) {
        h[k] = get_derivative(k, mu);
      }

      if (mu != z0double) {
        for (size_t k = NN; k > 0; k--) {
          h[k - 1] += h[k] * (z[0] - d_type(mu)) / k;
        }
      }
      divdiff[0] = h[0];

    } else if (!is_rebuilding && (mu_drifted(znew) || n >= current_max_len)) {
      z.push_back(znew);
      add_all(n + 1, std::nullopt);

    } else {
      z.push_back(znew);

      for (size_t k = NN; k > n; k--) {
        h[k - 1] += h[k] * (z[n] - d_type(mu)) / k;
      }

      d_type curr = h[n];

      for (size_t k = n; k >= 1; k--) {
        h[k - 1] = (h[k - 1] * n + h[k] * (z[n] - z[n - k])) / (n - k + 1);
      }

      divdiff[n] = curr;
    }
  }

public:
  explicit DivDiff(DerivativeFunc deriv_func, GetDoubleFunc double_func)
      : get_derivative(std::move(deriv_func)),
        get_double(std::move(double_func)) {}

  void add_element(d_type znew) { add_element_force(znew); }

  void remove_element() {
    if (!z.empty()) {
      size_t n = z.size() - 1;
      size_t NN = current_max_len + extralen;

      for (size_t k = 1; k <= n; k++) {
        h[k - 1] = (h[k - 1] * (n - k + 1) - h[k] * (z[n] - z[n - k])) / n;
      }
      for (size_t k = n + 1; k <= NN; k++) {
        h[k - 1] -= h[k] * (z[n] - mu) / k;
      }
      z.pop_back();
    }
  }

  void print_state() const {
    std::cout << "Length=" << z.size() << " | Shift mu=" << mu << "\n";
    std::cout << "z = {";
    for (size_t i = 0; i < z.size(); ++i) {
      std::cout << z[i] << (i < z.size() - 1 ? ", " : "}\n");
    }
    std::cout << "DivDiffs (NN!-scaled) = {";
    for (size_t i = 0; i < z.size(); ++i) {
      std::cout << divdiff[i] << (i < z.size() - 1 ? ", " : "}\n");
    }
    std::cout << "---\n";
  }

  [[nodiscard]] auto back() const -> d_type { return divdiff[z.size() - 1]; }
  [[nodiscard]] auto size() const -> size_t { return z.size(); }
};
