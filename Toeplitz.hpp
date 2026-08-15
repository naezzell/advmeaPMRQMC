#pragma once
#include <array>
#include <ostream>

template <size_t M> struct Toeplitz {
  std::array<double, M> coeffs;

  Toeplitz(double val = 0.0) : coeffs{} { coeffs[0] = val; }

  auto operator+=(const Toeplitz<M> &other) -> Toeplitz<M> & {
    for (size_t i = 0; i < M; ++i) {
      coeffs[i] += other.coeffs[i];
    }
    return *this;
  }
  auto operator-=(const Toeplitz<M> &other) -> Toeplitz<M> & {
    for (size_t i = 0; i < M; ++i) {
      coeffs[i] -= other.coeffs[i];
    }
    return *this;
  }
  auto operator*=(const Toeplitz<M> &other) -> Toeplitz<M> & {

    std::array<double, M> new_coeffs{};
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        new_coeffs[i] += coeffs[j] * other.coeffs[i - j];
      }
    }
    coeffs = new_coeffs;
    return *this;

    // this is slower, for some reason
    // std::array<double, M> new_coeffs{0};
    // for (size_t i = 0; i < M; ++i) {
    // 	const double c = coeffs[i];
    // 	for (size_t j = 0; j < M - i; ++j) {
    // 		new_coeffs[i + j] += c * other.coeffs[j];
    // 	}
    // }
    // coeffs = new_coeffs;
    // return *this;
  }
  auto operator*=(double scalar) -> Toeplitz<M> & {
    for (size_t i = 0; i < M; ++i) {
      coeffs[i] *= scalar;
    }
    return *this;
  }
  auto operator/=(double scalar) -> Toeplitz<M> & {
    const double reciprocal = 1.0 / scalar;
    for (size_t i = 0; i < M; ++i) {
      coeffs[i] *= reciprocal;
    }
    return *this;
  }

  auto operator==(const Toeplitz<M> &other) const -> bool = default;

  auto get(size_t ind) const -> double { return coeffs[ind]; }
  auto set(size_t ind, double value) { coeffs[ind] = value; }
};

template <size_t M>
auto operator+(const Toeplitz<M> &lhs, const Toeplitz<M> &rhs) -> Toeplitz<M> {
  auto cpy = lhs;
  return cpy += rhs;
}

template <size_t M>
auto operator-(const Toeplitz<M> &lhs, const Toeplitz<M> &rhs) -> Toeplitz<M> {
  auto cpy = lhs;
  return cpy -= rhs;
}

template <size_t M>
auto operator*(const Toeplitz<M> &lhs, const Toeplitz<M> &rhs) -> Toeplitz<M> {
  auto cpy = lhs;
  return cpy *= rhs;
}

template <size_t M>
auto operator*(const Toeplitz<M> &lhs, double rhs) -> Toeplitz<M> {
  auto cpy = lhs;
  return cpy *= rhs;
}

template <size_t M>
auto operator/(const Toeplitz<M> &lhs, double rhs) -> Toeplitz<M> {
  auto cpy = lhs;
  return cpy /= rhs;
}

template <size_t M> auto get_double(const Toeplitz<M> &elem) {
  return elem.coeffs[0];
}

template <size_t M>
auto operator<<(std::ostream &os, const Toeplitz<M> &elem) -> std::ostream & {
  os << "{";
  for (size_t i = 0; i < M; ++i) {
    os << elem.coeffs[i] << ",";
  }
  os << "}";
  return os;
}
