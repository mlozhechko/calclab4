#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <numeric>

namespace ub = boost::numeric::ublas;

/*
 * synopsis
 */

template <class T>
T normCubic(const ub::matrix<T>& A);

template <class T>
T normOcta(const ub::matrix<T>& A);

/*
 * implementation
 */

template<class T>
T normCubic(const ub::matrix<T>& A) {
  ssize_t height = A.size1();
  ssize_t width = A.size2();

  T rowMax = 0;
  for (ssize_t i = 0; i < height; ++i) {
    T currentSum = 0;
    for (ssize_t j = 0; j < width; ++j) {
      currentSum += std::abs(A(i, j));
    }

    if (currentSum > rowMax) {
      rowMax = currentSum;
    }
  }

  return rowMax;
}

template<class T>
T normOcta(const ub::matrix<T>& A) {
  ssize_t height = A.size1();
  ssize_t width = A.size2();

  T columnMax = 0;
  for (ssize_t j = 0; j < width; ++j) {
    T currentMax = 0;
    for (ssize_t i = 0; i < height; ++i) {
      currentMax += std::abs(A(i, j));
    }

    if (currentMax > columnMax) {
      columnMax = currentMax;
    }
  }

  return columnMax;
}

