#pragma once

#include "system_solver.h"

/*
 * synopsis
 */
template <class T>
int BasicQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                 ub::matrix<T>& solution, T eps);

template <class T>
int shiftQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                 ub::matrix<T>& solution, T eps);

template <class T>
int HessenbergBasicQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                 ub::matrix<T>& solution, T eps);

template <class T>
int HessenbergShiftQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                           ub::matrix<T>& solution, T eps);

template <class T>
T scalarMult(const ub::matrix<T>& A, const ub::matrix<T>& B);

template <class T>
int inverseIteration(const ub::matrix<T>& A, const ub::matrix<T>& eigenVector, ub::matrix<T>& solution, T eps);

template <class T>
int RayleighIteration(const ub::matrix<T>& A, ub::matrix<T>& eigenVector, ub::matrix<T>& solution, T eps);

/*
 * implementation
 */
template <class T>
int BasicQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                 ub::matrix<T>& solution, T eps) {
  ub::matrix<T> A = sourceMatrix;
  ub::matrix<T>& X = solution;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  if (height != width) {
    return -1;
  }

  ub::matrix<T> Q, Qk, R;

  Qk = ub::zero_matrix<T>(height, width);
  ub::matrix<T> Ak(height, width);

  ub::matrix<T> AC = ub::zero_matrix<T>(height, width);

  Ak = A;
  do {
    Q = Qk;
    QRDecomposition(Ak, Qk, R);
    matrixMult(R, Qk, Ak);

    StatHolder::countIteration();

    for (ssize_t i = 1; i < height; ++i) {
      for (ssize_t j = 0; j < i; ++j) {
        AC(i, j) = Ak(i, j);
      }
    }
  } while (normCubic<T>(AC) > eps);

  X = ub::matrix<T>(height, 1);

  for (ssize_t i = 0; i < height; ++i) {
    X(i, 0) = Ak(i, i);
  }

  return 0;
}

template <class T>
static T shiftQREightCycle(ub::matrix<T>&A, ssize_t n, T eps) {
  ub::matrix<T>& Aint = A;
  Aint.resize(n, n);

  ub::matrix<T> lastA;

  ub::matrix<T> Q, R;
  ub::identity_matrix<T> I(n, n);

  do {
    T sigma = Aint(n - 1, n - 1);
    Aint = Aint - (I * sigma);

    if (QRDecomposition(Aint, Q, R) < 0) {
      std::cout << Aint << std::endl;
      throw std::runtime_error("qr failure");
    }

    matrixMult(R, Q, Aint);
    Aint += I * sigma;
    lastA.resize(1, Aint.size2() - 1);
    for (size_t i = 0; i < Aint.size2() - 1; ++i) {
      lastA(0, i) = Aint(n - 1, i);
    }
    StatHolder::countIteration();
  } while (normCubic(lastA) > eps);

  return Aint(n - 1, n -1);
}

template <class T>
int shiftQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                 ub::matrix<T>& solution, T eps) {
  T sigma = 1;

  ub::matrix<T> A = sourceMatrix;
  ub::matrix<T>& X = solution;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  if (height != width) {
    return -1;
  }

  solution.resize(height, 1);
  for (ssize_t n = height; n > 1; --n) {
    X(n - 1, 0) = shiftQREightCycle(A, n, eps);
  }
  X(0, 0) = A(0, 0);

  return 0;
}

template <class T>
static bool reduceToHessenbergForm(const ub::matrix<T>& sourceMatrix, ub::matrix<T>& result) {
  result = sourceMatrix;
  ub::matrix<T>& A = result;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  for (ssize_t j = 0; j < width - 2; ++j) {
    for (ssize_t i = j + 2; i < height; ++i) {
      T c = 0;
      T s = 0;

      if (initHessGivensCoefficients(A, j + 1, i, c, s) >= 0) {
        matrixFastRotate(A, j + 1, i, c, s);
        matrixFastRotateRight(A, j + 1, i, c, -s);
      }
    }
  }

  return true;
}

template <class T>
int HessenbergBasicQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                           ub::matrix<T>& solution, T eps) {
  ub::matrix<T> H;
  reduceToHessenbergForm(sourceMatrix, H);

  std::cout << "Reduction to Hessenberg form stat:" << std::endl;
  StatHolder::printInfo();
  StatHolder::reset();

  return BasicQREigen<T>(H, sourceVector, solution, eps);
}

template <class T>
int HessenbergShiftQREigen(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                           ub::matrix<T>& solution, T eps) {
  ub::matrix<T> H;
  reduceToHessenbergForm(sourceMatrix, H);

  std::cout << "Reduction to Hessenberg form stat:" << std::endl;
  StatHolder::printInfo();
  StatHolder::reset();

  return shiftQREigen<T>(H, sourceVector, solution, eps);
}

template <class T>
int inverseIteration(const ub::matrix<T>& A, const ub::matrix<T>& eigenVector, ub::matrix<T>& solution, T eps) {
  ssize_t height = eigenVector.size1();

  solution.resize(height, height);

  const ub::matrix<T>& EV = eigenVector;

  ub::identity_matrix<T> I(height, height);
  ub::matrix<T> res;

  for (ssize_t i = 0; i < height; ++i) {
    T ev = EV(i, 0);

    ub::matrix<T> Aev = A - I * ev;
    ub::matrix<T> AI;
    invertMatrix(Aev, AI);

    ub::matrix<T> X(height, 1);
    for (ssize_t j = 0; j < height; ++j) {
      X(j, 0) = 1;
    }
    ub::matrix<T> prevX = X;

    size_t n = 0;

    T prevCoeff = 0;
    T coeff = 0;
    do {
      prevCoeff = coeff;
      prevX = X;
      matrixMult(AI, prevX, X);
//      std::cout << X << std::endl;
      coeff = normCubic<T>(X);
      if (std::abs(coeff) < std::numeric_limits<T>::epsilon()) {
        std::cerr << "coeff is 0" << std::endl;
        return -1;
      }

      X = X * (1. / coeff);

      StatHolder::countIteration();

      ub::matrix<T> tmp = A - (ev * I);
      matrixMult(tmp, X, res);
    } while (std::sqrt(scalarMult(res, res)) > eps);

    for (size_t j = 0; j < height; ++j) {
      solution(i, j) = X(j, 0);
    }
  }

  return 0;
}

template <class T>
T scalarMult(const ub::matrix<T>& A, const ub::matrix<T>& B) {
  if ((A.size1() != B.size1()) || (A.size2() == 0) || (B.size2() == 0)) {
    throw std::runtime_error("scalar multiplication cannot be calculated");
  }

  ssize_t height = A.size1();

  T acc = 0;
  for (ssize_t i = 0; i < height; ++i) {
    acc += A(i, 0) * B(i, 0);
  }

  return acc;
}

template <class T>
int RayleighIteration(const ub::matrix<T>& A, ub::matrix<T>& eigenVector, ub::matrix<T>& solution, T eps) {
  ssize_t height = eigenVector.size1();
  solution.resize(height, height);
  size_t n = 0;
  const ub::identity_matrix<T> I(height, height);

  for (ssize_t i = 0; i < height; ++i) {
    ub::matrix<T> X(height, 1);
    for (size_t j = 0; j < height; ++j) {
      X(j, 0) = 0;
    }
    X(i, 0) = 1;

    ub::matrix<T> prevX;
    T ev = 0;

    ub::matrix<T> AD;
    do {
      prevX = X;
      ub::matrix<T> AM(height, 1);
      matrixMult(A, X, AM);
      ev = scalarMult(AM, X);
      ub::matrix<T> Aev = A - (I * ev);
      ub::matrix<T> Y;
      const int res = gaussSolve(Aev, X, Y);
      if (res < 0) {
        break;
      }

      const T coeff = std::sqrt(scalarMult(Y, Y));
      X = Y * (1. / coeff);
      ++n;

      ub::matrix<T> AX;
      matrixMult(A, X, AX);
      AD = AX - X * ev;
    } while (std::sqrt(scalarMult(AD, AD)) > eps);

    for (size_t j = 0; j < height; ++j) {
      solution(i, j) = X(j, 0);
    }
    eigenVector(i, 0) = ev;
  }

  std::cout << "Rl " << n << std::endl;

  return 0;
}