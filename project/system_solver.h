#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <numeric>
#include "utils.h"
#include "matrix_utils.h"

namespace ub = boost::numeric::ublas;

/*
 * synopsis:
 */

enum class stopCrit {
  error = 0,
  cont,
  stop
};

template <class T>
using stopCritType = std::function<stopCrit(const ub::matrix<T>&, const ub::matrix<T>&, const ub::matrix<T>&, T eps,
                             std::function<T(const ub::matrix<T>&)>, const ub::matrix<T>&)>;

template<class T>
[[maybe_unused]] int gaussSolve(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
               ub::matrix<T>& solution);

template <class T>
int initGivensCoefficients(const ub::matrix<T>& sourceMatrix, ssize_t row1, ssize_t row2,
                                  T& c, T& s);

template <class T>
int matrixTranspose(ub::matrix<T>& A);

template <class T>
int matrixFastRotate(ub::matrix<T>& A, ssize_t row1, ssize_t row2, T c, T s);

template<class U>
int QRDecomposition(const ub::matrix<U>& sourceMatrix, ub::matrix<U>& Q, ub::matrix<U>& R);

template<class T>
[[maybe_unused]] int QRSolve(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
            ub::matrix<T>& solution);

template <class T>
int matrixMult(const ub::matrix<T>& sourceMatrixA, const ub::matrix<T>& sourceMatrixB,
               ub::matrix<T>& result);

template <class T>
int invertMatrix(const ub::matrix<T>& A, ub::matrix<T>& result);

template <class T>
int invertDiagMatrix(const ub::matrix<T>& A, ub::matrix<T>& result);



template<class T>
stopCrit
ordinaryStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                         T eps, std::function<T(const ub::matrix<T>&)> norm,
                         const ub::matrix<T>& origSolution);

template<class T>
stopCrit
solutionStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                         T eps, std::function<T(const ub::matrix<T>&)> norm,
                         const ub::matrix<T>& origSolution);

template<class T>
stopCrit
deltaStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                         T eps, std::function<T(const ub::matrix<T>&)> norm,
                         const ub::matrix<T>& origSolution);

template<class T>
int fixedPointIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                        ub::matrix<T>& result, T tau, std::function<T(const ub::matrix<T>&)> norm,
                        T eps, const ub::matrix<T>& origSolution, stopCritType<T> stopCond);

template <class T>
int jacobiIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                    ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm, T eps,
                    const ub::matrix<T>& origSolution, stopCritType<T> stopCond);

template <class T>
int zeidelIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                    ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm, T eps,
                    const ub::matrix<T>& origSolution, stopCritType<T> stopCond);



template <class T>
int diag3RelaxationCalcC(const ub::matrix<T>& A3d, ub::matrix<T>& C, T w);

/*
 * in following method sourceMatrix is T(3 x N). Which contains values of three diagonals
 */
template <class T>
int diag3RelaxaionIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                            ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm,
                            T eps, T w, const ub::matrix<T>& origSolution,
                            stopCritType<T> stopCond);

template <class T>
T kEstimation(T q, T p0, T eps);

/*
 * implementation:
 */

template<class T>
[[maybe_unused]] int gaussSolve(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
               ub::matrix<T>& solution) {
  ub::matrix<T> A = sourceMatrix;
  ub::matrix<T> B = sourceVector;
  auto& X = solution;

  /*
   * ublas::matrix using linear memory model
   * therefore we cannot simply swap rows
   * and have to use transpositions matrix
   */
  ssize_t n = A.size1();
  std::vector<ssize_t> transposeVec(n);
  std::iota(transposeVec.begin(), transposeVec.end(), 0);
  auto& TR = transposeVec;

  for (ssize_t k = 0; k < n - 1; ++k) {
    T columnAbsMax = std::abs(A(TR[k], k));
    ssize_t columnAbsMaxIndex = k;
    for (ssize_t i = k + 1; i < n; ++i) {
      T absValue = std::abs(A(TR[i], k));
      if (absValue > columnAbsMax) {
        columnAbsMax = absValue;
        columnAbsMaxIndex = i;
      }
    }

    if (k != columnAbsMaxIndex) {
      std::swap(transposeVec[k], transposeVec[columnAbsMaxIndex]);
    }

    if (std::abs(A(TR[k], k)) < std::numeric_limits<T>::epsilon()) {
      std::cerr << "matrix det(A) == 0, system has infinite set of solutions" << std::endl;
      return -1;
    }

    for (ssize_t i = k + 1; i < n; ++i) {
      T c = A(TR[i], k) / A(TR[k], k);

      if (log::isDebug()) {
          A(TR[i], k) = 0;
      }

      for (ssize_t j = k + 1; j < n; ++j) {
        A(TR[i], j) -= c * A(TR[k], j);
      }
      B(TR[i], 0) -= c * B(TR[k], 0);

    }

    auto& logger = log::debug();
    logger << "cycle: " << k << "; TR: ";
    for (const auto& it: transposeVec) {
      logger << it << " ";
    }
    logger << "\n";
    log::debug() << "after gauss A: " << A << "\n";
    log::debug() << "after gauss B: " << B << "\n";
  }

  if (std::abs(A(TR[n - 1], n - 1)) < std::numeric_limits<T>::epsilon()) {
    std::cerr << "matrix det(A) == 0, system has infinite set of solutions" << std::endl;
    return -1;
  }

  auto& logger = log::debug();
  logger << "after gauss TR: ";
  for (const auto& it: transposeVec) {
    logger << it << " ";
  }
  logger << "\n";
  log::debug() << "after gauss A: " << A << "\n";
  log::debug() << "after gauss B: " << B << "\n";

  X.resize(n, 1);
  for (ssize_t i = n - 1; i >= 0; --i) {
    T ASum = 0;
    for (ssize_t j = n - 1; j > i; --j) {
      ASum += A(TR[i], j) * X(j, 0);
    }

    if (std::abs(A(TR[i], i)) < std::numeric_limits<T>::epsilon()) {
      std::cerr << "gauss calculation error" << std::endl;
    }

    X(i, 0) = (B(TR[i], 0) - ASum) / A(TR[i], i);
  }

  log::debug() << "result: " << X << "\n";

  return 0;
}

template<class T>
int initGivensCoefficients(const ub::matrix<T>& sourceMatrix, ssize_t row1, ssize_t row2,
                                  T& c, T& s) {
  const ub::matrix<T>& A = sourceMatrix;

  T denom = A(row1, row1) * A(row1, row1) + A(row2, row1) * A(row2, row1);
  denom = std::sqrt(denom);

  if (denom < std::numeric_limits<T>::epsilon()) {
    return -1;
  }

  c = A(row1, row1) / denom;
  s = A(row2, row1) / denom;

  return 0;
}


template<class T>
int matrixTranspose(ub::matrix<T>& A) {
  if (A.size1() != A.size2()) {
    std::cerr << "transpose of non-square matrices is not supported" << std::endl;
    return -1;
  }

  const ssize_t height = A.size1();
  const ssize_t width = A.size2();
  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = i + 1; j < width; ++j) {
      std::swap(A(i, j), A(j, i));
    }
  }
  return 0;
}

template<class T>
int matrixFastRotate(ub::matrix<T>& A, ssize_t row1, ssize_t row2, T c, T s) {
  ssize_t width = A.size2();
  for (size_t i = 0; i < width; ++i) {
    T tmp = A(row1, i) * c + A(row2, i) * s;
    A(row2, i) = A(row1, i) * (-s) + A(row2, i) * c;
    A(row1, i) = tmp;
  }

  return 0;
}

template<class U>
int QRDecomposition(const ub::matrix<U>& sourceMatrix, ub::matrix<U>& Q, ub::matrix<U>& R) {
  const ssize_t n = sourceMatrix.size1();

  ub::matrix<U> T = ub::identity_matrix(n, n);
  R = sourceMatrix;

  for (ssize_t i = 0; i < n - 1; ++i) {
    bool isDegenerateMatrix = true;

    for (ssize_t j = i + 1; j < n; ++j) {
      U c = 0, s = 0;
      log::debug() << "QR i: " << i << " j: " << j << "\n";
      log::debug() << "Q matrix " << T << "\n";
      log::debug() << "R matrix " << R << "\n";

      if (initGivensCoefficients(R, i, j, c, s) >= 0) {
        isDegenerateMatrix = false;

        log::debug() << "performing rotation c: " << c << " s: " << s << "\n";
        matrixFastRotate(T, i, j, c, s);
        matrixFastRotate(R, i, j, c, s);
      }
    }

    if (isDegenerateMatrix) {
      std::cerr << "matrix det == 0, QR decomposition cannot be completed" << std::endl;
      return -1;
    }
  }

  if (std::abs(R(n - 1, n - 1)) < std::numeric_limits<U>::epsilon()) {
    std::cerr << "matrix det == 0, QR decomposition cannot be completed" << std::endl;
    return -1;
  }

  log::debug() << "Q matrix " << T << "\n";
  log::debug() << "R matrix " << R << "\n";

  if (log::isDebug()) {
    auto QT = T;
    matrixTranspose(QT);
    ub::matrix<U> I;
    matrixMult(T, QT, I);
    log::debug() << "QT Q * Q^T = " << I << "\n";
  }

  matrixTranspose(T);
  Q = std::move(T);
  return 0;
}

template <class T>
[[maybe_unused]] int QRSolve(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
            ub::matrix<T>& solution) {
  ub::matrix<T> Q, R;
  if (QRDecomposition(sourceMatrix, Q, R) < 0) {
    std::cerr << "system cannot be solved with QR decomposition method" << std::endl;
    return -1;
  }

  log::debug() << "result of QR decomposition" << "\n";
  log::debug() << "Q: " << Q << "\n";
  log::debug() << "R: " << R << "\n";

  ub::matrix<T> Bs;
  matrixTranspose(Q);
  matrixMult(Q, sourceVector, Bs);

  ssize_t n = sourceVector.size1();
  ub::matrix<T>& X = solution;
  X.resize(n, 1);
  for (ssize_t i = n - 1; i >= 0; --i) {
    T ASum = 0;
    for (ssize_t j = n - 1; j > i; --j) {
      ASum += R(i, j) * X(j, 0);
    }

    if (std::abs(R(i, i)) < std::numeric_limits<T>::epsilon()) {
      std::cerr << "QR calculation error" << std::endl;
    }

    X(i, 0) = (Bs(i, 0) - ASum) / R(i, i);
  }

  return 0;
}

template <class T>
int matrixMult(const ub::matrix<T>& sourceMatrixA, const ub::matrix<T>& sourceMatrixB,
               ub::matrix<T>& result) {
  const ub::matrix<T>& A = sourceMatrixA;
  const ub::matrix<T>& B = sourceMatrixB;
  ub::matrix<T>& X = result;

  if (A.size2() != B.size1()) {
    log::debug() << "matrices cannot be multiplied" << "\n";
    return -1;
  }
  const ssize_t len = A.size2();
  const ssize_t height = A.size1();
  const ssize_t width = B.size2();

  X.resize(height, width);
  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = 0; j < width; ++j) {
      T val = 0;
      for (ssize_t k = 0; k < len; ++k) {
        val += A(i, k) * B(k, j);
      }
      X(i, j) = val;
    }
  }

  return 0;
}

template<class T>
int invertMatrix(const ub::matrix<T>& A, ub::matrix<T>& result) {
  if (A.size1() != A.size2()) {
    std::cerr << "non square matrices invert is not supported" << std::endl;
  }
  ssize_t n = A.size1();

  ub::matrix<T>& AI = result;
  AI = ub::zero_matrix<T>(n, n);

  ub::matrix<T> Q, R;
  QRDecomposition(A, Q, R);

  matrixTranspose(Q);

  for (ssize_t k = 0; k < n; ++k) {
    for (ssize_t i = n - 1; i >= 0; --i) {
      T ASum = 0;
      for (ssize_t j = n - 1; j > i; --j) {
        ASum += R(i, j) * AI(j, k);
      }

      if (std::abs(R(i, i)) < std::numeric_limits<T>::epsilon()) {
        std::cerr << "QR calculation error" << std::endl;
      }

      AI(i, k) = (Q(i, k) - ASum) / R(i, i);
    }
  }

  if (log::isDebug()) {
    ub::matrix<T> multRes;
    matrixMult(A, AI, multRes);

    log::debug() << "invert of matrix result: " << multRes << "\n";
  }

  return 0;
}

template<class T>
int invertDiagMatrix(const ub::matrix<T>& A, ub::matrix<T>& result) {
  ub::matrix<T>& R = result;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  if (height != width) {
    log::debug() << "matrix is not diag and cannot be inverted" << "\n";
    return -1;
  }
  R = ub::zero_matrix<T>(height, width);

  for (ssize_t i = 0; i < height; ++i) {
    T elem = A(i, i);
    if (std::abs(elem) <= std::numeric_limits<T>::epsilon()) {
      log::debug() << "matrix has det == 0 and cannot be inverted" << "\n";
      return -2;
    }

    R(i, i) = 1. / elem;
  }
  return 0;
}

template<class T>
stopCrit
ordinaryStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                         T eps, std::function<T(const ub::matrix<T>&)> norm,
                         const ub::matrix<T>& origSolution) {
  T normC = norm(C);
  if (normC <= std::numeric_limits<T>::epsilon()) {
    return stopCrit::error;
  }

  log::debug() << "norm C: " << normC << "\n";

  ub::matrix<T> deltaX = X - prevX;

  log::debug() << "deltaX: " << deltaX << " norm(deltaX): " << norm(deltaX) << "\n";
  bool result = norm(deltaX) <= (1.0 - normC) / normC * eps;
  if (result) {
    return stopCrit::stop;
  }

  return stopCrit::cont;
}

template<class T>
stopCrit
solutionStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                         T eps, std::function<T(const ub::matrix<T>&)> norm,
                         const ub::matrix<T>& origSolution) {

  if (X.size1() != origSolution.size1()) {
    return stopCrit::error;
  }

  ub::matrix<T> deltaX = X - origSolution;
  T deltaNorm = norm(deltaX);
  if (deltaNorm < eps) {
    return stopCrit::stop;
  }

  return stopCrit::cont;
}

template<class T>
stopCrit
deltaStoppingCriteria(const ub::matrix<T>& X, const ub::matrix<T>& prevX, const ub::matrix<T>& C,
                      T eps, std::function<T(const ub::matrix<T>&)> norm,
                      const ub::matrix<T>& origSolution) {
  if (prevX.size1() != X.size1()) {
    return stopCrit::error;
  }

  ub::matrix<T> deltaX = X - prevX;
  T normDeltaX = norm(deltaX);

  if (normDeltaX < eps) {
    return stopCrit::stop;
  }

  return stopCrit::cont;
}

template<class T>
int fixedPointIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                        ub::matrix<T>& result, T tau, std::function<T(const ub::matrix<T>&)> norm,
                        T eps, const ub::matrix<T>& origSolution, stopCritType<T> stopCond) {
  log::debug() << "enter fpi method" << "\n";
  const ub::matrix<T>& A = sourceMatrix;
  const ub::matrix<T>& B = sourceVector;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  if (height != width) {
    log::debug() << "fixed points iteration matrix is not square \n";
    return -1;
  }

  ub::matrix<T> E = ub::identity_matrix(height, width);

  ub::matrix<T> C = -(A * tau - E);
  ub::matrix<T> Y = B * tau;

  ub::matrix<T> X = Y; //ub::zero_matrix(height, 1);
  ub::matrix<T> prevX = ub::zero_matrix(height, 1);

  std::cout << "fpi norm(C): " << norm(C) << std::endl;

  if (norm(C) > 1) {
    std::cout << "C: " << C << std::endl;
    std::cerr << "norm(C) >= 1, system can not be solved" << std::endl;
    return -2;
  }

  ssize_t itc = 0;

  stopCrit status = stopCrit::cont;
  do {
    prevX = X;
    matrixMult(C, prevX, X);
    X = X + Y;
    log::debug() << "new X = " << X << "\n";
    status = stopCond(X, prevX, C, eps, norm, origSolution);

    ++itc;
    if (itc == 1) {
      T p0 = norm(prevX - X);
      T kEst = kEstimation(norm(C), p0, eps);
      std::cout << "k estimation. k > " << kEst << std::endl;
    }

  } while (stopCrit::cont == status);

  std::cout << "number of iterations " << itc << std::endl;

  if (stopCrit::error == status) {
    return -1;
  }

  result = X;
  return 0;
}

template<class T>
int jacobiIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                    ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm, T eps,
                    const ub::matrix<T>& origSolution, stopCritType<T> stopCond) {
  ub::matrix<T> A = sourceMatrix;
  ub::matrix<T> B = sourceVector;

  ssize_t height = A.size1();
  ssize_t width = A.size2();
  if (height != width) {
    return -1;
  }

  ub::matrix<T> X = ub::zero_matrix(height, 1);
  ub::matrix<T> prevX = ub::zero_matrix(height, 1);

  ub::matrix<T> C = ub::zero_matrix(height, width);
  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = 0; j < width; ++j) {
      if (i == j) {
        continue;
      }
      C(i, j) = -A(i, j) / A(i, i);
    }
  }

  const T nc = norm(C);
  std::cout << "jacobi norm(C): " << nc << " >= 1" << std::endl;
  if (nc > 1) {
    std::cout << "C: " << C << std::endl;
    std::cerr << "norm(C) >= 1. system can not be solved" << std::endl;
    return -1;
  }

  ssize_t itc = 0;

  stopCrit status = stopCrit::cont;
  do {
    prevX = X;
    for (ssize_t i = 0; i < height; ++i) {
      T acc = 0;
      for (ssize_t j = 0; j < width; ++j) {
        if (i == j) {
          continue;
        }
        acc += C(i, j) * prevX(j, 0);
      }
      acc += B(i, 0) / A(i, i);
      X(i, 0) = acc;
    }
    status = stopCond(X, prevX, C, eps, norm, origSolution);

    ++itc;
    if (itc == 1) {
      T estK = kEstimation(norm(C), norm(X - prevX), eps);
      std::cout << "k estimation: " << estK << std::endl;
    }

  } while (stopCrit::cont == status);

  std::cout << "number of iterations: " << itc << std::endl;

  if (status == stopCrit::error) {
    return -1;
  }

  result = X;
  return 0;
}

template<class T>
int zeidelIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                    ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm, T eps,
                    const ub::matrix<T>& origSolution, stopCritType<T> stopCond) {
  const ub::matrix<T>& A = sourceMatrix;
  const ub::matrix<T>& B = sourceVector;

  ssize_t height = A.size1();
  ssize_t width = A.size2();

  if (height != width) {
    return -1;
  }

  ub::matrix<T> E = ub::identity_matrix(height, width);

  ub::matrix<T> L = ub::zero_matrix(height, width);
  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = 0; j < i; ++j) {
      L(i, j) = A(i, j);
    }
  }
  ub::matrix<T> D = ub::zero_matrix(height, width);
  for (ssize_t i = 0; i < height; ++i) {
    D(i, i) = A(i, i);
  }
  ub::matrix<T> U = ub::zero_matrix(height, width);
  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = i + 1; j < width; ++j) {
      U(i, j) = A(i, j);
    }
  }

  ub::matrix<T> DI;
  int res = invertDiagMatrix(D, DI);
  if (res < 0) {
    std::cerr << "invert diag matrix error, res: " << res << std::endl;
    return -1;
  }

  ub::matrix<T> DIL;
  matrixMult(DI, L, DIL);

  DIL = E + DIL;
  ub::matrix<T> DILI;
  invertMatrix(DIL, DILI);
  // DILI == (E + D^(-1) * L)^(-1)

  ub::matrix<T> DIU;
  matrixMult(DI, U, DIU);
  DIU = -DIU;
  // DIU == (-D^(-1) * U)

  ub::matrix<T> C;
  matrixMult(DILI, DIU, C);
  // C == DILI * DIU == (E + D^(-1) * L)^(-1) * (-D^(-1) * U)

  ub::matrix<T> X = ub::zero_matrix(height, 1);
  ub::matrix<T> prevX = X;

  std::cout << "seidel method norm(C): " << norm(C) << std::endl;

  if (norm(C) >= 1) {
    std::cerr << "norm(C) = " << norm(C) << " >= 1" << std::endl;
    std::cerr << "system cannot be solved" << std::endl;
    return -1;
  }

  size_t itc = 0;

  stopCrit status = stopCrit::cont;
  do {
    prevX = X;
    for (ssize_t i = 0; i < height; ++i) {
      T acc = 0;

      for (ssize_t j = 0; j < i; ++j) {
        acc += -A(i, j) / A(i, i) * X(j, 0);
      }

      for (ssize_t j = i + 1; j < height; ++j) {
        acc += -A(i, j) / A(i, i) * prevX(j, 0);
      }

      acc += B(i, 0) / A(i, i);
      X(i, 0) = acc;
    }
    ++itc;

    if (itc == 1) {
      T estK = kEstimation(norm(C), norm(X - prevX), eps);
      std::cout << "esitmated k " << estK << std::endl;
    }
    status = stopCond(X, prevX, C, eps, norm, origSolution);
  } while (stopCrit::cont == status);

  std::cout << "number of iterations: " << itc << std::endl;

  if (status == stopCrit::error) {
    return -1;
  }

  result = X;
  return 0;
}

template<class T>
int diag3RelaxationCalcC(const ub::matrix<T>& A3d, ub::matrix<T>& C, T w) {
  ssize_t n = A3d.size1();
  ub::matrix<T> A = ub::zero_matrix(n, n);

  A(0, 0) = A3d(0, 1);
  A(0, 1) = A3d(0, 2);

  for (ssize_t i = 1; i < n - 1; ++i) {
    A(i, i - 1) = A3d(i, 0);
    A(i, i) = A3d(i, 1);
    A(i, i + 1) = A3d(i, 2);
  }

  A(n - 1, n - 2) = A3d(n - 1, 0);
  A(n - 1, n - 1) = A3d(n - 1, 1);

  ub::matrix<T> L = ub::zero_matrix(n, n);
  for (ssize_t i = 0; i < n; ++i) {
    for (ssize_t j = 0; j < i; ++j) {
      L(i, j) = A(i, j);
    }
  }
  ub::matrix<T> D = ub::zero_matrix(n, n);
  for (ssize_t i = 0; i < n; ++i) {
    D(i, i) = A(i, i);
  }
  ub::matrix<T> U = ub::zero_matrix(n, n);
  for (ssize_t i = 0; i < n; ++i) {
    for (ssize_t j = i + 1; j < n; ++j) {
      U(i, j) = A(i, j);
    }
  }

  ub::matrix<T> DI;
  invertDiagMatrix(D, DI);

  ub::matrix<T> wDI;
  wDI = DI * w;
  ub::matrix<T> E = ub::identity_matrix(n, n);
  ub::matrix<T> wDIL;
  matrixMult(wDI, L, wDIL);
  wDIL = wDIL + E;
  ub::matrix<T> wDILI;
  invertMatrix(wDIL, wDILI);

  ub::matrix<T> mwE = E * (1 - w);
  ub::matrix<T> wDIU;
  matrixMult(wDI, U, wDIU);
  wDIU = mwE - wDIU;

  matrixMult(wDILI, wDIU, C);

  return 0;
}

template<class T>
int diag3RelaxaionIteration(const ub::matrix<T>& sourceMatrix, const ub::matrix<T>& sourceVector,
                            ub::matrix<T>& result, std::function<T(const ub::matrix<T>&)> norm,
                            T eps, T w, const ub::matrix<T>& origSolution,
                            stopCritType<T> stopCond) {
  ssize_t height = sourceMatrix.size1();
  ub::matrix<T> X = ub::zero_matrix(height, 1);
  ub::matrix<T> prevX = X;

  ub::matrix<T> A = sourceMatrix;
  ub::matrix<T> B = sourceVector;
  ub::matrix<T> C;

  diag3RelaxationCalcC(A, C, w);
  std::cout << "norm(C): " << norm(C) << std::endl;

  ssize_t itc = 0;

  stopCrit status = stopCrit::cont;
  do {
    prevX = X;

    X(0, 0) = (1 - w) * prevX(0, 0) - w * A(0, 2) / A(0, 1) * prevX(1, 0) + w * B(0, 0) / A(0, 1);

    for (ssize_t i = 1; i < height - 1; ++i) {
      X(i, 0) = 0;
      X(i, 0) += -w * A(i, 0) / A(i, 1) * X(i - 1, 0 );
      X(i, 0) += (1 - w) * prevX(i, 0);
      X(i, 0) += -w * A(i, 2) / A(i, 1) * prevX(i + 1, 0);
      X(i, 0) += w * B(i, 0) / A(i, 1);
    }

    ssize_t last = height - 1;
    X(last, 0) = -w * A(last, 0) / A(last, 1) * X(last - 1, 0) + (1 - w) * prevX(last, 0) + w * B(last, 0) / A(last, 1);

    status = stopCond(X, prevX, C, eps, norm, origSolution);
    ++itc;

    if (itc == 1) {
      T estK = kEstimation(norm(C), norm(X - prevX), eps);
      std::cout << "est K: " << estK << std::endl;
    }
  } while (stopCrit::cont == status);

  std::cout << "number of iterations: " << itc << std::endl;

  if (status == stopCrit::error) {
    return -1;
  }

  result = X;

  return 0;
}

template <class T>
T kEstimation(T q, T p0, T eps) {
//  std::cout << q << " " << p0 << " " << eps << std::endl;

//  std::cout << eps * (1 - q) / p0 << std::endl;

  T up = std::log<T>(eps * (1 - q) / p0).real();
  T down = std::log<T>(q).real();

//  std::cout << up << " " << down << std::endl;
  return up / down;
}