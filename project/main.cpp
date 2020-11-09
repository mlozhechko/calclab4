#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "system_solver.h"
#include "matrix_utils.h"

namespace ub = boost::numeric::ublas;

template <class T>
int lab2Main(int argc, char** argv) {
  bool isMatrixSpecified = cmdOptionExists(argv, argv + argc, "-matrix");
  bool isVectorSpecified = cmdOptionExists(argv, argv + argc, "-vector");
  if (!isMatrixSpecified || !isVectorSpecified) {
    std::cerr << "source data is not specified" << std::endl;
    return -1;
  }

  if (cmdOptionExists(argv, argv + argc, "-debug")) {
    std::cout << "debug mode enabled" << std::endl;
    log::setEnabled(true);
    log::setPrecision(5);
  }

  if (!cmdOptionExists(argv, argv + argc, "-method")) {
    std::cerr << "solver method is not specified" << std::endl;
    return -2;
  }

  std::ifstream matrixStream(getCmdOption(argv, argv + argc, "-matrix"));
  std::ifstream vectorStream(getCmdOption(argv, argv + argc, "-vector"));

  ub::matrix<T> A, B;
  initMatrix(A, matrixStream);
  initVector(B, vectorStream);

  std::cout << "source A: " << A << std::endl;
  std::cout << "source B: " << B << std::endl;

  ub::matrix<T> X;
  std::string method(getCmdOption(argv, argv + argc, "-method"));

  /*
   * calculation
   */
  T eps = 1E-7;

  if (getCmdOption(argv, argv + argc, "-eps")) {
    eps = std::stod(getCmdOption(argv, argv + argc, "-eps"));
    std::cout << "eps is set to: " << eps << std::endl;
  }
  std::function<T(const ub::matrix<T>&)> norm = normCubic<T>;

  ub::matrix<T> origSolution;
  if (cmdOptionExists(argv, argv + argc, "-solution")) {
    std::ifstream solutionStream(getCmdOption(argv, argv + argc, "-solution"));
    initVector(origSolution, solutionStream);
  }

  if(!cmdOptionExists(argv, argv + argc, "-criteria")) {
    std::cout << "stopping criteria has not been selected" << std::endl;
    return -1;
  }
  std::string criteriaSelectorString = getCmdOption(argv, argv + argc, "-criteria");

  stopCritType<T> stopCrit;
  if (criteriaSelectorString == "ordinary") {
    stopCrit = ordinaryStoppingCriteria<T>;
  } else if (criteriaSelectorString == "solution") {
    stopCrit = solutionStoppingCriteria<T>;
  } else if (criteriaSelectorString == "delta") {
    stopCrit = deltaStoppingCriteria<T>;
  } else {
    std::cout << "stopping criteria has not been carefully selected" << std::endl;
    return -2;
  }

  if (method == "fpi") {
    std::cout << "using fixed point iteration method" << std::endl;
    /*
     * fpi works correctly with 6 test with 0.25 tau and Cubic norm
     * with this conditions norm(C) <= 1 (0.95)
     */
//    T delta = 0.001;
//    T tau = delta;
//
//    T minTau = -1;
//    T maxNorm = 1;
//    std::cout << " norm A " << norm(A) << std::endl;
//
//    for (; tau < 0.2; tau += delta) {
//      ssize_t height = A.size1();
//      ssize_t width = A.size2();
//      ub::matrix<T> E = ub::identity_matrix(height, width);
//      ub::matrix<T> C = -(A * tau - E);
//
//      std::cout << norm(C) << " " << tau << std::endl;
//      if (norm(C) < maxNorm) {
//        maxNorm = norm(C);
//        minTau = tau;
//      }
//    }
    T tau = 0.001;

    if (tau < 0) {
      std::cout << "norm(C) >= 1. system can not be calculated properly" << std::endl;
      return -7;
    }

    std::cout << "minimal norm C tau is: " << tau << std::endl;

    if (fixedPointIteration(A, B, X, tau, norm, eps, origSolution, stopCrit) < 0) {
      return -5;
    }
  } else if (method == "jacobi") {
    std::cout << "using jacobi method" << std::endl;
    if (jacobiIteration(A, B, X, norm, eps, origSolution, stopCrit) < 0) {
      return -1;
    }
  } else if (method == "seidel") {
    std::cout << "using seidel method" << std::endl;
    if (zeidelIteration(A, B, X, norm, eps, origSolution, stopCrit) < 0) {
      return -1;
    }
  } else if (method == "relax3d") {
    std::cout << "using relaxation method (3-diagonal matrices case)" << std::endl;
    std::cout << "warning(!) source matrices will be redefined" << std::endl;

    const T w = 1;
    const ssize_t N = 213;
    A = ub::zero_matrix<T>(N, 3);
    B = ub::zero_matrix<T>(N, 1);

    A(0, 1) = 4;
    A(0, 2) = 1;

    for (ssize_t i = 0; i < N - 1; ++i) {
      A(i, 0) = 1;
      A(i, 1) = 4;
      A(i, 2) = 1;
    }
    A(N - 1, 0) = 1;
    A(N - 1, 1) = 4;

    B(0, 0) = 6;
    for (ssize_t i = 1; i < N - 1; ++i) {
      B(i, 0) = 10 - 2 * ((i + 1) % 2);
    }
    B(N - 1, 0) = 9 - 3 * (N % 2);

    if (diag3RelaxaionIteration(A, B, X, norm, eps, w, origSolution, stopCrit) < 0) {
      return -1;
    }
  } else {
      std::cerr << "solver method cannot be parsed" << std::endl;
      return -3;
  }

  std::cout << "result is X = " << X << std::endl;

  if (cmdOptionExists(argv, argv + argc, "-solution")) {
    std::cout << "original solution delta " << X - origSolution << std::endl;
    std::cout << "norm delta " << norm(X - origSolution) << std::endl;
  }

  ub::matrix<T> NRes;
  matrixMult(A, X, NRes);
  std::cout << "residual: " << NRes - B << std::endl;
  std::cout << "norm residual: " << norm(NRes - B) << std::endl;

  return 0;
}


int main(int argc, char** argv) {
  if (!cmdOptionExists(argv, argv + argc, "-precision")) {
    std::cerr << "precision is not specified" << std::endl;
    return -1;
  }

  std::string precision = getCmdOption(argv, argv + argc, "-precision");
  if (precision == "double") {
    return lab2Main<double>(argc, argv);
  } else if (precision == "float") {
    return lab2Main<float>(argc, argv);
  }

  std::cerr << "precision cannot be parsed correctly" << std::endl;
  return -2;
}

#pragma clang diagnostic pop