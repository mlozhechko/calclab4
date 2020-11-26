#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <sstream>
#include "system_solver.h"
#include "matrix_utils.h"
#include "eigen_solver.h"

namespace ub = boost::numeric::ublas;

template <class T>
int lab4Main(int argc, char** argv) {
  bool isMatrixSpecified = cmdOptionExists(argv, argv + argc, "-matrix");
  if (!isMatrixSpecified) {
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

  ub::matrix<T> X;
  std::string method(getCmdOption(argv, argv + argc, "-method"));

  T eps = 1E-7;

  if (getCmdOption(argv, argv + argc, "-eps")) {
    eps = std::stod(getCmdOption(argv, argv + argc, "-eps"));
    std::cout << "eps is set to: " << eps << std::endl;
  }

  /*
   * calculation
   */
  ub::matrix<T> solution{};
  std::cout << "selected method: " << method << std::endl;
  std::cout << "source matrix A: " << A << std::endl;

  /*
   * B matrix is not initialized and not required for eigen values find algorithm
   * Interface was just copied from system_solver.h for linear systems.
   * It's better to be refactored
   */
  if (method == "QR") {
    BasicQREigen(A, B, solution, eps);
  } else if (method == "ShiftQR") {
    shiftQREigen(A, B, solution, eps);
  } else if (method == "HessQR") {
    HessenbergBasicQREigen(A, B, solution, eps);
  } else if (method == "HessShiftQR") {
    HessenbergShiftQREigen(A, B, solution, eps);
  } else if (method == "Rayleigh") {
    ub::matrix<T> EigenVectors;
    ub::matrix<T> EigenValues(A.size1(), 1);
    RayleighIteration(A, EigenValues, EigenVectors, eps);
    std::cout << "rayleigh result: EigenValues = " << EigenValues << " sol = " << EigenVectors << std::endl;
  } else {
    std::cout << "something went wrong" << std::endl;
    return -1;
  }

  std::cout << "Calculation info: " << std::endl;
  StatHolder::printInfo();
  StatHolder::reset();

  std::cout << "solution: " << solution << std::endl;
  ub::matrix<T> sv;
  inverseIteration(A, solution, sv, eps);
  std::cout << "eigen vectors: " << sv << std::endl;

  std::cout << "Inverse calculation info: " << std::endl;
  StatHolder::printInfo();
  StatHolder::reset();

  std::ostringstream stream{};
  ub::identity_matrix<T> I(A.size1(), A.size2());
  for (ssize_t i = 0; i < solution.size1(); ++i) {
    ub::matrix<T> As = A - (I * solution(i, 0));
    ub::matrix<T> Ev(A.size1(), 1);
    for (ssize_t j = 0; j < A.size1(); ++j) {
      Ev(j, 0) = sv(i, j);
    }

    ub::matrix<T> res;
    matrixMult(As, Ev, res);

    stream << std::sqrt(scalarMult(res, res)) << ", ";
  }

  std::cout << "values: " << stream.str() << std::endl;

  return 0;
}


int main(int argc, char** argv) {
  if (!cmdOptionExists(argv, argv + argc, "-precision")) {
    std::cerr << "precision is not specified" << std::endl;
    return -1;
  }

  std::string precision = getCmdOption(argv, argv + argc, "-precision");
  if (precision == "double") {
    return lab4Main<double>(argc, argv);
  } else if (precision == "float") {
    return lab4Main<float>(argc, argv);
  }

  std::cerr << "precision cannot be parsed correctly" << std::endl;
  return -2;
}

#pragma clang diagnostic pop