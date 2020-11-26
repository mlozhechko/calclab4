#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
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

  if (method == "QR") {
    BasicQREigen(A, B, solution, eps);
  } else if (method == "ShiftQR") {
    shiftQREigen(A, B, solution, eps);
  } else if (method == "HessQR") {
    HessenbergBasicQREigen(A, B, solution, eps);
  } else if (method == "HessShiftQR") {
    HessenbergShiftQREigen(A, B, solution, eps);
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

  ub::matrix<T> solR;
  ub::matrix<T> EV(A.size1(), 1);
  RayleighIteration(A, EV, solR, eps);

  std::cout << "rayleigh result: EV = " << EV << " sol = " << solR << std::endl;

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