#pragma once
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>

class log {
public:
  log() = default;

  static log& debug(const std::string& msg = "[debug] ") {
    static log ins;
    if (ins.isEnabled) {
      std::cout << msg;
    }
    return ins;
  }

  template<class T>
  log& operator<<(const T& obj) {
    if (isEnabled) {
      std::cout << obj;
    }
    return *this;
  }

  static void setEnabled(bool mode) {
    log::debug("").isEnabled = mode;
  }

  static void setPrecision(int number) {
    std::cout.precision(number);
  }

  static bool isDebug() {
    return debug("").isEnabled;
  }

private:
  bool isEnabled = false;
};

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

namespace ub = boost::numeric::ublas;

template <class T>
int initMatrix(ub::matrix<T>& matrix, std::ifstream& stream) {
  ssize_t height = 0, width = 0;
  stream >> height;
  width = height;
  matrix.resize(height, width);

  for (ssize_t i = 0; i < height; ++i) {
    for (ssize_t j = 0; j < width; ++j) {
      stream >> matrix(i, j);
    }
  }

  return 0;
}

template <class T>
int initVector(ub::matrix<T>& matrix, std::ifstream& stream) {
  ssize_t height = 0;
  stream >> height;
  matrix.resize(height, 1);

  for (ssize_t i = 0; i < height; ++i) {
    stream >> matrix(i, 0);
  }

  return 0;
}