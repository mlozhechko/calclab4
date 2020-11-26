#pragma once

#include <iostream>

class StatHolder {
public:
  static void countIteration(ssize_t n = 1) {
    instance().m_iterationsCount += n;
  }

  static void countMultiplication(ssize_t n = 1) {
    instance().m_multiplicationsCount += n;
  }

  static void printInfo() {
    std::cout << "mulitplications: " << instance().m_multiplicationsCount
      << " iterations: " << instance().m_iterationsCount << std::endl;
  }

  static void reset() {
    instance().m_iterationsCount = 0;
    instance().m_multiplicationsCount = 0;
  }
private:
  static StatHolder& instance() {
    static StatHolder statHolder;
    return statHolder;
  }

  size_t m_iterationsCount = 0;
  size_t m_multiplicationsCount = 0;
};