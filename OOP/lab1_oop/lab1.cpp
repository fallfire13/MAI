#include <iostream>

#include "complex.h"

int main() {
  Complex a;
  Complex b;
  //std::cout << "Enter numbers:\n[real] and [image]" << std::endl;
  a.read(std::cin);
  b.read(std::cin);
  //std::cout << "Real:\n";
  //std::cout << "Image:\n";
  std::cout << "SUM:\n";
  a.add(b).awrite(std::cout);
  std::cout << "SUB:\n";
  a.sub(b).awrite(std::cout);
  std::cout << "MULTIPLY:\n";
  a.mul(b).awrite(std::cout);
  std::cout << "DIVIDE:\n";
  a.div(b).awrite(std::cout);
  std::cout << "EQUAL:\n";
  
switch (a.equ(b)) {
  case 0:
    std::cout << "Numbers are not equal" << std::endl;
    break;

  case 1:
    std::cout << "Numbers are equal" << std::endl;
    break;
}

  std::cout << "Conjugate number for a:\n";
  a.conj().awrite(std::cout);
  std::cout << "Module compare for a and b:\n";

switch (a.mod_comp(b)) {
  case 1:
    std::cout << "Module a more module b" << std::endl;
    break;

  case -1:
    std::cout << "Module a less than module b" << std::endl;
    break;

  case 0:
    std::cout << "Module a is equal to module b" << std::endl;
    break;
}


  return 0;
}