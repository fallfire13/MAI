#include "complex.h"
#include <cmath>

#include <iostream>

void Complex::read(std::istream& is) {
  is >> real >> image;
}

void Complex::awrite(std::ostream& os) const {
  os << real << ' ' << image << '\n';
}

Complex Complex::add(const Complex& rhs) const {
  Complex d;
  d.real = real + rhs.real;
  d.image = image + rhs.image;
  return d;
}

Complex Complex::sub(const Complex& rhs) const {
  Complex d;
  d.real = real - rhs.real;
  d.image = image - rhs.image;
  return d;
}

Complex Complex::mul (const Complex& rhs) const {
  Complex d;
  d.real = real * rhs.real - image * rhs.image;
  d.image = real * rhs.image + rhs.real * image;
  return d;
}

Complex Complex::div (const Complex& rhs) const {
  Complex d;
  d.real = (real * rhs.real + image * rhs.image)/(rhs.real * rhs.real + rhs.image *rhs.image);
  d.image = (image * rhs.real - real * rhs.image)/(rhs.real * rhs.real + rhs.image *rhs.image);
  return d;
}  

int Complex::equ (const Complex& rhs) const {
  int result = 0;
  if (real == rhs.real && image == rhs.image) {
    result = 1;
  } else {
    result =0;
  }
  return result;
}

Complex Complex::conj () const {
  Complex d;
  d.real = real;
  d.image = image * -1;
  if (d.image==0) {
  d.image = image;
  } else {}
  return d;   
}    

int Complex::mod_comp (const Complex& rhs) const {
  int result = 0;
  double a = sqrt(real*real + image*image);
  double b = sqrt(rhs.real*rhs.real + rhs.image*rhs.image);

  if (a > b) {
    result = 1;
  } else if (a < b) {
    result =-1;
  } else {
    result =0;
  }
  return result;
}

