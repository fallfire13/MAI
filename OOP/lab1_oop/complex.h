#ifndef D_COMPLEX_H
#define D_COMPLEX_H

#include <iostream>

struct Complex {
private:
  double real; 
  double image; 
  
public:
  void read(std::istream& is);
  void awrite(std::ostream& os) const;

  Complex add(const Complex& rhs) const; 
  Complex sub(const Complex& rhs) const;
  Complex mul(const Complex& rhs) const;
  Complex div(const Complex& rhs) const;
  Complex conj() const;
  
  int mod_comp(const Complex& rhs) const;
  int equ(const Complex& rhs) const;
  
};

#endif