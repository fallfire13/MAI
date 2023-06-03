#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

const int BASE = 10000;
const int BASE_POW = 4;

class TLongInt {
public:
    TLongInt() {};
    TLongInt(std::string&);
    TLongInt(int n);
    ~TLongInt() {};

    TLongInt operator+(const TLongInt&);
    TLongInt operator-(const TLongInt&);
    TLongInt operator*(const TLongInt&) const;
    TLongInt operator/(const TLongInt&);
    TLongInt power(int r);
    bool operator==(const TLongInt&) const;
    bool operator<(const TLongInt&) const;
    bool operator>(const TLongInt&) const;
    bool operator<=(const TLongInt&) const;
    size_t size() const { return m_data.size(); }
    friend std::ostream& operator<<(std::ostream&, const TLongInt&);

private:
    void delete_zeros();
    std::vector<int> m_data;
};

TLongInt::TLongInt(std::string &input) {
    std::stringstream radixSS;
    if (input[0] == '0') {
        int i = 1;
        for (;input[i] == '0'; ++i);
        input = (i == (int) input.size()) ? "0" : input.substr(i);
    }
    m_data.clear();
    for (int i = input.size() - 1; i >= 0; i -= BASE_POW) {
        int start = i - BASE_POW + 1;
        start = (start < 0) ? 0 : start;
        int end = i - start + 1;
        radixSS << input.substr(start, end);
        int radix = 0;
        radixSS >> radix;
        m_data.push_back(radix);
        radixSS.clear();
    }
}

TLongInt::TLongInt(int n) {
    if (n < BASE)
        m_data.push_back(n);
    else {
        for(; n; n /= BASE)
            m_data.push_back(n % BASE);
    }
}

TLongInt TLongInt::operator+(const TLongInt &other) {
    TLongInt res;
    int carry = 0;
    for (int i = 0; i < std::max(m_data.size(), other.m_data.size()) || carry; ++i) {
        int aa = i < (int) m_data.size() ? m_data[i] : 0;
        int bb = i < (int) other.m_data.size() ? other.m_data[i] : 0;
        res.m_data.push_back(aa + bb + carry);
        carry = res.m_data.back() >= BASE;
        if (carry)
            res.m_data.back() -= BASE;
    }
    return res;
}

TLongInt TLongInt::operator-(const TLongInt &other) {
    TLongInt res;
    int carry = 0;
    for (int i = 0; i < m_data.size() || carry; ++i) {
        int aa = i < (int) m_data.size() ? m_data[i] : 0;
        int bb = i < (int) other.m_data.size() ? other.m_data[i] : 0;
        res.m_data.push_back(aa - carry - bb);
        carry = res.m_data.back() < 0;
        if (carry)
            res.m_data.back() += BASE;
    }
    res.delete_zeros();
    return res;
}

TLongInt TLongInt::operator*(const TLongInt &other) const {
    TLongInt res;
    res.m_data.resize(m_data.size() + other.m_data.size());
    int num1Size = (int) m_data.size();
    int num2Size = (int) other.m_data.size();
    for (int i = 0; i < num1Size; ++i) {
        int carry = 0;
        for (int j = 0; j < num2Size || carry; ++j) {
            int bb = j < num2Size ? other.m_data[j] : 0;
            res.m_data[i + j] += m_data[i] * bb + carry;
            carry = res.m_data[i + j] / BASE;
            res.m_data[i + j] -= carry * BASE;
        }
    }
    res.delete_zeros();
    return res;
}

TLongInt TLongInt::operator/(const TLongInt &other) {
    TLongInt res, cv = TLongInt(0);
    res.m_data.resize(m_data.size());
    for (int i = (int) m_data.size() - 1; i >= 0; --i) {
        cv.m_data.insert(cv.m_data.begin(), m_data[i]);
        if (!cv.m_data.back())
            cv.m_data.pop_back();
        int x = 0, l = 0, r = BASE;
        while (l <= r) {
            int m = (l + r) / 2;
            TLongInt cur = other * TLongInt(m);
            if (cur <= cv) {
                x = m;
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        res.m_data[i] = x;
        cv = cv - other * TLongInt(x);
    }
    res.delete_zeros();
    return res;
}

TLongInt TLongInt::power(int r) {
    TLongInt res(1);
    while (r) {
        if (r % 2)
            res = res * (*this);
        (*this) = (*this) * (*this);
        r /= 2;
    }
    return res;
}

bool TLongInt::operator==(const TLongInt &other) const {
    return this->m_data == other.m_data;
}

bool TLongInt::operator<(const TLongInt &other) const {
  // Если размеры неодинаковые, то сразу можем сказать,
  // какое число меньше.
  if( other.size() != this->size() ) {
      return this->size() < other.size();
  }

  for( int i = other.size() - 1; i >= 0; --i) {
      if( other.m_data[i] != this->m_data[i] ) {
          return this->m_data[i] < other.m_data[i];
      }
  }
  return false;
}

bool TLongInt::operator>(const TLongInt &other) const {
  // Если размеры неодинаковые, то сразу можем сказать,
  // какое число меньше.
  if( other.size() != this->size() ) {
      return this->size() > other.size();
  }

  for( int i = other.size() - 1; i >= 0; --i) {
      if( other.m_data[i] != this->m_data[i] ) {
          return this->m_data[i] > other.m_data[i];
      }
  }
  return false;
}

bool TLongInt::operator<=(const TLongInt &other) const {
    return !((*this) > other) ;
}

void TLongInt::delete_zeros() {
    while (m_data.size() > 1 && !m_data.back())
        m_data.pop_back();
}

std::ostream &operator<<(std::ostream &stream, const TLongInt &num) {
    int n = num.m_data.size();
    if (!n)
        return stream;
    stream << num.m_data[n - 1];
    for (int i = n - 2; i >= 0; --i)
        stream << std::setfill('0') << std::setw(BASE_POW) << num.m_data[i];

    return stream;
}

int main(void) {
    std::string strNum1, strNum2;
    char op;
    while (std::cin >> strNum1 >> strNum2 >> op) {
        TLongInt num1(strNum1);
        TLongInt num2(strNum2);
        switch(op) {
            case '+':
                std::cout << num1 + num2 << std::endl;
                break;

            case '-':
                if (num1 < num2)
                    std::cout << "Error" << std::endl;
                else
                    std::cout << num1 - num2 << std::endl;
                break;

            case '*':
                std::cout << num1 * num2 << std::endl;
                break;

            case '/':
                if (num2 == TLongInt(0))
                    std::cout << "Error" << std::endl;
                else
                    std::cout << num1 / num2 << std::endl;
                break;

            case '^':
                if (num1 == TLongInt(0)) {
                    if (num2 == TLongInt(0))
                        std::cout << "Error" << std::endl;
                    else
                        std::cout << "0" << std::endl;
                } else if (num1 == TLongInt(1)) {
                    std::cout << "1" << std::endl;
                } else
                    std::cout << num1.power(std::stoi(strNum2)) << std::endl;
                break;

            case '<':
                num1 < num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;

            case '>':
                num1 > num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;

            case '=':
                num1 == num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;
        }
    }

    return 0;
}
