#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <cctype>
#include <algorithm>
#include <map>
#include <iterator>
#include <cmath>

const int OUTWORD = 0;
const int INWORD = 1;

struct TAnswer{
    int strNumber;
    int posNumber;
};

void PatternParsing(std::vector<std::pair<int, int> > &pattern, int &max, std::map <int, int> &dict);
bool TextParsing(std::deque<std::pair<int, TAnswer> > &text, int pSize, int shift);
void PreFunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array, int max, int pSize);
void Nfunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array,int pSize);
void Lfunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array, std::vector<int> &nArray, std::vector<int> & llArray,int pSize);

int main() {
    int max = 0;
    std::vector<std::pair<int, int> > pattern;
    std::map <int, int> dict;
    PatternParsing(pattern, max, dict);
    int pSize = pattern.size();
    std::map<int, int>::iterator It;
    std::deque<std::pair<int, TAnswer> > text;
    TextParsing(text, pSize, 0);
    std::vector<int> bcArray(max);
    std::vector<int> nFuncArray(pSize);
    std::vector<int> lArray(pSize);
    std::vector<int> llArray(pSize);

    PreFunc(pattern, bcArray, max, pSize);
    Nfunc(pattern, nFuncArray, pSize);
    Lfunc(pattern, lArray, nFuncArray, llArray, pSize);

    if(pSize > 1) {
        bool flag = true;
        while(flag) {
            int i;
            int shift;
            for(i = pSize - 1; i >= 0 && pattern[i].first == text[i].first; --i) {
            }
            if(i == -1) {
                printf("%d, %d\n", text[0].second.strNumber, text[0].second.posNumber);
                shift = pSize - llArray[1];
            } else {
                int bc = -1;
                It = dict.find(text[i].first);
                if(It != dict.end()) {
                    bc = It->second;
                }
                if(bc == -1) {
                    bc = pSize;
                } else {
                    // std::cout << bc << " " << max << '\n';
                    bc = bcArray[bc] - (pSize - 1 - i);
                }
                int lShift;
                if(i == pSize - 1) {
                    lShift = 1;
                } else {
                    lShift = lArray[i + 1] > 0 ? pSize - 1 - lArray[i + 1] : pSize - 1 - llArray[i + 1];
                }
                if(lShift == 0) {
                    lShift = 1;
                }
                shift = std::max(bc, lShift);
            }
            flag = TextParsing(text, pSize, shift);
        }
    } else if(pSize == 1) {
        bool flag = true;
        int i;
        int shift = 1;
        while(flag) {
            for(i = pSize - 1; i >= 0 && pattern[i].first == text[i].first; --i) {
            }
            if(i == -1) {
                printf("%d, %d\n", text[0].second.strNumber, text[0].second.posNumber);
            }
            flag = TextParsing(text, pSize, shift);
        }
    }

    return 0;
}

bool TextParsing(std::deque<std::pair<int, TAnswer> > &text, int pSize, int shift) {
    char ch;
    int i;
    if(!text.empty()) {
        i = pSize - shift;
    } else {
        i = 0;
    }
    static int nOfString = 1;
    static int nOfWord = 1;
    for(int m = shift; m > 0; --m) {
        text.pop_front();
    }
    text.resize(pSize);
    while(i < pSize) {
        while(!isdigit(ch = getchar())) {
            if(ch == EOF) {
                return false;
            }
            if(ch == '\n') {
                ++nOfString;
                nOfWord = 1;
            }
        }
        text[i].first = 0;
        text[i].first += ch - '0';
        while(isdigit(ch = getchar())) {
            text[i].first *= 10;
            text[i].first += ch - '0';
        }

        text[i].second.strNumber = nOfString;
        text[i].second.posNumber = nOfWord;

        if(ch == '\n') {
            ++nOfString;
            nOfWord = 1;
        } else {
            ++nOfWord;
        }
        ++i;
    }
    return true;
}

void PatternParsing(std::vector<std::pair<int, int> > &pattern, int &max, std::map <int, int> &dict) {
    char ch;
    int i = 0, j = 0;
    int condition = OUTWORD;

    std::map<int, int>::iterator it;
    while ((ch = getchar()) != '\n') {
        switch (condition) {
            case OUTWORD:
                if (ch == EOF) {
                    return;
                }
                if (isdigit(ch)) {
                    condition = INWORD;
                    pattern.push_back({ ch - '0', 0 });
                }
                break;
            case INWORD:
                if (isdigit(ch)) {
                    pattern[i].first *= 10;
                    pattern[i].first += ch - '0';
                } else {
                    condition = OUTWORD;
                    it = dict.find(pattern[i].first);
                    if (it != dict.end()) {
                        pattern[i].second = it->second;
                    } else {
                        pattern[i].second = j;
                        dict.insert({pattern[i].first, pattern[i].second});
                        ++j;
                    }
                    ++i;
                }
                break;
        }
    }
    if (!pattern.empty()) {
        it = dict.find(pattern[i].first);
        if (it != dict.end()) {
            pattern[i].second = it->second;
        } else {
            pattern[i].second = j;
            dict.insert({pattern[i].first, pattern[i].second});
            ++j;
        }
        max = j;
    }
}

void PreFunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array, int max, int pSize) {
    for (int i = 0; i < max; ++i) {
        array[i] = pSize;
    }
    for (int i = 0; i < pSize - 1; ++i) {
        if (pattern[i].second >= array.size()) continue;
        array[pattern[i].second] = pSize - 1 - i;
    }
}

void Nfunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array, int pSize) {
    for(int i = 0; i < pSize; ++i) {
        array[i] = 0;
    }
    for (int i  = pSize - 2, l = pSize - 1, r = pSize - 1; i >= 0; --i) {
        if (i >= l) {
            array[i] = std::min(i - l + 1, array[pSize - 1 - r + i]);
        }

        while (i - array[i] >= 0 && pattern[pSize - 1 - array[i]].first == pattern[i - array[i]].first) {
            ++array[i];
        }

        if (i - array[i] + 1 < l) {
            l = i - array[i] + 1;
            r = i;
        }
    }
}

void Lfunc(std::vector<std::pair<int, int> > &pattern, std::vector<int> &array, std::vector<int> &nArray, std::vector<int> &llArray, int pSize) {
    int k = 0;
    for(int i = 0; i < pSize; ++i) {
        array[i] = 0;
    }
    int max = 0;
    int i = 0;
    for(i = 0; i < pSize - 1; ++i) {
        if(nArray[i] != 0) {
            k = pSize - nArray[i];
            array[k] = i;
        }
        if (nArray[i] > max && nArray[i] == i + 1) {
            max = nArray[i];
        }
        llArray[pSize - 1 - i] = max;
    }
    if (nArray[i] > max && nArray[i] == i + 1) {
        max = nArray[i];
    }
    llArray[pSize - 1 - i] = max;
}
