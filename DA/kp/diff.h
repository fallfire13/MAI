#ifndef DIFF_H
#define DIFF_H

#include <cstdint>
#include <vector>

#include "structs.h"

std::vector<TAction> build_trace(
        const std::vector< std::vector<int64_t> >& trace, 
        int64_t x, int64_t y, size_t total);

template<typename T>
std::vector<TAction> 
find_diff(const T& data1, const T& data2) {
    const size_t len1  = data1.size();
    const size_t len2  = data2.size();
    const size_t total = len1 + len2;

    std::vector<int64_t> extensions(2 * total + 1);
    std::vector< std::vector<int64_t> > trace;

    extensions[1 + total] = 0;
    for (int64_t path = 0; path <= total; ++path) {

        trace.push_back(extensions);

        for (int64_t diag = -path; diag <= path; diag += 2) {
            int64_t x, y;
            bool go_down = (diag == -path 
                            || (diag != path 
                                && extensions[diag - 1 + total] < extensions[diag + 1 + total]));

            if (go_down) {
                x = extensions[diag + 1 + total];
            } else {
                x = extensions[diag - 1 + total] + 1;
            }

            y = x - diag;

            while (x < len1 && y < len2 && data1[x] == data2[y]) {
                ++x;
                ++y;
            }

            extensions[diag + total] = x;
            if (x >= len1 && y >= len2) {
                return build_trace(trace, len1, len2, total);
            }
        }
    }

    return std::vector<TAction> ();
}



#endif