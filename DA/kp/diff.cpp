#include "diff.h"

std::vector<TAction> build_trace(
        const std::vector< std::vector<int64_t> >& trace,
        int64_t x, int64_t y, size_t total) {
    std::vector<TAction> diff_actions;

    for (int64_t d = trace.size() - 1; d >= 0; --d) {
        const std::vector<int64_t>& layer = trace[d];

        int64_t k = x - y;
        int64_t prev_k;

        bool went_down = (k == -d || (k != d && layer[k - 1 + total] < layer[k + 1 + total]));
        if (went_down) {
            prev_k = k + 1;
        } else {
            prev_k = k - 1;
        }

        int64_t prev_x = layer[prev_k + total];
        int64_t prev_y = prev_x - prev_k;

        while (x > prev_x && y > prev_y) {
            --x;
            --y;
            diff_actions.push_back({TAction::KEEP, x, y});
        }

        if (d == 0) {
            continue;
        }

        if (x == prev_x) {
            y = prev_y;
            diff_actions.push_back({TAction::ADD, x, y});
        } else if (y == prev_y) {
            x = prev_x;
            diff_actions.push_back({TAction::DEL, x, y});
        }
    }

    return std::vector<TAction>(diff_actions.rbegin(), diff_actions.rend());
}