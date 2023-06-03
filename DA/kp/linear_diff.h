#ifndef LINEAR_DIFF_H
#define LINEAR_DIFF_H

#include <vector>
#include <utility>
#include <cmath>

#include "structs.h"

namespace {
    struct Box {
        Box(int64_t left,  int64_t top, 
                int64_t right, int64_t bottom) {
            this->left   = left;
            this->top    = top;
            this->right  = right;
            this->bottom = bottom;

            width  = right - left;
            height = bottom - top;
            size   = width + height;
            delta  = width - height;
        }

        int64_t left;
        int64_t top;
        int64_t right;
        int64_t bottom;
        int64_t width;
        int64_t height;
        int64_t size;
        int64_t delta;
    };

    using Point = std::pair<int64_t, int64_t>;
    using Snake = std::pair<Point, Point>;

    size_t wrap_index(int64_t i, int64_t size) {
        while (i >= size) {
            i -= size;
        }
        while (i < 0) {
            i += size;
        }
        return i;
    }

    template<typename T>
    std::pair<Snake, bool> forwards(
                const Box& box, std::vector<int64_t>& forw_snakes,
                std::vector<int64_t>& back_snakes, int64_t depth,
                int64_t total, const T& a, const T& b) {
        for (int64_t k = depth; k >= -depth; k -= 2) {
            const int64_t size = 2 * total + 1;

            int64_t c = k - box.delta;
            int64_t x; 
            int64_t px;
            int64_t y;
            int64_t py;

            bool go_down = ( k == -depth || 
                    (k != -depth && forw_snakes[wrap_index(k - 1, size)] < forw_snakes[wrap_index(k + 1, size)]) );
            if (go_down) {
                px = x = forw_snakes[wrap_index(k + 1, size)];
            } else {
                px = forw_snakes[wrap_index(k - 1, size)];
                x = px + 1;
            }

            y  = box.top + (x - box.left) - k;
            py = (depth == 0 || x != px) ? y : y - 1;

            while (x < box.right && y < box.bottom && a[x] == b[y]) {
                ++x;
                ++y;
            }

            forw_snakes[wrap_index(k, size)] = x;

            bool delta_odd = box.delta % 2 == 1;
            bool c_between = c >= -depth + 1 && c <= depth - 1;
            if (delta_odd && c_between && y >= back_snakes[wrap_index(c, size)]){
                if (depth == 1) {
                    while (x > px && y > py) {
                        --x;
                        --y;
                    }
                }
                Snake snake = std::make_pair(
                                 std::make_pair(px, py),
                                 std::make_pair(x, y)
                              );
                return std::make_pair(snake, true);
            }
        }

        return std::make_pair(Snake(), false);
    }

    template<typename T>
    std::pair<Snake, bool> backwards(
                const Box& box, std::vector<int64_t>& forw_snakes,
                std::vector<int64_t>& back_snakes, int64_t depth,
                int64_t total, const T& a, const T& b) {
        for (int64_t c = depth; c >= -depth; c -= 2) {
            const int64_t size = 2 * total + 1;

            int64_t k = c + box.delta;
            int64_t x;
            int64_t px;
            int64_t y;
            int64_t py;
            
            bool go_up = ( c == -depth || 
                    (c != depth && back_snakes[wrap_index(c - 1, size)] > back_snakes[wrap_index(c + 1, size)]) );
            if (go_up) {
                py = y = back_snakes[wrap_index(c + 1, size)];
            } else {
                py = back_snakes[wrap_index(c - 1, size)];
                y  = py - 1;
            }

            x  = box.left + (y - box.top) + k;
            px = (depth == 0 || y != py) ? x : x + 1;
            
            while (x > box.left && y > box.top && a[x - 1] == b[y - 1]) {
                --x;
                --y;
            }

            back_snakes[wrap_index(c, size)] = y;

            bool delta_even = box.delta % 2 == 0;
            bool k_between = k >= -depth && k <= depth;
            if (delta_even && k_between && x <= forw_snakes[wrap_index(k, size)]) {
                Snake snake = std::make_pair(
                                 std::make_pair(x, y),
                                 std::make_pair(px, py)
                              );
                return std::make_pair(snake, true);
            }
        }

        return std::make_pair(Snake(), false);
    }

    template<typename T>
    std::pair<Snake, bool> midpoint(const Box& box, const T& a, const T& b) {
        if (box.size == 0) {
            return std::make_pair(Snake(), false);
        }

        int64_t max_d = ceil(box.size / 2.);
        std::vector<int64_t> forw_snakes(2 * max_d + 1);
        forw_snakes[1] = box.left;
        std::vector<int64_t> back_snakes(2 * max_d + 1);
        back_snakes[1] = box.bottom;

        std::pair<Snake, bool> snake;

        for (int64_t depth = 0; depth <= max_d; ++depth) {
            snake = forwards(box, forw_snakes, back_snakes, depth, max_d, a, b);
            if (snake.second) {
                return snake;
            }
            snake = backwards(box, forw_snakes, back_snakes, depth, max_d, a, b);
            if (snake.second) {
                return snake;
            }
        }

        snake.second = false;
        return snake;
    }

    template<typename T>
    std::vector<Point> find_path(
            int64_t left,  int64_t top, 
            int64_t right, int64_t bottom, 
            const T& a, const T& b) {
        Box box(left, top, right, bottom);
        std::pair<Snake, bool> snake = midpoint(box, a, b);

        std::vector<Point> result;

        if (!snake.second) {
            return result;
        }

        Point start = snake.first.first;
        Point end   = snake.first.second;

        std::vector<Point> head = find_path(box.left, box.top, start.first, start.second, a, b);
        std::vector<Point> tail = find_path(end.first, end.second, box.right, box.bottom, a, b);

        if (head.empty()) {
            result.push_back(start);
        } else {
            result.insert(result.end(), head.begin(), head.end());
        }

        if (tail.empty()) {
            result.push_back(end);
        } else {
            result.insert(result.end(), tail.begin(), tail.end());
        }

        return result;
    }

    template<typename T>
    std::vector<TAction> find_diff_linear(const T& data1, const T& data2) {
        std::vector<Point> path = find_path(0, 0, data1.size(), data2.size(), data1, data2);
        std::vector<TAction> diff_actions;
        int64_t x = 0;
        int64_t y = 0;

        for (size_t i = 0; i < path.size(); ++i) {
            const Point& p = path[i];

            while (x < p.first && y < p.second) {
                diff_actions.push_back({TAction::KEEP, x, y});
                ++x;
                ++y;
            }

            if (p.first - x < p.second - y) {
                diff_actions.push_back({TAction::ADD, x, y});
                ++y;
            } else if (p.first - x > p.second - y) {
                diff_actions.push_back({TAction::DEL, x, y});
                ++x;
            }
        }

        return diff_actions;
    }
}

#endif