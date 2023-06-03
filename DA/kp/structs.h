#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstdint>

struct TAction {
    enum {ADD, DEL, KEEP} type;
    int64_t x, y;
};

#endif