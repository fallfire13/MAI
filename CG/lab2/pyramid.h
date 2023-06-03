#ifndef PYRAMID_H
#define PYRAMID_H

#include "polygon.h"

class Pyramid
{
private:
    std::vector<Polygon> polygons;
    bool displayHidenLines;
public:
    Pyramid();
    Pyramid(const std::vector<Polygon> &p);

    void set_displayHidenLines(bool b);
    bool get_displayHidenLines();
    void change_all_polygons(const std::vector<std::vector<double>> &v);
    void add_polygon(const Polygon &p);
    void draw(QPainter *ptr, int center_x, int center_y);
};

#endif // PYRAMID_H
