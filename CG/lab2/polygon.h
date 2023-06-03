#ifndef POLYGON_H
#define POLYGON_H

#include <vector>
#include <QPainter>

class Polygon
{
    std::vector<std::vector<double>> verticies;
public:
    Polygon();
    Polygon(const std::vector<std::vector<double>> &v);

    std::vector<double> get_normal();
    void change_verticies(const std::vector<std::vector<double>> &v);
    void add_vertex(const std::vector<double> &v);
    void add_vertex(double x, double y, double z, double d);
    void clear_verticies();
    void draw(QPainter *ptr, int center_x, int center_y);
};

#endif // POLYGON_H
