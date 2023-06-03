#include "pyramid.h"

Pyramid::Pyramid() : displayHidenLines(false)
{

}

Pyramid::Pyramid(const std::vector<Polygon> &p) : Pyramid() {
    polygons = p;
}

void Pyramid::set_displayHidenLines(bool b) {
    displayHidenLines = b;
}

bool Pyramid::get_displayHidenLines() {
    return displayHidenLines;
}

void Pyramid::change_all_polygons(const std::vector<std::vector<double>> &v) {
    for (auto &it: polygons) {
        it.change_verticies(v);
    }
}

void Pyramid::add_polygon(const Polygon &p) {
    polygons.push_back(p);
}

void Pyramid::draw(QPainter *ptr, int center_x, int center_y) {
    for (auto p : polygons) {
        auto p_normal = p.get_normal();
        if (p_normal[2] > 0) {
            p.draw(ptr, center_x, center_y);
        } else {
            if (displayHidenLines) {
                QPen new_pen(Qt::gray, 1, Qt::DashLine);
                QPen old_pen = ptr->pen();
                ptr->setPen(new_pen);
                p.draw(ptr, center_x, center_y);
                ptr->setPen(old_pen);
            }
        }
    }
}
