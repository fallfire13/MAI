#include "polygon.h"
#include <QTextStream>

Polygon::Polygon()
{

}

Polygon::Polygon(const std::vector<std::vector<double>> &v) : verticies(v) {}

std::vector<double> Polygon::get_normal() {
    std::vector<double> first = {
        verticies[1][0] - verticies[0][0],
        verticies[1][1] - verticies[0][1],
        verticies[1][2] - verticies[0][2]
    };
    std::vector<double> second = {
        verticies[verticies.size() - 1][0] - verticies[0][0],
        verticies[verticies.size() - 1][1] - verticies[0][1],
        verticies[verticies.size() - 1][2] - verticies[0][2]
    };
    std::vector<double> normal = {
        first[1] * second[2] - second[1] * first[2],
        second[0] * first[2] - first[0] * second[2],
        first[0] * second[1] - second[0] * first[1]
    };

    return normal;
}

void Polygon::change_verticies(const std::vector<std::vector<double>> &v) {
    for (auto &it: verticies) {
        std::vector<double> res(4);
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                res[i] += v[i][j] * it[j];
            }
        }
        it = res;
    }
}

void Polygon::add_vertex(const std::vector<double> &v) {
    verticies.push_back(v);
}

void Polygon::add_vertex(double x, double y, double z, double d) {
    verticies.push_back(std::vector<double>{x, y, z, d});
}

void Polygon::clear_verticies() {
    verticies.clear();
}

void Polygon::draw(QPainter *ptr, int center_x, int center_y) {
    for (size_t i = 0; i < verticies.size() - 1; i++) {
        ptr->drawLine(static_cast<int>(verticies[i][0] + center_x),
                      static_cast<int>(verticies[i][1] + center_y),
                      static_cast<int>(verticies[i + 1][0] + center_x),
                      static_cast<int>(verticies[i + 1][1] + center_y));
    }
    ptr->drawLine(static_cast<int>(verticies[0][0] + center_x),
                  static_cast<int>(verticies[0][1] + center_y),
                  static_cast<int>(verticies[verticies.size() - 1][0] + center_x),
                  static_cast<int>(verticies[verticies.size() - 1][1] + center_y));
}
