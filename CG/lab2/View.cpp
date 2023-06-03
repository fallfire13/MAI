#include "View.h"
#include <QPainter>
#include <vector>
#include <QResizeEvent>
#include <algorithm>
#include <cmath>

View::View(QWidget *parent) : QWidget(parent), step(50)
{
    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::white);
    setPalette(pal);

    setAutoFillBackground(true);

    std::vector<double> a{0, -4. * step, 0, 1};

    std::vector<double> b{4. * sin(1. * M_PI/8) * step, 4. * cos(1. * M_PI/8) * step, 0, 1};
    std::vector<double> c{4. * sin(2. * M_PI/8) * step, 4. * cos(2. * M_PI/8) * step, 0, 1};
    std::vector<double> d{4. * sin(3. * M_PI/8) * step, 4. * cos(3. * M_PI/8) * step, 0, 1};

    std::vector<double> e{4. * step, 0, 0, 1};

    std::vector<double> f{4. * sin(5. * M_PI/8) * step, 4. * cos(5. * M_PI/8) * step, 0, 1};
    std::vector<double> g{4. * sin(6. * M_PI/8) * step, 4. * cos(6. * M_PI/8) * step, 0, 1};
    std::vector<double> h{4. * sin(7. * M_PI/8) * step, 4. * cos(7. * M_PI/8) * step, 0, 1};

    std::vector<double> i{0, 4. * step, 0, 1};

    std::vector<double> j{4. * sin(9. * M_PI/8) * step, 4. * cos(9. * M_PI/8) * step, 0, 1};
    std::vector<double> k{4. * sin(10. * M_PI/8) * step, 4. * cos(10. * M_PI/8) * step, 0, 1};
    std::vector<double> l{4. * sin(11. * M_PI/8) * step, 4. * cos(11. * M_PI/8) * step, 0, 1};

    std::vector<double> m{-4. * step, 0, 0, 1};

    std::vector<double> n{4. * sin(13. * M_PI/8) * step, 4. * cos(13. * M_PI/8) * step, 0, 1};
    std::vector<double> o{4. * sin(14. * M_PI/8) * step, 4. * cos(14. * M_PI/8) * step, 0, 1};
    std::vector<double> q{4. * sin(15. * M_PI/8) * step, 4. * cos(15. * M_PI/8) * step, 0, 1};


    std::vector<double> s{0, 0, 4. * step, 1};

    std::vector<double> t{0, 0, 0, 1};


    Polygon p;

    p.add_vertex(s);
    p.add_vertex(c);
    p.add_vertex(b);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(q);
    p.add_vertex(s);
    p.add_vertex(i);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(i);
    p.add_vertex(s);
    p.add_vertex(b);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(a);
    p.add_vertex(s);
    p.add_vertex(j);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(h);
    p.add_vertex(s);
    p.add_vertex(a);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(d);
    p.add_vertex(c);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(e);
    p.add_vertex(d);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(f);
    p.add_vertex(e);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(g);
    p.add_vertex(f);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(h);
    p.add_vertex(g);
    fig.add_polygon(p);
    p.clear_verticies();
/*
    p.add_vertex(s);
    p.add_vertex(i);
    p.add_vertex(h);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(j);
    p.add_vertex(i);
    fig.add_polygon(p);
    p.clear_verticies();
*/
    p.add_vertex(s);
    p.add_vertex(k);
    p.add_vertex(j);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(l);
    p.add_vertex(k);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(m);
    p.add_vertex(l);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(n);
    p.add_vertex(m);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(o);
    p.add_vertex(n);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(q);
    p.add_vertex(o);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(i);
    p.add_vertex(b);
    p.add_vertex(c);
    p.add_vertex(d);
    p.add_vertex(e);
    p.add_vertex(f);
    p.add_vertex(g);
    p.add_vertex(h);
    p.add_vertex(a);
    p.add_vertex(j);
    p.add_vertex(k);
    p.add_vertex(l);
    p.add_vertex(m);
    p.add_vertex(n);
    p.add_vertex(o);
    p.add_vertex(q);
    fig.add_polygon(p);
    p.clear_verticies();

/*
    std::vector<double> a{-1.5 * step, -1.5 * step, 0, 1};
    std::vector<double> b{-1.5 * step, 1.5 * step, 0, 1};
    std::vector<double> c{1.5 * step, 1.5 * step, 0, 1};
    std::vector<double> d{1.5 * step, -1.5 * step, 0, 1};


    std::vector<double> s{0, 0, 150, 1};


    Polygon p;
    p.add_vertex(s);
    p.add_vertex(b);
    p.add_vertex(a);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(a);
    p.add_vertex(d);
    p.add_vertex(s);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(s);
    p.add_vertex(d);
    p.add_vertex(c);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(b);
    p.add_vertex(s);
    p.add_vertex(c);
    fig.add_polygon(p);
    p.clear_verticies();

    p.add_vertex(a);
    p.add_vertex(b);
    p.add_vertex(c);
    p.add_vertex(d);

    fig.add_polygon(p);
    p.clear_verticies();
*/
}

Pyramid &View::get_obelisk() {
    return fig;
}

void View::paintEvent(QPaintEvent *) {
    QPainter ptr{this};
    ptr.setPen(QPen(Qt::black, 2));

    center_x = width() / 2;
    center_y = height() / 2;

    /***************************/
    fig.draw(&ptr, center_x, center_y);
    /***************************/
}

void View::resizeEvent(QResizeEvent *e) {
    if (e->oldSize().width() == -1 || e->oldSize().height() == -1)
        return;
    double coef_x = width() / static_cast<double>(e->oldSize().width());
    double coef_y = height() / static_cast<double>(e->oldSize().height());
    double coef_z = sqrt(std::pow(width(), 2) + std::pow(height(), 2))
                    / sqrt(std::pow(static_cast<double>(e->oldSize().width()), 2) + std::pow(static_cast<double>(e->oldSize().height()), 2));
    std::vector<std::vector<double>> matrix_s{
        std::vector<double>{coef_x, 0, 0, 0},
        std::vector<double>{0, coef_y, 0, 0},
        std::vector<double>{0, 0, coef_z, 0},
        std::vector<double>{0, 0, 0, 1}
    };
    fig.change_all_polygons(matrix_s);
    update();
}

