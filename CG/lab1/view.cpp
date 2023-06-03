#include "view.h"
#include "panel.h"
#include <QPainter>
#include <QResizeEvent>
#include <algorithm>
#include <cmath>
#include <utility>
#include <QTextStream>

View::View(QWidget* parent) : QWidget(parent), pan(0) , step_x(50), step_y(50)
{
    QPalette pal(palette());
    pal.setColor(QPalette::Window, Qt::white);
    setPalette(pal);
    setAutoFillBackground(true);
}

const Panel* View::controlPanel() const
{
    return pan;
}

void View::setControlPanel(const Panel* p)
{
    pan = p;
    update();
}

void View::setStep(const int &x, const int &y)
{
    step_x = x;
    step_y = y;
}

std::pair<int, int> View::getStep()
{
    return std::make_pair(step_x, step_y);
}

void draw_coords(View *v)
{
    const int div_x = v->set_x;
    const int div_y = v->set_y;
    const double pi = atan(1) * 4;
    v->x_center = v->width() / 2.0;
    v->y_center = v->height() / 2.0;


    QPainter ptr{v};
       ptr.setPen(QPen(Qt::black, 3));

    //const QPointF center(v->width() / 2.0, v->height() / 2.0);

    // drawing axis x
    QPoint p1{0, static_cast<int>(v->y_center)};
    QPoint p2{v->width(), static_cast<int>(v->y_center)};

    const int arrow_length = 10;
    ptr.drawLine(p1, p2);

    //axis x - arrow
    QPointF p_branch1{static_cast<double>(v->width()), static_cast<double>(v->y_center)};
        QPointF p_branch2{arrow_length * cos(-11 * pi / 12) + v->width(),
                    arrow_length * sin(-11 * pi / 12) + v->y_center};
        ptr.drawLine(p_branch1, p_branch2);
        p_branch2 = {arrow_length * cos(11 * pi / 12) + v->width(),
                     arrow_length * sin(11 * pi / 12) + v->y_center};
        ptr.drawLine(p_branch1, p_branch2);

    // drawing axis y
    p1.setX(static_cast<int>(v->x_center));
    p1.setY(0);
    p2.setX(static_cast<int>(v->x_center));
    p2.setY(v->height());
    ptr.drawLine(p1, p2);

    //axis y - arrow
    p_branch1 = {static_cast<double>(v->x_center), 0};
    p_branch2 = {arrow_length * cos(5 * pi / 12) + v->x_center,
                arrow_length * sin(5 * pi / 12)};
    ptr.drawLine(p_branch1, p_branch2);
    p_branch2 = {arrow_length * cos(7 * pi / 12) + v->x_center,
                 arrow_length * sin(7 * pi / 12)};
    ptr.drawLine(p_branch1, p_branch2);

    ptr.setPen(QPen(Qt::black, 3));
    ptr.drawText(QPointF(v->x_center - (v->step_x / 4), v->y_center + (v->step_y / 4)),
        QString::number(0));

    //drawing grid

        p_branch1.setY(v->y_center + v->step_y / 4);
        p_branch2.setY(v->y_center - v->step_y / 4);
        p1.setY(0);
        p2.setY(v->height());
        for (int x = static_cast<int>(v->step_x), num = 0; x + v->x_center < v->width() || v->x_center - x > 0;
             x += v->step_x, num += div_x) {
            //grid sticks - positive
            ptr.setPen(Qt::gray);
            p1.setX(static_cast<int>(x + v->x_center));
            p2.setX(static_cast<int>(x + v->x_center));
            ptr.drawLine(p1, p2);

            //points - positive
            ptr.setPen(QPen(Qt::black, 3));
            p_branch1.setX(x + v->x_center);
            p_branch2.setX(x + v->x_center);
            ptr.drawLine(p_branch1, p_branch2);

            ptr.drawText(QPointF(x + v->x_center - v->step_x / 6, static_cast<int>(p_branch1.y()) - v->step_y / 2),
                         QString::number(num + div_x));

            //grid sticks - negative

            ptr.setPen(Qt::gray);
            p1.setX(static_cast<int>(v->x_center - x));
            p2.setX(static_cast<int>(v->x_center - x));
            ptr.drawLine(p1, p2);

            //points - negative
            ptr.setPen(QPen(Qt::black, 3));
            p_branch1.setX(v->x_center - x);
            p_branch2.setX(v->x_center - x);
            ptr.drawLine(p_branch1, p_branch2);

            ptr.drawText(QPointF(v->x_center - x - v->step_x / 6, static_cast<int>(p_branch1.y()) - v->step_y / 2),
                         QString::number(-1 * (num + div_x)));
        }

        p_branch1.setX(v->x_center + v->step_x / 4);
        p_branch2.setX(v->x_center - v->step_x / 4);
        p1.setX(0);
        p2.setX(v->width());
        for (int y = static_cast<int>(v->step_y), num = 0; y + v->y_center < v->height() || v->y_center - y > 0;
             y += v->step_y, num += div_y) {
            //grid sticks - negative

            ptr.setPen(Qt::gray);
            p1.setY(static_cast<int>(y + v->y_center));
            p2.setY(static_cast<int>(y + v->y_center));
            ptr.drawLine(p1, p2);

            //points - negative
            ptr.setPen(QPen(Qt::black, 3));
            p_branch1.setY(y + v->y_center);
            p_branch2.setY(y + v->y_center);
            ptr.drawLine(p_branch1, p_branch2);

            ptr.drawText(QPointF(static_cast<int>(p_branch1.x()) + v->step_x / 6, v->y_center - y + v->step_y / 6),
                         QString::number(num + div_y));

            //grid sticks - positive
            ptr.setPen(Qt::gray);
            p1.setY(static_cast<int>(v->y_center - y));
            p2.setY(static_cast<int>(v->y_center - y));
            ptr.drawLine(p1, p2);

            //points - positive
            ptr.setPen(QPen(Qt::black, 3));
            p_branch1.setY(v->y_center - y);
            p_branch2.setY(v->y_center - y);
            ptr.drawLine(p_branch1, p_branch2);

            ptr.drawText(QPointF(static_cast<int>(p_branch1.x()) + v->step_x / 6, v->y_center + y + v->step_y / 6),
                         QString::number(-1 * (num + div_y)));
        }
}

void draw_func(View *v) {
    QPainter ptr{v};
    ptr.setPen(QPen(Qt::red, 3));
    //v->x_center = v->width() / 2.0;
    //v->y_center = v->height() / 2.0;
    const double step = 0.01;
    QPointF p1{static_cast<double>(v->x_center), (v->y_center)};
    QPointF p2{};
    QPointF p3{p1};
    QPointF p4{};
    QPointF p5{p1};
    QPointF p6{};
    QPointF p7{p1};
    QPointF p8{};
    for (double x = step; v->pan->set_a() + x != 0 && v->pan->set_B() < v->pan->set_a() && v->pan->set_A() > -(v->pan->set_a()) && x <= v->pan->set_B() && x >= v->pan->set_A(); x += step) {
        p2 = {v->x_center + x * v->step_x/ v->set_x, v->y_center - sqrt((pow(x, 2) * (v->pan->set_a() - x ) / (v->pan->set_a() + x))) * v->step_y / v->set_y };
        ptr.drawLine(p1, p2);
        p1 = p2;
        p4 = {v->x_center + x * v->step_x/ v->set_x, v->y_center + sqrt((pow(x, 2) * (v->pan->set_a() - x ) / (v->pan->set_a() + x))) * v->step_y / v->set_y };
        ptr.drawLine(p3, p4);
        p3 = p4;
        p6 = {v->x_center - x * v->step_x/ v->set_x, v->y_center + sqrt((pow(x, 2) * (v->pan->set_a() + x ) / (v->pan->set_a() - x))) * v->step_y / v->set_y };
        ptr.drawLine(p5, p6);
        p5 = p6;
        p8 = {v->x_center - x * v->step_x/ v->set_x, v->y_center - sqrt((pow(x, 2) * (v->pan->set_a() + x ) / (v->pan->set_a() - x))) * v->step_y / v->set_y };
        ptr.drawLine(p7, p8);
        p7 = p8;
   }
}

void View::paintEvent(QPaintEvent*)
{
    if (!pan)
        return;
    draw_coords(this);
    draw_func(this);
}


void View::resizeEvent(QResizeEvent *r_event)
{
    if (r_event->oldSize().width() == -1 || r_event->oldSize().height() == -1)
            return;
    double coef_x = width() / static_cast<double>(r_event->oldSize().width());
    double coef_y = height() / static_cast<double>(r_event->oldSize().height());
    if (step_x * coef_x < 1 || step_y * coef_y < 1) {
            update();
            return;
        }
        step_x *= coef_x;
        step_y *= coef_y;
        x_center *= coef_x;
        y_center *= coef_y;
        update();
}


void View::mousePressEvent(QMouseEvent *event) {
    previousPoint = event->pos();
}

void View::mouseMoveEvent(QMouseEvent *event) {
    QPointF newPoint = event->pos();
    double delta_x = newPoint.x() - previousPoint.x();
    double delta_y = newPoint.y() - previousPoint.y();

    x_center += delta_x;
    y_center += delta_y;

    previousPoint = newPoint;
    update();
}
