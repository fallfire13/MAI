#ifndef VIEW_H
#define VIEW_H

#include <QWidget>
#include <utility>

class Panel;

class View : public QWidget
{
public:
    View(QWidget* parent = nullptr);
    const Panel* controlPanel() const;
    void setControlPanel(const Panel* p);
    void setStep(const int &, const int &);
    std::pair<int, int> getStep();
    int set_x = 1;
    int set_y = 1;
    double get_x(const double &r, const double &fi);
    double get_y(const double &r, const double &fi);
    friend void draw_coords(View *);
    friend void draw_func(View *);
protected:
    void paintEvent(QPaintEvent *);
    void resizeEvent(QResizeEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
private:
    const Panel* pan;

    double step_x;
    double step_y;
    double x_center;
    double y_center;
    QPointF previousPoint;
};


#endif // VIEW_H
