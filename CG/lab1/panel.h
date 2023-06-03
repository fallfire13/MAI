#ifndef PANEL_H
#define PANEL_H

#include <QWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QVBoxLayout>
class QPushButton;

class QSpinBox;
class QDoubleSpinBox;

class Panel : public QWidget
{
    Q_OBJECT
public:
    Panel(QWidget* parent = nullptr);
    double set_a() const;
    double set_A() const;
    double set_B() const;
    //void set_div_x(const int &x);
    //void set_div_y(const int &y);
signals:
    void a_changed(double);
    void A_changed(double);
    void B_changed(double);
    //void x_changed(int);
    //void y_changed(int);
    void scale_inc(bool);
    void scale_dec(bool);
private:
    QDoubleSpinBox* a;
    QDoubleSpinBox* A;
    QDoubleSpinBox* B;
    //QSpinBox* x;
    //QSpinBox* y;
    QPushButton *inc;
    QPushButton *dec;
};

#endif // PANEL_H
