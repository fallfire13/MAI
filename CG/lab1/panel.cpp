#include "panel.h"

#include <QPushButton>

Panel::Panel(QWidget *parent) : QWidget(parent)
{
    QLabel* lbl_a(new QLabel("a:"));
    a = new QDoubleSpinBox;
    a->setRange(-1000, 1000);
    a->setSingleStep(0.1);
    a->setValue(3);
    QLabel* lbl_A(new QLabel("A:"));
    A = new QDoubleSpinBox;
    A->setRange(-999.9, 999.9);
    A->setSingleStep(0.1);
    A->setValue(-2.9);
    QLabel* lbl_B(new QLabel("B:"));
    B = new QDoubleSpinBox;
    B->setRange(-999.9, 999.9);
    B->setSingleStep(0.1);
    B->setValue(2.9);

    inc = new QPushButton("+", this);
    dec = new QPushButton("-", this);

    QVBoxLayout* lout(new QVBoxLayout);
    lout->addWidget(lbl_a);
    lout->addWidget(a);
    lout->addWidget(lbl_A);
    lout->addWidget(A);
    lout->addWidget(lbl_B);
    lout->addWidget(B);
    lout->addWidget(inc);
    lout->addWidget(dec);
    lout->addStretch();
    setLayout(lout);

    connect(a, SIGNAL(valueChanged(double)),
        this, SIGNAL(a_changed(double)));
    connect(A, SIGNAL(valueChanged(double)),
        this, SIGNAL(A_changed(double)));
    connect(B, SIGNAL(valueChanged(double)),
        this, SIGNAL(B_changed(double)));
    connect(inc, SIGNAL(clicked(bool)),
                this, SIGNAL(scale_inc(bool)));
    connect(dec, SIGNAL(clicked(bool)),
                this, SIGNAL(scale_dec(bool)));
}

double Panel::set_a() const
{
    return a->value();
}

double Panel::set_A() const
{
    return A->value();
}

double Panel::set_B() const
{
    return B->value();
}
