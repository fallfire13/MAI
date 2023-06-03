#include "panel.h"
#include <QVBoxLayout>

panel::panel(QWidget *parent) : QWidget(parent)
{
    left = new QPushButton("left", this);
    right = new QPushButton("right", this);
    up = new QPushButton("up", this);
    down = new QPushButton("down", this);
    display_invis_lines = new QCheckBox("Display hiden lines", this);

    QVBoxLayout *lout = new QVBoxLayout;
    lout->addWidget(left);
    lout->addWidget(right);
    lout->addWidget(up);
    lout->addWidget(down);
    lout->addWidget(display_invis_lines);
    lout->addStretch();
    setLayout(lout);

    connect(left, SIGNAL(pressed()),
            this, SIGNAL(push_left()));
    connect(right, SIGNAL(pressed()),
            this, SIGNAL(push_right()));
    connect(up, SIGNAL(pressed()),
            this, SIGNAL(push_up()));
    connect(down, SIGNAL(pressed()),
            this, SIGNAL(push_down()));
    connect(display_invis_lines, SIGNAL(clicked(bool)),
            this, SIGNAL(clicked_display(bool)));
}
