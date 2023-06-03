#ifndef PANEL_H
#define PANEL_H

#include <QWidget>
#include <QPushButton>
#include <QCheckBox>

class panel : public QWidget
{
    Q_OBJECT
public:
    explicit panel(QWidget *parent = nullptr);
private:
    QPushButton *left;
    QPushButton *right;
    QPushButton *up;
    QPushButton *down;
    QCheckBox *display_invis_lines;
signals:
    void push_left();
    void push_right();
    void push_up();
    void push_down();
    void clicked_display(bool);
public slots:
};

#endif // PANEL_H
