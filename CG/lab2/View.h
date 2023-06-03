#ifndef VIEW_H
#define VIEW_H

#include <QWidget>
#include <pyramid.h>

class View : public QWidget
{
    Q_OBJECT
public:
    explicit View(QWidget *parent = nullptr);
    int get_center_x();
    int get_center_y();
    Pyramid &get_obelisk();
private:
    int center_x;
    int center_y;
    int step;
    Pyramid fig;
protected:
    void paintEvent(QPaintEvent *);
    void resizeEvent(QResizeEvent *);
signals:

public slots:
};

#endif // VIEW_H
