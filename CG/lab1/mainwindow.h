#ifndef MAINWINDOW_H
#define MAINWINDOW_H

class View;

#include <QMainWindow>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
private slots:
    void redrawOnValueChange(double);
    void scale_inc(bool);
    void scale_dec(bool);
private:
    View* view;

};
#endif // MAINWINDOW_H