#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class View;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    View *vw;
private slots:
    void turn_left();
    void turn_right();
    void turn_up();
    void turn_down();
    void display_hiden_lines(bool);
};

#endif // MAINWINDOW_H
