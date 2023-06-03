#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QAction>
#include <QDockWidget>
#include "View.h"
#include "panel.h"
#include <cmath>
#include <QTextStream>
#include <QtMath>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    panel *pan = new panel;
    QDockWidget *dock = new QDockWidget("panel");
    dock->setWidget(pan);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    vw = new View;
    setCentralWidget(vw);

    QAction *quitAct = new QAction("&Quit", this);
    quitAct->setToolTip("Quit application");
    quitAct->setStatusTip("Closes the application");
    connect(quitAct, SIGNAL(triggered()),
            qApp, SLOT(quit()));

    QMenu *fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction(quitAct);

    QAction *panelAct = dock->toggleViewAction();
    panelAct->setStatusTip("Toggle panel");

    QMenu *viewMenu = menuBar()->addMenu("&View");
    viewMenu->addAction(panelAct);

    connect(pan, SIGNAL(push_left()),
            this, SLOT(turn_left()));
    connect(pan, SIGNAL(push_right()),
            this, SLOT(turn_right()));
    connect(pan, SIGNAL(push_up()),
            this, SLOT(turn_up()));
    connect(pan, SIGNAL(push_down()),
            this, SLOT(turn_down()));
    connect(pan, SIGNAL(clicked_display(bool)),
            this, SLOT(display_hiden_lines(bool)));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::turn_left() {
    std::vector<std::vector<double>> matrix_Ry{
        std::vector<double>{cos(M_PI / 12.), 0, sin(M_PI / 12.), 0},
        std::vector<double>{0, 1, 0, 0},
        std::vector<double>{-sin(M_PI / 12.), 0, cos(M_PI / 12.), 0},
        std::vector<double>{0, 0, 0, 1}
    };

    vw->get_obelisk().change_all_polygons(matrix_Ry);
    vw->update();
}

void MainWindow::turn_right() {
    std::vector<std::vector<double>> matrix_Ry{
        std::vector<double>{cos(-M_PI / 12.), 0, sin(-M_PI / 12.), 0},
        std::vector<double>{0, 1, 0, 0},
        std::vector<double>{-sin(-M_PI / 12.), 0, cos(-M_PI / 12.), 0},
        std::vector<double>{0, 0, 0, 1}
    };

    vw->get_obelisk().change_all_polygons(matrix_Ry);
    vw->update();
}

void MainWindow::turn_up() {
    std::vector<std::vector<double>> matrix_Rx{
        std::vector<double>{1, 0, 0, 0},
        std::vector<double>{0, cos(-M_PI / 12.), -sin(-M_PI / 12.), 0},
        std::vector<double>{0, sin(-M_PI / 12.), cos(-M_PI / 12.), 0},
        std::vector<double>{0, 0, 0, 1}
    };

    vw->get_obelisk().change_all_polygons(matrix_Rx);
    vw->update();
}

void MainWindow::turn_down() {
    std::vector<std::vector<double>> matrix_Rx{
        std::vector<double>{1, 0, 0, 0},
        std::vector<double>{0, cos(M_PI / 12.), -sin(M_PI / 12.), 0},
        std::vector<double>{0, sin(M_PI / 12.), cos(M_PI / 12.), 0},
        std::vector<double>{0, 0, 0, 1}
    };

    vw->get_obelisk().change_all_polygons(matrix_Rx);
    vw->update();
}

void MainWindow::display_hiden_lines(bool) {
    bool cur_val = vw->get_obelisk().get_displayHidenLines();
    vw->get_obelisk().set_displayHidenLines(!cur_val);

    vw->update();
}
