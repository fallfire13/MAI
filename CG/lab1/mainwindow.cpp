#include "mainwindow.h"
#include "panel.h"
#include "view.h"

#include<QDockWidget>
#include<QAction>
#include<QMenuBar>
#include<QApplication>
#include<QToolBar>
#include<QStatusBar>
#include <QTextStream>
#include <QResizeEvent>
#include <QAction>
#include <QScrollArea>
#include <algorithm>
#include <utility>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    Panel* pan(new Panel);
    QDockWidget* dock(new QDockWidget("Controls"));
    dock->setWidget(pan);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    view = new View;
    view->setControlPanel(pan);
    view->setMinimumHeight(20);
    view->setMinimumWidth(20);

    setCentralWidget(view);

    QAction* quitAct(new QAction(
        QIcon(":/buttons/g4192.png"), "&Quit", this));
    quitAct->setShortcut(QKeySequence("Ctrl+Q"));
    quitAct->setToolTip("Quit application");
    quitAct->setStatusTip("Closes the application");
    connect(quitAct, SIGNAL(triggered()), qApp, SLOT(quit()));

    QMenu* fileMenu(menuBar()->addMenu("&File"));
    fileMenu->addAction(quitAct);

    QAction* panelAct(dock->toggleViewAction());
    panelAct->setStatusTip("Toggle panel");

    QMenu* viewMenu(menuBar()->addMenu("&View"));
    viewMenu->addAction(panelAct);

    QToolBar* bar(addToolBar("Application"));
        bar->addAction(quitAct);
    statusBar()->showMessage("Ready");

    connect(pan, SIGNAL(a_changed(double)),
        this, SLOT(redrawOnValueChange(double)));
    connect(pan, SIGNAL(B_changed(double)),
        this, SLOT(redrawOnValueChange(double)));
    connect(pan, SIGNAL(scale_inc(bool)),
        this, SLOT(scale_inc(bool)));
    connect(pan, SIGNAL(scale_dec(bool)),
        this, SLOT(scale_dec(bool)));
}

MainWindow::~MainWindow()
{
}

void MainWindow::redrawOnValueChange(double)
{
    view->update();
}

void MainWindow::scale_inc(bool) {
    auto s = view->getStep();
    if (s.first >= 100 || s.second >= 100) {
        return;
    }
    view->setStep(s.first + 10, s.second + 10);
    view->update();
}

void MainWindow::scale_dec(bool) {
    auto s = view->getStep();
    if (s.first <= 10 || s.second <= 10) {
        return;
    }
    view->setStep(s.first - 10, s.second - 10);
    view->update();
}