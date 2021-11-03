# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QWidget, QMessageBox

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from numpy.lib.function_base import gradient


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        desktop = QApplication.desktop()
        width = desktop.width()
        height = desktop.height()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(width, height)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1280, 720))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(9)
        self.centralwidget.setFont(font)
        self.centralwidget.setStyleSheet("background-color:rgb(248,248,248);")
        self.centralwidget.setObjectName("centralwidget")

        self.HboxLayout11 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.HboxLayout11.setContentsMargins(0, 0, 0, 0)
        self.HboxLayout11.setSpacing(0)
        self.HboxLayout11.setObjectName("gridLayout")

        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")

        self.window_left_main = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_left_main.sizePolicy().hasHeightForWidth())
        self.window_left_main.setSizePolicy(sizePolicy)
        # self.window_left_main.setStyleSheet("background-color:rgb(0,0,255);")
        self.window_left_main.setObjectName("window_left_main")

        self.horizontalLayout_6.addWidget(self.window_left_main)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(20, 148, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem)

        self.cameraStatusHBoxLayout = QtWidgets.QHBoxLayout()
        self.cameraStatusHBoxLayout.addItem(QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        self.cameraStatus = QLabel()
        self.cameraStatus.setFixedSize(40, 40)
        self.cameraStatus.setStyleSheet("background-color:red;\n"
            "border-radius:20%;")

        self.cameraStatusHBoxLayout.addWidget(self.cameraStatus)
 
        self.cameraStatusHBoxLayout.addItem(QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        self.cameraStatusLabel = QLabel("摄像头未连接")
        self.cameraStatusLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(12)
        self.cameraStatusLabel.setFont(font)

        self.cameraStatusHBoxLayout.addWidget(self.cameraStatusLabel)

        self.cameraStatusHBoxLayout.addItem(QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        self.cameraStatusHBoxLayout.setStretch(0, 1)
        self.cameraStatusHBoxLayout.setStretch(1, 1)
        self.cameraStatusHBoxLayout.setStretch(2, 1)
        self.cameraStatusHBoxLayout.setStretch(3, 1)
        self.cameraStatusHBoxLayout.setStretch(4, 1)
        
        self.cameraStatusWidget = QWidget()
        self.cameraStatusWidget.setLayout(self.cameraStatusHBoxLayout)
        self.cameraStatusWidget.setStyleSheet("background-color:rgb(200,200,200);")

        self.verticalLayout_2.addItem(QtWidgets.QSpacerItem(40, 30, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))




        self.verticalLayout_2.addWidget(self.cameraStatusWidget)
        self.verticalLayout_2.addItem(QtWidgets.QSpacerItem(40, 30, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setStyleSheet("border:0px solid rgb(0,0,0);\n")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.verticalLayout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)

        self.topCameraLabel = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.topCameraLabel.setFont(font)
        self.topCameraLabel.setObjectName("topCameraLabel")

        self.horizontalLayout.addWidget(self.topCameraLabel)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)

        self.topCamera = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.topCamera.setFont(font)
        self.topCamera.setObjectName("topCamera")
        self.topCamera.setAlignment(QtCore.Qt.AlignVCenter)
        self.horizontalLayout.addWidget(self.topCamera)

        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 1)
        self.horizontalLayout.setStretch(4, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.leftCameraLabel = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.leftCameraLabel.setFont(font)
        # self.leftCameraLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.leftCameraLabel.setObjectName("leftCameraLabel")
        self.horizontalLayout_2.addWidget(self.leftCameraLabel)

        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.leftCamera = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.leftCamera.setFont(font)
        self.leftCamera.setObjectName("leftCamera")
        self.leftCamera.setAlignment(QtCore.Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.leftCamera)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_2.setStretch(4, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem7)

        self.bottomCameraLabel = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.bottomCameraLabel.setFont(font)
        self.bottomCameraLabel.setObjectName("bottomCameraLabel")
        self.horizontalLayout_3.addWidget(self.bottomCameraLabel)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem8)
        self.bottomCamera = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.bottomCamera.setFont(font)
        self.bottomCamera.setObjectName("bottomCamera")
        self.bottomCamera.setAlignment(QtCore.Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.bottomCamera)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem9)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 1)
        self.horizontalLayout_3.setStretch(4, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_8.addItem(spacerItem7)
        self.rightCameraLabel = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.rightCameraLabel.setFont(font)
        # self.rightCameraLabel.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.rightCameraLabel.setObjectName("rightCameraLabel")
        self.horizontalLayout_8.addWidget(self.rightCameraLabel)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem8)
        self.rightCamera = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.rightCamera.setFont(font)
        self.rightCamera.setObjectName("rightCamera")
        self.rightCamera.setAlignment(QtCore.Qt.AlignVCenter)

        self.horizontalLayout_8.addWidget(self.rightCamera)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem9)
        self.horizontalLayout_8.setStretch(0, 1)
        self.horizontalLayout_8.setStretch(1, 1)
        self.horizontalLayout_8.setStretch(2, 1)
        self.horizontalLayout_8.setStretch(3, 1)
        self.horizontalLayout_8.setStretch(4, 2)

        self.verticalLayout.addLayout(self.horizontalLayout_8)


        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_10.addItem(spacerItem7)
        self.allCameraLabel = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.allCameraLabel.setFont(font)
        # self.allCameraLabel.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.allCameraLabel.setObjectName("allCameraLabel")
        self.horizontalLayout_10.addWidget(self.allCameraLabel)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem8)
        self.allCamera = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.allCamera.setFont(font)
        self.allCamera.setObjectName("allCamera")
        self.allCamera.setAlignment(QtCore.Qt.AlignVCenter)

        self.horizontalLayout_10.addWidget(self.allCamera)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem9)
        self.horizontalLayout_10.setStretch(0, 1)
        self.horizontalLayout_10.setStretch(1, 1)
        self.horizontalLayout_10.setStretch(2, 1)
        self.horizontalLayout_10.setStretch(3, 1)
        self.horizontalLayout_10.setStretch(4, 2)

        self.verticalLayout.addLayout(self.horizontalLayout_10)

        spacerItem10 = QtWidgets.QSpacerItem(20, 107, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem10)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(6, 2)

        self.verticalLayout_2.addWidget(self.groupBox)

        spacerItem17 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem17)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        spacerItem11 = QtWidgets.QSpacerItem(60, 60, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem11)
        self.cameraControl = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraControl.sizePolicy().hasHeightForWidth())
        self.cameraControl.setSizePolicy(sizePolicy)
        self.cameraControl.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        self.cameraControl.setFont(font)
        self.cameraControl.setStyleSheet("border:1px solid rgb(0,0,0);\n"
                                         "background-color:rgb(220, 228, 253);\n"
                                         "border-radius:20%;")
        self.cameraControl.setObjectName("cameraControl")
        self.horizontalLayout_4.addWidget(self.cameraControl)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem12)
        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem16)

        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_9.setStretch(4, 2)

        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem15)

        self.reset = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.reset.sizePolicy().hasHeightForWidth())
        self.reset.setSizePolicy(sizePolicy)
        self.reset.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        self.reset.setFont(font)
        self.reset.setStyleSheet("border:1px solid rgb(0,0,0);\n"
                                 "background-color:rgb(220, 228, 253);\n"
                                 "border-radius:10%;")
        self.reset.setObjectName("reset")
        self.horizontalLayout_9.addWidget(self.reset)
        self.horizontalLayout_9.setStretch(0, 5)
        self.horizontalLayout_9.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem13)

        self.verticalLayout_2.setStretch(0, 1)  # space
        self.verticalLayout_2.setStretch(1, 1)  # camera status
        self.verticalLayout_2.setStretch(2, 1)  # space
        self.verticalLayout_2.setStretch(3, 4)  # group box
        self.verticalLayout_2.setStretch(4, 1)  # space
        self.verticalLayout_2.setStretch(5, 1)  # camera button
        self.verticalLayout_2.setStretch(6, 1)  # space
        self.verticalLayout_2.setStretch(7, 1)  # reset button
        self.verticalLayout_2.setStretch(8, 1)  # space

        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        spacerItem14 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem14)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 10)
        self.horizontalLayout_5.setStretch(2, 1)

        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6.setStretch(0, 5)
        self.horizontalLayout_6.setStretch(1, 2)
        self.HboxLayout11.addLayout(self.horizontalLayout_6)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "工业视觉缺陷检测"))
        self.topCameraLabel.setText(_translate("MainWindow", "上方缺陷数量(处):"))
        self.topCamera.setText(_translate("MainWindow", "0"))
        self.leftCameraLabel.setText(_translate("MainWindow", "左侧缺陷数量(处):"))
        self.leftCamera.setText(_translate("MainWindow", "0"))
        self.bottomCameraLabel.setText(_translate("MainWindow", "下方缺陷数量(处):"))
        self.bottomCamera.setText(_translate("MainWindow", "0"))
        self.rightCameraLabel.setText(_translate("MainWindow", "右侧缺陷数量(处):"))
        self.rightCamera.setText(_translate("MainWindow", "0"))
        self.allCameraLabel.setText(_translate("MainWindow", "合计缺陷数量(处):"))
        self.allCamera.setText(_translate("MainWindow", "0"))
        self.cameraControl.setText(_translate("MainWindow", " 打开摄像头 "))
        self.reset.setText(_translate("MainWindow", "重置"))
        # self.menunutMachine.setTitle(_translate("MainWindow", "工业视觉缺陷检测"))
