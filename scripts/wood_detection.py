# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'wood_detection.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.setEnabled(True)
        dialog.resize(1251, 542)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(dialog.sizePolicy().hasHeightForWidth())
        dialog.setSizePolicy(sizePolicy)
        dialog.setMinimumSize(QtCore.QSize(1251, 524))
        dialog.setMaximumSize(QtCore.QSize(1251, 550))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(15)
        dialog.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../../../../.designer/backup/icon.jpeg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap("../../../../../../.designer/backup/icon.jpeg"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        dialog.setWindowIcon(icon)
        self.pushButton = QtWidgets.QPushButton(dialog)
        self.pushButton.setGeometry(QtCore.QRect(1130, 236, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(15)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(1130, 281, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(15)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(dialog)
        self.label.setGeometry(QtCore.QRect(1116, 316, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(dialog)
        self.textEdit.setGeometry(QtCore.QRect(1104, 350, 141, 191))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(11)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.radioButton = QtWidgets.QRadioButton(dialog)
        self.radioButton.setGeometry(QtCore.QRect(1110, 38, 112, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(dialog)
        self.radioButton_2.setGeometry(QtCore.QRect(1110, 68, 112, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(dialog)
        self.radioButton_3.setGeometry(QtCore.QRect(1110, 98, 112, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.lineEdit = QtWidgets.QLineEdit(dialog)
        self.lineEdit.setEnabled(True)
        self.lineEdit.setGeometry(QtCore.QRect(1193, 99, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lineEdit.setFont(font)
        self.lineEdit.setAutoFillBackground(False)
        self.lineEdit.setDragEnabled(False)
        self.lineEdit.setReadOnly(False)
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(dialog)
        self.label_2.setGeometry(QtCore.QRect(1190, 80, 66, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.radioButton_4 = QtWidgets.QRadioButton(dialog)
        self.radioButton_4.setGeometry(QtCore.QRect(1110, 10, 112, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.pushButton_3 = QtWidgets.QPushButton(dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(1198, 164, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(1152, 134, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(1152, 195, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(dialog)
        self.pushButton_6.setGeometry(QtCore.QRect(1106, 164, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.widget = QtWidgets.QWidget(dialog)
        self.widget.setGeometry(QtCore.QRect(-1, -1, 1101, 551))
        self.widget.setObjectName("widget")
        self.graphicsView = QtWidgets.QGraphicsView(self.widget)
        self.graphicsView.setGeometry(QtCore.QRect(7, 12, 540, 531))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.widget)
        self.graphicsView_2.setGeometry(QtCore.QRect(560, 12, 540, 531))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setGeometry(QtCore.QRect(553, 12, 3, 531))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.widget_2 = QtWidgets.QWidget(dialog)
        self.widget_2.setGeometry(QtCore.QRect(-1, 0, 1101, 551))
        self.widget_2.setObjectName("widget_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.widget_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(8, 10, 1091, 531))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(dialog)
        self.lineEdit_2.setEnabled(True)
        self.lineEdit_2.setGeometry(QtCore.QRect(1193, 59, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setAutoFillBackground(False)
        self.lineEdit_2.setDragEnabled(False)
        self.lineEdit_2.setReadOnly(False)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_3 = QtWidgets.QLabel(dialog)
        self.label_3.setGeometry(QtCore.QRect(1189, 40, 66, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton_7 = QtWidgets.QPushButton(dialog)
        self.pushButton_7.setGeometry(QtCore.QRect(1192, 14, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "木头检测系统"))
        self.pushButton.setText(_translate("dialog", "输入"))
        self.pushButton_2.setText(_translate("dialog", "检测"))
        self.label.setText(_translate("dialog", "   检测信息"))
        self.radioButton.setText(_translate("dialog", "单图"))
        self.radioButton_2.setText(_translate("dialog", "视频"))
        self.radioButton_3.setText(_translate("dialog", "多视图"))
        self.label_2.setText(_translate("dialog", "拼接顺序"))
        self.radioButton_4.setText(_translate("dialog", "拍摄"))
        self.pushButton_3.setText(_translate("dialog", "保存"))
        self.pushButton_4.setText(_translate("dialog", "播放"))
        self.pushButton_5.setText(_translate("dialog", "重置"))
        self.pushButton_6.setText(_translate("dialog", "暂停"))
        self.label_3.setText(_translate("dialog", "采集排数"))
        self.pushButton_7.setText(_translate("dialog", "设置"))

