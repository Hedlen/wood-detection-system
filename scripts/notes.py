# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'notes.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_Notes(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(705, 437)
        Dialog.setMinimumSize(QtCore.QSize(705, 437))
        Dialog.setMaximumSize(QtCore.QSize(705, 437))
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(3, 10, 701, 421))
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "采集提示"))

