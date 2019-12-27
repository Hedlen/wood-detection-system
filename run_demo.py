from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from scripts.detection import Detection_Ui
import sys


if  __name__=="__main__":
    print("Starting")
    app = QApplication(sys.argv)
    w = Detection_Ui()
    w.show()
    sys.exit(app.exec_())
