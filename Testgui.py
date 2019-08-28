import numpy as np
import sys
#from gsp import GstreamerPlayer
import datetime
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLayout, QDialog, QApplication, QMainWindow, QFileDialog, QPushButton, QWidget, QLabel, QButtonGroup
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

class picker(QDialog):
     def __init__(self):
         super(picker, self).__init__()
         loadUi('main.ui', self)
         self.setWindowTitle("Picker Efficiency Calculator")
         self.setWindowIcon(QtGui.QIcon('index.png'))
         #self.webEngineView.show(QUrl(url))
         self.map_button.clicked.connect(self.openFileDialogM)
         self.FIT_button.clicked.connect(self.openFileDialogF)
         self.evaluate_button.clicked.connect(self.openResult)
         #self.pushButton_2.clicked.connect(self.openMain)
         self.show()

     def openFileDialogM(self):
         self.filepath_map = QFileDialog.getOpenFileName(self,'Multiple File','*.kml')
         f = "".join(self.filepath_map[0])
         self.map_line.setText(f)

     def openFileDialogF(self):
         self.filepath_FIT = QFileDialog.getOpenFileName(self,'Multiple File','*.FIT')
         f = "".join(self.filepath_FIT[0])
         self.FIT_line.setText(f)
     
     def openResult(self,filepath):
         #window.hide()
         super(picker, self).__init__()
         loadUi('result.ui', self)
         self.setWindowTitle("Picker Efficiency Result")
         self.setWindowIcon(QtGui.QIcon('index.png'))
         self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = picker()
    window.show()
    sys.exit(app.exec_())