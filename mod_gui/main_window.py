# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import sys


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, 
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import QDir, Qt, QUrl, QSize


from .video_player import *


from mod_speech import mfcc2
from mod_blink import lie_blink

class Ui_MainWindow(QWidget):
    fileName=""
    res=""
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_name = QtWidgets.QLabel(self.centralwidget)
        self.label_name.setGeometry(QtCore.QRect(150, 180, 55, 16))
        self.label_name.setObjectName("label_name")
        self.label_age = QtWidgets.QLabel(self.centralwidget)
        self.label_age.setGeometry(QtCore.QRect(150, 230, 55, 16))
        self.label_age.setObjectName("label_age")
        self.label_dob = QtWidgets.QLabel(self.centralwidget)
        self.label_dob.setGeometry(QtCore.QRect(150, 280, 55, 16))
        self.label_dob.setObjectName("label_dob")
        
        self.le_name = QtWidgets.QLineEdit(self.centralwidget)
        self.le_name.setGeometry(QtCore.QRect(260, 180, 251, 22))
        self.le_name.setObjectName("le_name")
        

        self.le_age = QtWidgets.QLineEdit(self.centralwidget)
        self.le_age.setGeometry(QtCore.QRect(260, 230, 251, 22))
        self.le_age.setObjectName("le_age")
        self.le_dob = QtWidgets.QLineEdit(self.centralwidget)
        self.le_dob.setGeometry(QtCore.QRect(260, 280, 251, 22))
        self.le_dob.setObjectName("le_dob")

        self.btn_submit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_submit.setGeometry(QtCore.QRect(310, 410, 93, 28))
        self.btn_submit.setObjectName("btn_submit")
        self.btn_submit.clicked.connect(self.new_window)
        # self.btn_submit.clicked.connect(self.run_code) #run all the algos

        self.label_reg_form = QtWidgets.QLabel(self.centralwidget)
        self.label_reg_form.setGeometry(QtCore.QRect(270, 50, 251, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_reg_form.setFont(font)
        self.label_reg_form.setObjectName("label_reg_form")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(340, 440, 55, 16))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_upload_video = QtWidgets.QLabel(self.centralwidget)
        self.label_upload_video.setGeometry(QtCore.QRect(140, 330, 81, 16))
        self.label_upload_video.setObjectName("label_upload_video")
        self.label_path_video = QtWidgets.QLabel(self.centralwidget)
        self.label_path_video.setGeometry(QtCore.QRect(260, 330, 251, 16))
        self.label_path_video.setText("")
        self.label_path_video.setObjectName("label_path_video")

        self.btn_open_video = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_video.setGeometry(QtCore.QRect(530, 330, 93, 28))
        self.btn_open_video.setToolTip("Open Video File")
        self.btn_open_video.setStatusTip("Open Video File")
        self.btn_open_video.setObjectName("btn_open_video")
        self.btn_open_video.clicked.connect(self.abrir)

        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_name.setText(_translate("MainWindow", "Name"))
        self.label_age.setText(_translate("MainWindow", "Age"))
        self.label_dob.setText(_translate("MainWindow", "DOB"))
        self.btn_submit.setText(_translate("MainWindow", "Submit"))
        self.label_reg_form.setText(_translate("MainWindow", "Registration Form"))
        self.label_upload_video.setText(_translate("MainWindow", "Upload Video"))
        self.btn_open_video.setText(_translate("MainWindow", "Browse"))

    def abrir(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Selecciona los mediose",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if self.fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self.fileName)))
            #self.playButton.setEnabled(True)
            #print(fileName)
            # f = self.fileName.split("/")[-1]
            self.label_path_video.setText(self.fileName)
            #self.play()

    def run_code(self, vplayer):
        print('run func started')
        self.res = lie_blink.main(self.fileName)
        self.res_sp = mfcc2.main(self.fileName)
        print('result obtained')

        vplayer.enable_result(res=self.res)


    def new_window(self):
        
        name=self.le_name.text()
        full_path = self.fileName
        print('before VideoPlayer, res=', self.res)
        self.player = VideoPlayer(name, file_path=full_path)

        

        self.player.setWindowTitle("Player")
        self.player.resize(900, 700)
        self.player.show()
        
        # start eye code
        self.run_code(vplayer=self.player)
        print('eye run code return')




if __name__ == "__main__":
    app1 = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app1.exec_())

def main():
    app1 = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app1.exec_())
