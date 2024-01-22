from os.path import join

import numpy as np  # Matris operations on image
import pyttsx3
from PIL import Image  # Read image,resize,
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from keras.models import load_model  # load cnn models


class Ui_MainWindow(object):
    def setupUi(self, MainWindow, silent_mode_off=False):
        # Filepaths #
        self.image_filepath = join('images', 'no_image.jpg')
        self.cnn_model_path = join('models', 'ai_model.h5')  # CNN (prediction) model path
        self.silent_mode_off = silent_mode_off
        self.model_pred = load_model(self.cnn_model_path)
        self.dim = (32, 32)

        # GUI DESIGN

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(820, 400)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(820, 400))
        MainWindow.setMaximumSize(QtCore.QSize(820, 400))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        MainWindow.setFont(font)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_predict = QtWidgets.QToolButton(self.centralwidget)
        self.btn_predict.setGeometry(QtCore.QRect(550, 150, 91, 31))
        self.btn_predict.setObjectName("btn_predict")
        self.btn_read = QtWidgets.QPushButton(self.centralwidget)
        self.btn_read.setGeometry(QtCore.QRect(660, 150, 91, 31))
        self.btn_read.setObjectName("btn_read")
        self.gbox_prediction = QtWidgets.QGroupBox(self.centralwidget)
        self.gbox_prediction.setGeometry(QtCore.QRect(420, 220, 331, 101))
        self.gbox_prediction.setObjectName("gbox_prediction")
        self.lbl_prediction = QtWidgets.QLabel(self.gbox_prediction)
        self.lbl_prediction.setGeometry(QtCore.QRect(10, 30, 301, 51))
        self.lbl_prediction.setObjectName("lbl_prediction")
        self.gbox_choosemodel = QtWidgets.QGroupBox(self.centralwidget)
        self.gbox_choosemodel.setGeometry(QtCore.QRect(410, 50, 351, 91))
        self.gbox_choosemodel.setObjectName("gbox_choosemodel")
        self.cbox_model = QtWidgets.QComboBox(self.gbox_choosemodel)
        self.cbox_model.setGeometry(QtCore.QRect(20, 40, 311, 41))
        self.cbox_model.setObjectName("cbox_model")
        self.cbox_model.addItem("")
        # self.cbox_model.addItem("")
        # self.cbox_model.addItem("")
        # self.cbox_model.addItem("")
        # self.cbox_model.addItem("")
        self.gbox_image = QtWidgets.QGroupBox(self.centralwidget)
        self.gbox_image.setGeometry(QtCore.QRect(30, 40, 351, 281))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.gbox_image.setFont(font)
        self.gbox_image.setObjectName("gbox_image")
        self.lbl_image = QtWidgets.QLabel(self.gbox_image)
        self.lbl_image.setGeometry(QtCore.QRect(40, 30, 271, 221))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_image.setFont(font)
        self.lbl_image.setText("")
        self.lbl_image.setPixmap(QtGui.QPixmap(self.image_filepath))
        self.lbl_image.setScaledContents(True)
        self.lbl_image.setObjectName("lbl_image")
        self.btn_addImage = QtWidgets.QPushButton(self.centralwidget)
        self.btn_addImage.setGeometry(QtCore.QRect(430, 150, 101, 31))
        self.btn_addImage.setObjectName("btn_addImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.statusbar.setFont(font)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        ## BUTTON CONNECTIONS
        self.btn_predict.clicked.connect(self.to_predictSign)
        self.btn_read.clicked.connect(self.read_voice)
        self.btn_addImage.clicked.connect(self.addImage)
        # BUTTON ENABLED INITIALIZATIONS
        self.btn_read.setEnabled(False)
        self.btn_predict.setEnabled(False)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def returnSignStr(self, index):
        '''
        This function gives us the predictions meanings.
        :param index: Predicted class id or label id
        :return:  What's the id's meaning.
        '''
        all_labels = {0: 'Speed limit is 20km/h',
                      1: 'Speed limit is 30km/h',
                      2: 'Speed limit is 50km/h',
                      3: 'Speed limit is 60km/h',
                      4: 'Speed limit is 70km/h',
                      5: 'Speed limit is 80km/h',
                      6: 'End of speed limit is 80km/h',
                      7: 'Speed limit is 100km/h',
                      8: 'Speed limit is 120km/h',
                      9: 'No passing',
                      10: 'No passing veh is over 3.5 tons',
                      11: 'Right-of-way at intersection',
                      12: 'Priority road',
                      13: 'Yield',
                      14: 'Stop',
                      15: 'No vehicles',
                      16: 'Veh is bigger than 3.5 tons prohibited',
                      17: 'No entry',
                      18: 'General caution',
                      19: 'Dangerous curve left',
                      20: 'Dangerous curve right',
                      21: 'Double curve',
                      22: 'Bumpy road',
                      23: 'Slippery road',
                      24: 'Road narrows on the right',
                      25: 'Road work',
                      26: 'Traffic signals',
                      27: 'Pedestrians',
                      28: 'Children crossing',
                      29: 'Bicycles crossing',
                      30: 'Beware of ice/snow',
                      31: 'Wild animals crossing',
                      32: 'End speed + passing limits',
                      33: 'Turn right ahead',
                      34: 'Turn left ahead',
                      35: 'Ahead only',
                      36: 'Go straight or right',
                      37: 'Go straight or left',
                      38: 'Keep right',
                      39: 'Keep left',
                      40: 'Roundabout mandatory',
                      41: 'End of no passing',
                      42: 'End no passing veh is bigger than 3.5 tons'}
        return all_labels[index]

    def addImage(self):
        '''
        This function is using for add image in GUI interface.
        :return: full filepath of the image is added.
        '''
        filename = QFileDialog.getOpenFileName(None, 'Add Image', '',
                                               'Image File (*.jpg | *.png | *.jpeg)')  # get file full_path array
        self.image_filepath = filename[0]  # get file fullpath
        print("INFO: Image is added & filepath:", self.image_filepath)  # log
        self.lbl_image.setPixmap(QtGui.QPixmap(self.image_filepath))  # load image to show user
        self.lbl_image.setScaledContents(True)  # scale the image
        self.lbl_prediction.setText("There is no prediction for now.")
        self.statusbar.showMessage("INFO: Image is added", 2000)  # info
        self.btn_read.setEnabled(False)
        self.btn_predict.setEnabled(True)

    def to_pixel(self, image_path):
        '''
        This function gives us image' intensity values
        :param image_path: full filepath of image
        :return:  intensity values of the image
        '''
        print("Pixel value turned.")
        img = Image.open(image_path)
        img = img.resize(self.dim)
        img = np.asarray(img) / 255.  # normalize
        img = np.expand_dims(img,
                             axis=0)  # add one more dimension to not facing dimension problem of models inputs shape
        return img

    def read_voice(self):
        # Predicted Traffic sign  is read loudly.
        print("Silent mode off :", self.silent_mode_off)
        if not self.silent_mode_off:
            str_will_read = self.lbl_prediction.text()  # get predicted label meaning
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Engine's talking speed
            engine.say(str_will_read)
            engine.runAndWait()

    def to_predictSign(self):
        # Predict traffic sign.
        print("Predict buton pressed.")
        model_type = self.cbox_model.currentText()  # return model type
        print("MODEL TYPE:", model_type)
        pixel_value = self.to_pixel(self.image_filepath)  # get intensity values
        print("Pixel value shape:", pixel_value.shape)
        # Apply operations in case of model type
        predicted_classid = self.model_pred.predict(pixel_value)[0]  # predict label id probabilities by given model
        predicted_classid = np.argmax(predicted_classid)  # decide label id
        print("Predicted!:", predicted_classid)

        prediction_str = self.returnSignStr(predicted_classid)  # get label_id's meaning
        self.lbl_prediction.setText(str(prediction_str))
        if not self.silent_mode_off:
            self.btn_read.setEnabled(True)

        self.statusbar.showMessage("INFO: Traffic sign is predicted.")
        self.read_voice()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Traffic-Sign Notifier"))
        self.btn_predict.setText(_translate("MainWindow", "Predict"))
        self.btn_read.setText(_translate("MainWindow", "Read"))
        self.gbox_prediction.setTitle(_translate("MainWindow", "Prediction"))
        self.lbl_prediction.setText(_translate("MainWindow", "There is no prediction for now."))
        self.gbox_choosemodel.setTitle(_translate("MainWindow", "Choose a Prediction Model"))
        self.cbox_model.setItemText(0, _translate("MainWindow", "Convolutional Neural Network"))
        # self.cbox_model.setItemText(1, _translate("MainWindow", "Support Vector Machine"))
        # self.cbox_model.setItemText(2, _translate("MainWindow", "Random Forest"))
        # self.cbox_model.setItemText(3, _translate("MainWindow", "MLP"))
        # self.cbox_model.setItemText(4, _translate("MainWindow", "CNN"))
        self.gbox_image.setTitle(_translate("MainWindow", "Image"))
        self.btn_addImage.setText(_translate("MainWindow", "Add Image"))
