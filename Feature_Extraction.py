import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PyQt5 import Qt, uic, QtWidgets


class Proj(QtWidgets.QDialog):
    def __init__(self) -> None:
        """Initialize"""
        # Loading UI form
        super(Proj, self).__init__()

        # self.setWindowTitle('Image Viewer')
        uic.loadUi('C:\mahmoud\My Work\Gasco\Machine learning and AI\computer vision\QT5 Projects\sift and harris/Feature_Extraction_Pyqt.ui',
                   self)
        self.layout = QVBoxLayout()


        self.originview = QLabel(self.imglabel)
        self.originview.setFixedSize(291, 350)
        self.Show_Img.clicked.connect(self.openFileDialog)
       

        self.Resview = QLabel(self.Reslabel_1)
        self.Resview.setFixedSize(291, 350)
        self.Show_Res_Img_harris.clicked.connect(self.applyharris)
        self.Show_Res_Img_sift.clicked.connect(self.applysift)
        self.image  = None


    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                  options=options)
        if fileName:
            self.image  = cv2.imread(fileName)
            self.displayImage(self.image, self.originview)

    def   applyharris(self):
        if self.image is None:
            print('No image loaded' )  
        else:
             gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
             gray_image = np.float32(gray_image)
             corner_img = cv2.cornerHarris(gray_image,7,5,self.Hth.value())
             corner_img = cv2.normalize(corner_img, None, 0, 255, cv2.NORM_MINMAX)
             corner_img = np.uint8(corner_img)
             self.displayImage(corner_img, self.Resview)
    def   applysift(self):
        if self.image is None:
            print('No image loaded' )  
        else:
             sift = cv2.SIFT_create()       
             gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
             kp = sift.detect(gray_image,None)
             img=cv2.drawKeypoints(gray_image,kp,self.image)
             self.displayImage(img, self.Resview)

    

        

    def displayImage(self, img, label):
        if len(img.shape)==2:
            height,width =img.shape
            bytesPerLine =width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            label.setPixmap(QPixmap.fromImage(qImg))
        else:
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            label.setPixmap(QPixmap.fromImage(qImg)


)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Proj()
    viewer.show()
    sys.exit(app.exec_())

