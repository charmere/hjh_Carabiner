import copy
import platform
import queue
import threading
import cv2
import os
import sys
import json

import numpy as np
from qtUI.mainWindow import Ui_MainWindow
from threading import Thread, Semaphore, Lock
from utils.utils import letterbox, _async_raise
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.Qt import QCursor
from functools import partial
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore, QtGui
import shutil
import time
import logging
from model.infer import Yolo
from camera.get_data2 import CameraManager
from camera import mvsdk
from queue import Queue

logging.basicConfig(level=logging.INFO  # 设置日志输出格式
                    , filename="log/demo.log"  # log日志输出的文件位置和文件名
                    , filemode="w", format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"  # 日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                    )

global_img_list = [None, None, None, None]

class StartUI():

    def __init__(self):

        self.open_camera_ = 0
        app = QApplication(sys.argv)
        self._main_window = QMainWindow()
        self._main_ui = Ui_MainWindow()
        self._main_ui.setupUi(self._main_window)
        self._main_window.setWindowFlags(QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self._main_window.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        self._main_window.showMaximized()
        self.__setImgWhenInit()
        self._main_window.show()
        logging.info("ui init successed.")
        self.pool = Semaphore(3)
        self._camera_status = False
        self.cap = cv2.VideoCapture()
        self._setCap()
        self.pls = []
        self.CAM_NUM = -1
        self._infer_img_num = 0

        # 图像获取方式 2 表示 从传感器获取 0 表示不间断获取
        self.state = 0

        self.yolo = Yolo()

        self.mvsdk = mvsdk
        self._bindClickedEvent()
        self.lock = Lock()

        if app.exec_() == 0:
            # for pl in self.pls:
            #     try:
            #         _async_raise(pl.ident, SystemExit)
            #     except Exception as e:
            #         logging.info(e)
            self.open_camera_ = 0
            self._main_window.close()
            time.sleep(0.2)
            sys.exit(0)


    def __setImgWhenInit(self):
        img_path = "./backgroundImages/1.jpg"
        cv_img = cv2.imread(img_path)
        logging.info("update img")

        width = self._main_ui.window_left_main.width()
        height = self._main_ui.window_left_main.height()
        
        cv_img = cv2.resize(cv_img, (width, height))
        src_img = letterbox(cv_img, new_shape=(height, width))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        temp_imgSrc = QImage(
            src_img,
            src_img.shape[1],
            src_img.shape[0],
            src_img.shape[1] * 3,
            QImage.Format_RGB888,
        )
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(width, height)

        self._main_ui.window_left_main.setPixmap(pixmap_imgSrc)

    def _bindClickedEvent(self):

        self._main_ui.cameraControl.clicked.connect(self.open_camera)
        self._main_ui.reset.clicked.connect(self._resetInforMation)
        logging.info("bind event to wdigets.")

    def _left_label_img(self):
        img_path = "./backgroundImages/1.jpg"
        cv_img = cv2.imread(img_path)
        logging.info("update img")

        width = self._main_ui.window_left_main.width()
        height = self._main_ui.window_left_main.height()

        cv_img = cv2.resize(cv_img, (width, height))
        src_img = letterbox(cv_img, new_shape=(height, width))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        temp_imgSrc = QImage(
            src_img,
            src_img.shape[1],
            src_img.shape[0],
            src_img.shape[1] * 3,
            QImage.Format_RGB888,
        )
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(width, height)

        self._main_ui.window_left_main.setPixmap(pixmap_imgSrc)

    def _setCap(self):

        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)
        self.cap.set(22, 72)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        logging.info("set camera parameter.")

    def _updateCameraStatus(self, color, msg):

        self._main_ui.cameraStatus.setStyleSheet("border-radius:20%;\n"
                                                 "background-color:{0};".format(str(color)))
        self._main_ui.videoMsg.setText(msg)

    def updateCameraStatus(self, state="open"):
        if state == "open":
            self._main_ui.cameraStatusLabel.setText("摄像头已连接")
            self._main_ui.cameraStatus.setStyleSheet("border-radius:20%;\n"
                                                     "background-color:green")
        elif state == "close":
            self._main_ui.cameraStatusLabel.setText("摄像头已断开")
            self._main_ui.cameraStatus.setStyleSheet("border-radius:20%;\n"
                                                     "background-color:red")




    def _resetInforMation(self):
        self.closeCamera()
        self.lock.acquire()
        global global_img_list
        global_img_list = [None, None, None, None]
        self.lock.release()

        self._main_ui.topCamera.setText("0")
        self._main_ui.leftCamera.setText("0")
        self._main_ui.rightCamera.setText("0")
        self._main_ui.bottomCamera.setText("0")
        self._main_ui.allCamera.setText("0")
        self._main_ui.cameraStatusLabel.setText("摄像头已断开")
        self._main_ui.cameraStatus.setStyleSheet("border-radius:20%;\n"
                                                 "background-color:red")
        self._main_ui.window_left_main.clear()
        try:
            _async_raise(self.showImgThread.ident, SystemExit)

        except Exception as e:
            logging.info(e)

        try:
            _async_raise(self.startInference.ident, SystemExit)
        except Exception as e:
            logging.info(e)

        try:
            _async_raise(self.showImgThread.ident, SystemExit)
        except Exception as e:
            logging.info(e)
        self.__setImgWhenInit()

    def closeCamera(self):
        self.open_camera_ = 0
        time.sleep(0.2)
        self._main_ui.cameraControl.disconnect()
        self._main_ui.cameraControl.clicked.connect(self.open_camera)
        self._main_ui.cameraControl.setText(" 打开摄像头 ")
        self._main_ui.cameraStatusLabel.setText("摄像头已断开")
        self._main_ui.cameraStatus.setStyleSheet("border-radius:20%;\n"
                                                 "background-color:red")



    def _showImgFromModel(self, cv_img_list, resultFromModel_list):
        if type(cv_img_list[0]) is np.ndarray and type(cv_img_list[1]) is np.ndarray and type(cv_img_list[2]) is np.ndarray and type(cv_img_list[3]) is np.ndarray:

            img1 = np.hstack((cv_img_list[0], cv_img_list[1]))
            img2 = np.hstack((cv_img_list[2], cv_img_list[3]))
            cv_img = np.vstack((img1, img2))
            width = self._main_ui.window_left_main.width()
            height = self._main_ui.window_left_main.height()
            src_img = letterbox(cv_img, new_shape=(height, width))
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            temp_imgSrc = QImage(
                src_img,
                src_img.shape[1],
                src_img.shape[0],
                src_img.shape[1] * 3,
                QImage.Format_RGB888,
            )

            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(width, height)
            src_num1 = int(self._main_ui.topCamera.text())
            dst_num1 = resultFromModel_list[0] + src_num1

            src_num2 = int(self._main_ui.leftCamera.text())
            dst_num2 = resultFromModel_list[0] + src_num2

            src_num3 = int(self._main_ui.bottomCamera.text())
            dst_num3 = resultFromModel_list[0] + src_num3

            src_num4 = int(self._main_ui.rightCamera.text())
            dst_num4 = resultFromModel_list[0] + src_num4

            src_all = int(self._main_ui.allCamera.text())
            dst_all = sum(resultFromModel_list) + src_all

            if self.open_camera_ == 1:

                self._main_ui.topCamera.setText(str(dst_num1))
                self._main_ui.leftCamera.setText(str(dst_num2))
                self._main_ui.bottomCamera.setText(str(dst_num3))
                self._main_ui.rightCamera.setText(str(dst_num4))
                self._main_ui.allCamera.setText(str(dst_all))

                self._main_ui.window_left_main.setPixmap(pixmap_imgSrc)


    def open_camera(self):

        if self.mvsdk.CAMERA_STATUS_DEVICE_IS_OPENED == -18:
            self._main_ui.cameraControl.disconnect()
            self._main_ui.cameraControl.setText(" 关闭摄像头 ")
            self._main_ui.cameraControl.clicked.connect(self.closeCamera)
            self.updateCameraStatus("open")
            self.open_camera_ = 1
            DevList = self.mvsdk.CameraEnumerateDevice()
            nDev = len(DevList)
            self.pls = []
            for ip in range(nDev):
                p = threading.Thread(target=self.open_camera_thread, args=(ip, self.state, ))
                # p.setDaemon(True)
                p.start()
                self.pls.append(p)

            self.startInference = Thread(target=self.startInferenceFunc)
            self.startInference.setDaemon(True)
            self.startInference.start()
        else:
            QtWidgets.QMessageBox.warning(self._main_window, "提示", "摄像头正在使用")

    def startInferenceFunc(self):

        global global_img_list

        while True:
            try:
                if None not in global_img_list:
                    pass
                else:
                    # print('>>>>>>>>',global_img_list)
                    pass
            except Exception as e:
                if self.open_camera_:
                    self.lock.acquire()
                    l = copy.deepcopy(global_img_list)
                    global_img_list = [None, None, None, None]
                    Thread(target=self.inferAndShow, args=(l, )).start()
                    self.lock.release()

    def inferAndShow(self, l):

        img_list, result_list = self.yolo.inference(l)
        self.showImgThread = Thread(target=self._showImgFromModel, args=(img_list, result_list, ))
        self.showImgThread.start()



    def open_camera_thread(self, ip, code=2):
        save_img_path = './camera/data_b4'
        # 枚举相机
        DevList = self.mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)


        DevInfo = DevList[ip]

        camera_name = DevInfo.GetFriendlyName()
        # 创建文件夹
        save_img_path = os.path.join(save_img_path, camera_name)
        if not os.path.isdir(save_img_path):
            os.makedirs(save_img_path)
        img_i = len(os.listdir(save_img_path))

        # 打开相机
        hCamera = 0
        try:
            hCamera = self.mvsdk.CameraInit(DevInfo, -1, -1)
        except self.mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = self.mvsdk.CameraGetCapability(hCamera)
        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            self.mvsdk.CameraSetIspOutFormat(hCamera, self.mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            self.mvsdk.CameraSetIspOutFormat(hCamera, self.mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        #self.mvsdk.CameraSetTriggerMode(hCamera, code)

        # 软件触发
        #self.mvsdk.CameraSetTriggerMode(hCamera, code)
        # 硬件触发
        self.mvsdk.CameraSetTriggerMode(hCamera, code)

        # 设置Gain
        self.mvsdk.CameraSetAnalogGain(hCamera, 3)

        # 手动曝光，曝光时间30ms
        self.mvsdk.CameraSetAeState(hCamera, 0)
        self.mvsdk.CameraSetExposureTime(hCamera, 30 * 100)

        # CameraSetFrameSpeed(hCamera)
        #fps = self.mvsdk.CameraGetFrameSpeed(hCamera)
        # print('fps',fps)

        # 让SDK内部取图线程开始工作
        self.mvsdk.CameraPlay(hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        pFrameBuffer = self.mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        fps = self.mvsdk.CameraGetFrameSpeed(hCamera)
        global global_img_list
        while self.open_camera_:
            try:
                if not quene.empty():
                    break
                pRawData, FrameHead = self.mvsdk.CameraGetImageBuffer(hCamera, 200)
                self.mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                self.mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

                # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
                # linux下直接输出正的，不需要上下翻转
                if platform.system() == "Windows":
                    self.mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
                frame_data = (self.mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == self.mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

                self.lock.acquire()
                if camera_name.split("-")[-1] == "Above":
                    global_img_list[0] = frame
                elif camera_name.split("-")[-1] == "Front":
                    global_img_list[1] = frame
                elif camera_name.split("-")[-1] == "Left":
                    global_img_list[2] = frame
                elif camera_name.split("-")[-1] == "Right":
                    global_img_list[3] = frame
                self.lock.release()

            except self.mvsdk.CameraException as e:

                if e.error_code != self.mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

        # 关闭相机
        self.mvsdk.CameraUnInit(hCamera)
        self.mvsdk.CameraAlignFree(pFrameBuffer)


if __name__ == "__main__":

    quene = Queue(5)
    ui = StartUI()
