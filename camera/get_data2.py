# coding=utf-8
import cv2
import numpy as np
from camera import mvsdk
import platform

import threading
import time
import os
import cv2
import os
import sys
import json
import shutil
import time
import logging
import threading

logging.basicConfig(level=logging.INFO  # 设置日志输出格式
                    , filename="log/demo.log"  # log日志输出的文件位置和文件名
                    , filemode="w", format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"  # 日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                    )


class CameraManager():

    def __init__(self, model, widget_list, showImgCallback):
        self.mvsdk = mvsdk
        self.model = model
        self.widget_list = widget_list
        self.showImgCallback = showImgCallback

    def main_loop(self):
        # 枚举相机
        DevList = self.mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        print('相机数量', nDev)
        pls = []
        for ip in range(nDev):
            p = threading.Thread(target=self.open_camera, args=(ip,))
            p.start()
            pls.append(p)

    def startCamera(self):
        try:
            self.main_loop()
        finally:
            cv2.destroyAllWindows()

    def open_camera(self, ip, code=0):
        save_img_path = './data_b4'
        # os.makedirs()

        # 枚举相机
        DevList = self.mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        # i = 0 if nDev == 1 else int(input("Select camera: "))
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
        # print('fps',fps)
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            try:
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

                # cv2.imwrite(save_img_path + '/' + str(img_i) + '.jpg', frame)
                img_i += 1

                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                # print(ip)
                # model infer
                # img = self.model.inference(frame)
                # if camera_name.split("-")[-1] == "Above":
                self.showImgCallback(frame, self.widget_list[ip])
                #frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)
                # cv2.imshow("Press q to end/" + camera_name + str(ip), frame)
                # if img_i > 20:
                #    break

            except self.mvsdk.CameraException as e:
                # print(e)
                if e.error_code != self.mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

        # 关闭相机
        self.mvsdk.CameraUnInit(hCamera)

        # 释放帧缓存
        self.mvsdk.CameraAlignFree(pFrameBuffer)
