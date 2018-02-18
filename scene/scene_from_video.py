# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import itertools as it
from tqdm import tqdm as show_progress

from scene import Scene

# 从视频构建场景？
class SceneFromVideo(Scene):
    def construct(self, file_name,
                  freeze_last_frame = True,
                  time_range = None):
        # 使用opencv来读取视频 Capture 捕获
        cap = cv2.VideoCapture(file_name)
        self.shape = (
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        )
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        # 1除以每秒多少帧得到一帧花费的时间
        self.frame_duration = 1.0/fps
        #获取视频有多少帧
        frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        #判断是否有视频时间范围的限制
        if time_range is None:
            start_frame = 0
            end_frame = frame_count
        else:
            start_frame, end_frame = map(lambda t : fps*t, time_range)

        frame_count = end_frame - start_frame
        print("Reading in " + file_name + "...")
        #显示帧的进度
        for count in show_progress(range(start_frame, end_frame+1)):
            returned, frame = cap.read()
            if not returned:
                break
            #将读取帧放入帧的集合
            self.frames.append(frame)
        #释放视频捕获对象
        cap.release()

        #如果要将最后一帧固定住就执行下面的条件
        if freeze_last_frame and len(self.frames) > 0:
            self.original_background = self.background = self.frames[-1]

    #对视频应用高斯模糊
    def apply_gaussian_blur(self, ksize = (5, 5), sigmaX = 5):
        self.frames = [
            cv2.GaussianBlur(frame, ksize, sigmaX)
            for frame in self.frames
        ]

    #对视频应用边缘检测
    def apply_edge_detection(self, threshold1 = 50, threshold2 = 100):
        edged_frames = [
            cv2.Canny(frame, threshold1, threshold2)
            for frame in self.frames
        ]
        for index in range(len(self.frames)):
            for i in range(3):
                self.frames[index][:,:,i] = edged_frames[index]

