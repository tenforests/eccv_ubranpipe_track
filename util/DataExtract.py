# 提取图片
import os
from pathlib import Path
import cv2
import numpy as np


def extractFrame(datasetPath,outputPath):
    for filePath in Path(datasetPath).rglob('*.mp4'):
        #图片文件夹不存在创建
        dir = outputPath+filePath.stem+".mp4"
        if not Path(dir).exists():
            os.makedirs(dir)
        cap = cv2.VideoCapture(str(filePath))
        # 获取总帧数，并按照平均间隔获取九帧
        max_frame_idx = cap.get(7)
        num_frames = 13
        ave_frames_per_group = int(max_frame_idx) // num_frames
        frame_idx = np.arange(0, num_frames) * ave_frames_per_group
        if cap.isOpened():
            a = 1
            for x in frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(x))
                rval,frame = cap.read()
                filename = dir +'/'+ str(a) +'.jpg'
                cv2.imencode('.jpg', frame,[cv2.IMWRITE_JPEG_QUALITY,100])[1].tofile(filename)
                filename = dir +'/'+ str(a) +'.png'
                cv2.imencode('.png', frame,[cv2.IMWRITE_PNG_COMPRESSION,0])[1].tofile(filename)
                a+=1
        print(filePath," done")
        cap.release()

extractFrame("/home/data1/qxh/dataset/dataset_urbanpipe/eccv_data_raw_video/","/home/data1/qxh/dataset/dataset_urbanpipe/data_image_13/")
