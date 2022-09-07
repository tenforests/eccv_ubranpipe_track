from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np


def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def getFrame():
    a = 0
    Min = 100000.0
    Max = -1.0
    total = 0.0
    list = []
    for filePath in Path('/home/data1/qxh/dataset/dataset_urbanpipe/eccv_data_raw_video/').rglob('*.mp4'):
        cap = cv2.VideoCapture(str(filePath))
        # 获取总帧数，并按照平均间隔获取九帧
        frameNumber = cap.get(7)
        # total += frameNumber
        if frameNumber < Min :
            Min = frameNumber
        if frameNumber > Max :
            Max = frameNumber
        cap.release()
        list.append(frameNumber)
    plt.hist(list,50,rwidth=0.9)
    plt.xlabel("frames")
    plt.ylabel("number")
    plt.show()
    print("min",Min,"max",Max,"mean",np.mean(list),'var',np.var(list),'std',np.std(list))

getFrame()