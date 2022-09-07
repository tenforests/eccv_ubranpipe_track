import json
import os
from pathlib import Path
import cv2
import numpy as np
from skmultilearn.model_selection import iterative_stratification,IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# 需要搞一个数据加载器，用于加载标签和读取视频


def loadTrainData(JSONpath):
    labels = []
    filePaths = []
    with open(JSONpath,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
    for key in json_data:
        # filePaths.append(str(Path(key).stem))
        filePaths.append(key)
        labels.append(json_data[key])
    # print(filePaths[0])
    # print(labels[0])
    return labels,filePaths

def loadTestData(JSONpath):
    filePaths = []
    with open(JSONpath,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
    for filePath in json_data['test_video_name']:
        filePaths.append(str(filePath))
    # print(filePaths[0])
    return filePaths

# 提取图片
def extractFrame(datasetPath,outputPath):
    for filePath in Path(datasetPath).rglob('*.mp4'):
        #图片文件夹不存在创建
        dir = outputPath+filePath.stem
        if not Path(dir).exists():
            os.mkdirs(dir)

        cap = cv2.VideoCapture(str(filePath))
        # 获取总帧数，并按照平均间隔获取九帧
        FrameNumber = cap.get(7)
        gap = FrameNumber / 8
        if cap.isOpened():
            a = 0
            for x in range(0,int(FrameNumber),int(gap)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(x))
                retval,frame = cap.read()
                filename = dir +'/'+ str(a) +'.jpg'
                cv2.imencode('.jpg', frame)[1].tofile(filename)
                a+=1
        cap.release()


# 统计原始label中各label数量
def count_class_num(labels):
    class_num = {}
    for label in labels:
        for x in range(17):
            if label[x] == 1:
                if x in class_num.keys():
                    class_num[x] += 1
                else:
                    class_num[x] = 1
    class_num = dict(sorted(class_num.items(),key=lambda  c:c[0]))
    class_num = list(class_num.values())
    return class_num

# 划分训练，验证数据集
def split(labels,filePaths,percent):
    return iterative_stratification.iterative_train_test_split(filePaths,labels,percent)

def split2():
    k_fold = IterativeStratification(n_splits=5, order=1)
    return k_fold


def getFrame(filePaths):
    frames = {}
    for filePath in filePaths:
        cap = cv2.VideoCapture('/home/data1/qxh/dataset/dataset_urbanpipe/eccv_data_raw_video/'+filePath)
        # 获取总帧数，并按照平均间隔获取九帧
        frameNumber = cap.get(7)
        frames[filePath] = int(frameNumber)
        cap.release()
        print(filePath,'done')
    return frames

def generateNewTextFile(txtPath,filePaths,labels,frames):
    print("beginGenerateText")
    with open(txtPath,"w") as f:
        if labels is None:
            for filePath in filePaths:
                f.write(filePath+' 1 '+str(frames[filePath])+' '+'\n')
        else:
            for filePath,label in zip(filePaths,labels):
                labelString = ''
                for x in range(0,17):
                    if label[x] == 1:
                        labelString +=str(x)
                        labelString +=' '
                f.write(filePath+' 1 '+str(frames[filePath])+' '+labelString+' '+'\n')



def rename(path):
    a = 0
    dirs = os.listdir(path)
    for dir in dirs:
        os.rename(os.path.join(path,dir),os.path.join(path,str(dir)+".mp4"))
        print(a,"done")
        a+= 1


def checkNum(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if os.path.isdir(os.path.join(path,dir)):
            i_list = os.listdir(os.path.join(path,dir))
            x = len(i_list)
            if x < 9 :
                print(dir,x)
                for i in i_list:
                    print(i)


def split3():
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return mskf


# test_list = loadTestData("./sifar-pytorch-main/test.json")
# frames = getFrame(test_list)
# generateNewTextFile("./test.txt",test_list,None,frames)
# checkNum('/home/data1/qxh/dataset/dataset_urbanpipe/data_image/')
# rename("/home/data1/qxh/dataset/dataset_urbanpipe/data_image/")
labels,filePaths = loadTrainData("/home/data1/qxh/dataset/dataset_urbanpipe/annotations/train.json")
# labels,filePaths = loadTrainData("./train.json")
labels = MultiLabelBinarizer().fit_transform(labels)
filePaths = np.array(filePaths)
# X_train, y_train, X_test, y_test = split(labels, np.array(filePaths).reshape(-1, 1), 0.2)
frames = getFrame(filePaths)
generateNewTextFile("./train.txt",filePaths,labels,frames)

# k_fold = split2()
# k_fold = split3()
# i = 1
# filePaths = np.array(filePaths).reshape(-1, 1)
# for train, test in k_fold.split(filePaths, labels):
#     X_train = filePaths[train]
#     print(type(X_train))
#     X_test = filePaths[test]
#     print(type(X_test))
#     y_train = labels[train]
#     print(type(y_train))
#     y_test = labels[test]
#     print(type(y_test))
#     X_train = X_train.flatten()
#     X_test = X_test.flatten()
# # frames = getFrame(['187.mp4','3010.mp4'])
# # print(frames)
# # generateNewTextFile("./train.txt",np.array(['187.mp4','3010.mp4']),np.array([[0,0,0,1,1],[1,0,0,1,1]]),frames)
#     generateNewTextFile("/home/data1/qxh/dataset/dataset_urbanpipe/5_split2/split"+str(i)+"_train.txt",X_train,y_train,frames)
#     generateNewTextFile("/home/data1/qxh/dataset/dataset_urbanpipe/5_split2/split"+str(i)+"_val.txt",X_test, y_test,frames)
#     i+=1
print("done")
# labels,filePaths = loadTrainData("./train.json")
# labels = MultiLabelBinarizer().fit_transform(labels)
# filePaths = np.array(filePaths)
# filePaths = filePaths.reshape(-1, 1)
# X_train, y_train, X_test, y_test = split(labels, filePaths, 0.2)
# print(count_class_num(labels))
# print(count_class_num(y_train))


