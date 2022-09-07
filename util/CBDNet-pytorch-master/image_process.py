import os, time, scipy.io, shutil
from pathlib import Path

from PIL import Image
from torchvision.transforms import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import glob
from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw
import imgaug.augmenters as iaa

# rootpath=os.path.abspath(os.path.join(os.getcwd(), "../"))


datasmap=[
         ['/home/data1/qxh/dataset/dataset_urbanpipe/data_image','total.txt',
          'data_image','data_image_Denoising',]
         ]




#
def adjust_gamma(image, gamma=3.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def img_shapen(img):
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1, 2, 2, 2, -1],
                       [-1, 2, 8, 2, -1],
                       [-2, 2, 2, 2, -1],
                       [-1, -1, -1, -1, -1]])/8
    output=cv2.filter2D(img,-1,kernel)
    return output


# parser = argparse.ArgumentParser(description = 'Test')
# parser.add_argument('input_filename', type=str)
# parser.add_argument('output_filename', type=str)
# args = parser.parse_args()

save_dir = './save_model/'

model = Network()
model.cuda()
model = nn.DataParallel(model)

model.eval()
if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    print("load finish")
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)
count = 0
# 遍历图片文件夹
for filePath in Path("/home/data1/qxh/dataset/dataset_urbanpipe/data_image").iterdir():
    #文件夹创建
    if filePath.is_dir():
        dir = "/home/data1/qxh/dataset/dataset_urbanpipe/data_image_denoising/" + filePath.stem + ".mp4"
        if not Path(dir).exists():
            os.makedirs(dir)
        #遍历九图
        for i in range(0, 9):
            #访问图片
            imagePath = os.path.join(str(filePath), str(i) + ".jpg")
            img = Image.open(imagePath)
            #归一化
            transf = transforms.ToTensor()
            img = transf(img).unsqueeze(0).cuda()
            #神经网络处理
            with torch.no_grad():
                _, output = model(img)
            output=output.squeeze(0)
            #图片存储
            img = transforms.ToPILImage()(output)
            img.save(dir+"/"+str(i)+".jpg")
        count+=1

# filePath = "/home/data1/qxh/dataset/dataset_urbanpipe/data_image/d19081.mp4/2.jpg"
# img = Image.open(filePath)
# aug = iaa.AdditiveGaussianNoise(scale=0.2*255)
# aug(img)
# transf = transforms.ToTensor()
# img = transf(img).unsqueeze(0).cuda()
# tmp = transforms.ToPILImage()(img)
# tmp.save("/home/data1/qxh/code/super-image/CBDNet-pytorch-master/old.jpg")
# with torch.no_grad():
#     _, output = model(img)
# output = output.squeeze(0)
# img = transforms.ToPILImage()(output)
# img.save("/home/data1/qxh/code/super-image/CBDNet-pytorch-master/new.jpg")

print(count+" done")


# for datamap in datasmap:
#
#
#     '''视频文件路径'''
#     basefilepath=datamap[0]
#     print(basefilepath)
#
#     count=0


    # with open(txtpath, "r") as f:
    #     for line in f.readlines():
    #         count=count+1
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #
    #         ##子文件名
    #         filepath=line[0:line.rfind('.mp4',1)+4]
    #
    #
    #         ##全路径
    #         filepath=basefilepath+filepath
    #
    #         ###替换下文件名，用于创建新目录
    #         new_filepath=filepath.replace(datamap[2],datamap[3])
    #
    #         dir=new_filepath[0:new_filepath.rfind('/',1)+1]
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)
    #
    #         # cap = cv2.VideoCapture(filepath)
    #         # fps = cap.get(cv2.CAP_PROP_FPS)
    #
    #
    #         # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #         # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #
    #         size = (int(width), int(height))
    #
    #
    #         #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         video = cv2.VideoWriter(new_filepath, fourcc , fps , size ,True )  # size可能会与result尺寸不匹配
    #
    #         for i in range(0,9):
    #             success,frame = cap.read()
    #
    #             if success:
    #                 # i += 1
    #                 # print('i = ',i)
    #                 # if(i>=2711 and i <= 8887):
    #                 #     frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_CUBIC)
    #                 #     frame = frame[0:size[1], 0:size[0]] # frame[Height, Width] 宽高不一致，所以需要交换位置
    #
    #                 #图片resize
    #                 # '''
    #                 # attation!
    #                 # '''
    #                 # frame=cv2.resize(frame,size)
    #
    #
    #                 ##光照增强
    #                 # input_image = adjust_gamma(frame, gamma=3)
    #                 #cv2.imwrite('./gic_img.png',input_image)
    #                 # cv2.imwrite('/home/data1/wzh/code/CBDNet-pytorch-master/testdata/light.png',input_image)
    #
    #                 # # 锐化
    #                 # input_image = img_shapen(input_image)
    #                 # cv2.imwrite('/home/data1/wzh/code/CBDNet-pytorch-master/testdata/shapen_light.png',input_image)
    #
    #
    #                 #归一化
    #                 input_image = read_img(input_image)
    #
    #
    #                 #去噪 (1,c,h,w)
    #                 input_var = torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
    #
    #                 with torch.no_grad():
    #                     _, output = model(input_var)
    #
    #                 output_image = chw_to_hwc(output[0, ...].cpu().numpy())
    #                 output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[:, :, ::-1]
    #
    #                 #cv2.imwrite('/home/data1/wzh/code/CBDNet-pytorch-master/testdata/denosing_.png',output_image)
    #
    #                 video.write(output_image)
    #
    #                 #cv2.imwrite('./denoise_img.png',input_image)
    #             else:
    #                  print('end---' + str(count))
    #                  break

