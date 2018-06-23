import os
from keras.preprocessing.image import img_to_array
import glob, random, cv2
import numpy as np
from data_argmentation import *
from PIL import Image


# size为缩小后每个小块的大小
def img_pingpu_2(img_ori, size):
    img_sub2 = cv2.resize(img_ori, (size, size), interpolation=cv2.INTER_AREA)
    img_pin_sub2 = np.zeros((size*2, size*2, 3), img_ori.dtype)

    img_pin_sub2[0:size, 0:size] = img_sub2
    img_pin_sub2[0:size, size:size * 2] = img_sub2
    img_pin_sub2[size:size * 2, 0:size] = img_sub2
    img_pin_sub2[size:size * 2, size:size * 2] = img_sub2
    return img_pin_sub2


def label_pingpu_2(label_ori, size):
    label_sub2 = cv2.resize(label_ori, (size, size), interpolation=cv2.INTER_AREA)
    label_pin_sub2 = np.zeros((size*2, size*2), label_ori.dtype)

    label_pin_sub2[0:size, 0:size] = label_sub2
    label_pin_sub2[0:size, size:size*2] = label_sub2
    label_pin_sub2[size:size*2, 0:size] = label_sub2
    label_pin_sub2[size:size*2, size:size*2] = label_sub2
    return label_pin_sub2


def label_binary(labelimg):
    labelimg /= 255
    labelimg[labelimg > 0.5] = 1
    labelimg[labelimg <= 0.5] = 0
    return labelimg


# 多尺度输入，背景平铺
def generatedata(path, batchsize):
    imgs = glob.glob(path + "image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img_ori = cv2.imread(path + "image\\" + midname)
            label_ori = cv2.imread(path + "label\\" + midname, cv2.IMREAD_GRAYSCALE)
            # img_ori, label_ori = data_argmentation(img_ori, label_ori) # 辐射正确函数可以这样调用
            img = img_to_array(img_ori).astype('float32')
            label = img_to_array(label_ori).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            # 降采样1倍
            img_pin_sub2 = img_pingpu_2(img_ori, 256)
            label_pin_sub2 = label_pingpu_2(label_ori, 256)

            img = img_to_array(img_pin_sub2).astype('float32')
            label = img_to_array(label_pin_sub2).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)

            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), np.array(imglabels))
                cnt = 0
                imgdatas = []
                imglabels = []





