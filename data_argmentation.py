import cv2
import random
import numpy as np
from skimage import exposure


# 随机size，进行高斯平滑
def GaussioanBlurSize(Size, img):
    KSIZE = Size * 2 + 3
    n = random.randint(0, 1)
    if n == 0:
        sigma = 2.2
    elif n == 1:
        sigma = 1.5
    else:
        sigma = 3
    dst = cv2.GaussianBlur(img, (KSIZE, KSIZE), sigma, KSIZE)
    print("高斯平滑size：", KSIZE, " sigma:",sigma)
    return dst


# 随机椒盐噪声
def saltpepper(img, n):
    m = int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    print("椒盐噪声比率：", n)
    return img


def flip(img, img_label, n):
    # n=1,横向翻转图像;n=0,纵向翻转图像;n=-1,同时在横向和纵向翻转图像,n=2,不翻转
    if n == 2:
        pass
    else:
        flipped = cv2.flip(img, n)
        flipped_label = cv2.flip(img_label, n)
        print("翻转：", n)
    return flipped, flipped_label


# 直方图均衡化，如果n=1，进行；n=0，不进行，n=2，线性拉伸。把原始图像的灰度直方图从比较集中的某个灰度区间变成在全部灰度范围内的均匀分布。
def Balance(img, n):
    if n == 1:
        split = cv2.split(img)
        for i in range(3):
            cv2.equalizeHist(split[i], split[i])
        img = cv2.merge(split)
        print("执行均衡化")
    if n == 0:
        print("未执行均衡化")
        # pass
    if n == 2:
        split = cv2.split(img)
        for i in range(3):
            split[i] = exposure.rescale_intensity(split[i])
        img = cv2.merge(split)
        print("执行线性拉伸")
    return img


def data_argmentation(img,img_label):
    # 翻转
    img, img_label = flip(img, img_label, random.randint(-1, 2))
    # 直方图均衡化或线性拉伸
    img = Balance(img, random.randint(0, 2))
    # 高斯平滑
    y_n = random.randint(0, 1)
    if y_n == 1:
        img = GaussioanBlurSize(random.randint(0, 1), img)
    # 椒盐噪声
    y_n = random.randint(0, 1)
    if y_n == 1:
        img = saltpepper(img, random.uniform(0, 0.01))
    return img, img_label
