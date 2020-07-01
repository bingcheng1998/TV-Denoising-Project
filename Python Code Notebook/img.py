#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import os


# plt.rcParams['savefig.dpi'] = 82 #图片像素
 #分辨率

img_list = ['shape3.bmp','cameraman.bmp','barbara.bmp','barbarasmall.bmp', 'elaine.bmp','horse.bmp','satellite.bmp','shape1.bmp']

def original(order = 0):
    im = Image.open('./img/'+img_list[order])
    im_array = np.array(im)
    im_array = im_array.astype(np.float64)
    return im_array/255

def original255(order = 0):
    im = Image.open('./img/'+img_list[order])
    im_array = np.array(im)
    im_array = im_array.astype(np.float64)
    return im_array

def gaussNoised(order = 0, sigma = 25):
    # 返回高斯模糊后的数组
    def gaussNoise(im_array, sigma):
        im_array_flat = im_array.flatten()
        for i in range(im_array.shape[0]*im_array.shape[1]):
            pointInFlat = int(im_array_flat[i])+ random.gauss(0,sigma)
            if pointInFlat < 0:
                pointInFlat = 0
            if pointInFlat > 255:
                pointInFlat = 255
            im_array_flat[i] = pointInFlat
        im_array = im_array_flat.reshape([im_array.shape[0],im_array.shape[1]])
        return im_array
    
    im_array = original255(order)
#     sigma = sigma #设定高斯函数的标准差
    im_array_noised = gaussNoise(im_array, sigma)
    return im_array_noised/255



def saltNoised(order = 0, salt_percent = 0.05, salt_color = 255):
    def saltNoise(im_array, salt_percent = salt_percent, salt_color = salt_color):
        # salt_percent 为噪点的最大占比，真实个数少于这个占比
        salt_num = im_array.shape[0]*im_array.shape[1]*salt_percent
        for k in range(int(salt_num)):
            i = random.randint(0,im_array.shape[0]-1)
            j = random.randint(0,im_array.shape[1]-1)
            im_array[i][j] = salt_color
        return im_array
    
    im_array = original255(order)
    im_array_noised = saltNoise(im_array, salt_percent, salt_color)
    return im_array_noised/255

# def showImg(im_array):
#     im_array = im_array - min(min(im_array))
#     im_array = im_array/max(max(im_array))*255
#     im_array = im_array.astype(np.uint8)
#     im_show = Image.fromarray(im_array)
#     return im_show
def showImg(im_array):
#     print(im_array.min())
#     im_array = im_array - im_array.min()
    for i in range(im_array.shape[0]):
        for j in range(im_array.shape[1]):
            if im_array[i][j] < 0:
                im_array[i][j] = 0
            if im_array[i][j] > 1:
                im_array[i][j] = 1
    im_array = im_array*255
#     print('im_array.shape',im_array.shape)
    im_array = im_array.astype(np.uint8)
    im_show = Image.fromarray(im_array)
    return im_show

def inlineImg(x_local, title = None, dpi = 84):
    plt.imshow(showImg(x_local),cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title) # 图像题目
    plt.rcParams['figure.dpi'] = dpi
    plt.show()

def checkPath(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
def sava(image, img_name, img_folder = 'default'):
    # save np.array as a bmp file
    path = './gen-img/'+img_folder+'/'
    checkPath(path)
    L = showImg(image).convert('L')
    L.save(path+img_name+'.bmp')