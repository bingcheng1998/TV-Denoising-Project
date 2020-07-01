#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os

def saveImg2bmp(img_name):
	im = Image.open('./jpg/'+img_name)
	L = im.convert('L')
	im_array = np.array(L)
	L.save('./img/'+img_name[:-4]+'.bmp')

# img_name = 'barbara.jpg'

# saveImg2bmp(img_name)
filePath = './jpg'
for file_name in os.listdir(filePath):
	saveImg2bmp(file_name)