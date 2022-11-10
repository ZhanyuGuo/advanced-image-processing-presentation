'''
Author: 梁超 1466858359@qq.com
Date: 2022-11-08 19:07:26
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:04:21
FilePath: \Machined:\CV\test\src\watershed.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class watershed_Segmentation:
    
    def __init__(self, path):
        self.path = path


    def watershed_segment(self):
        
        # 读取原始图像
        path = self.path + '/coin.png'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 图像灰度化处理
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 图像阈值化处理
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # 图像开运算消除噪声
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

        # 图像膨胀操作确定背景区域
        sure_bg = cv.dilate(opening, kernel, iterations = 3)
 
        # 距离运算确定前景区域
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # 寻找未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # 标记变量
        ret, markers = cv.connectedComponents(sure_fg)

        # 所有标签加一，以确保背景不是0而是1
        markers = markers + 1

        # 用0标记未知区域
        markers[unknown == 255] = 0

        #分水岭算法实现图像分割
        markers = cv.watershed(img, markers)
        img[markers == -1] = [255,0,0]

        #用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']

        #显示图像
        titles = [u'标记区域', u'图像分割']  
        images = [markers, img]  
        for i in range(2):  
            plt.subplot(1, 2, i + 1)
            plt.imshow(images[i], 'gray')  
            plt.title(titles[i])  
            plt.xticks([]), plt.yticks([])  
        plt.show()