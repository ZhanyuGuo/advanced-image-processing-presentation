# -*- coding: utf-8 -*-
'''
Author: 梁超 1466858359@qq.com
Date: 2022-11-08 19:36:33
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:22:53
FilePath: \Machined:\CV\test\src\snake.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import os


class snake_Segmentation:
    
    def __init__(self, path):
        self.path = path


    def snake_segment(self):
        
        # 读取原始图像
        path = self.path + '/snake.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 灰度化处理
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 圆的参数方程：(130, 80) r = 58
        t = np.linspace(0, 2*np.pi, 400) 
        x = 130 + 58*np.cos(t)
        y = 80 + 58*np.sin(t)

        # 构造初始Snake
        init = np.array([x, y]).T 

        # Snake模型迭代输出
        snake = active_contour(gaussian(img, 3), snake = init, alpha = 0.1, beta = 1, gamma = 0.01, w_line = 0, w_edge = 10)
        # snake:  初始蛇坐标。对于周期性边界条件，端点不得重复。
        # alpha:  蛇长度形状参数。数值越高，蛇的收缩速度越快。
        # beta:   蛇平滑形状参数。值越高，蛇越平滑。
        # gamma:  显式时间步长参数
        # w_line: 控制对亮度的吸引力。使用负值吸引黑暗区域。
        # w-edge: 控制对边的吸引力。使用负值从边缘击退蛇


        # 绘图显示
        plt.figure(figsize = (5, 5))
        plt.imshow(img, cmap = "gray")
        plt.plot(init[:, 0], init[:, 1], '--r', lw = 3)
        plt.plot(snake[:, 0], snake[:, 1], '-b', lw = 3)
        plt.xticks([]), plt.yticks([])

        plt.figure(figsize = (5, 5))
        path = self.path + '/snakerst.png'
        standardImg = cv.imread(os.path.abspath(os.path.join(__file__, path)))
        plt.imshow(standardImg)
        plt.xticks([]), plt.yticks([])

        plt.show()