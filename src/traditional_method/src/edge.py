# -*- coding: utf-8 -*-
"""
Author: 梁超 1466858359@qq.com
Date: 2022-11-06 21:32:26
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:08:18
FilePath: \Machined:\CV\test\src\edge.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class edge_Segmentation:
    def __init__(self, path):
        self.path = path + '/test_mini.jpg'
        # self.path = path + "/lindau_000058_000019_leftImg8bit_resize.png"

    def edge_segment(self):

        # 读取图像
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)))
        b, g, r = cv.split(img)
        rgbImg = cv.merge([r, g, b])

        # 灰度化处理图像
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 二值化
        ret, binary = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Sobel算子
        x = cv.Sobel(grayImage, cv.CV_16S, 1, 0)  # 对x求一阶导
        y = cv.Sobel(grayImage, cv.CV_16S, 0, 1)  # 对y求一阶导
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # Roberts算子
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        # 转uint8
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 拉普拉斯算子
        dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)
        Laplacian = cv.convertScaleAbs(dst)

        # Canny算子(高斯滤波降噪)
        gaussian = cv.GaussianBlur(grayImage, (5, 5), 0)
        Canny = cv.Canny(gaussian, 50, 150)

        # Prewitt算子
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        # 转uint8
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 高斯拉普拉斯算子
        gaussian = cv.GaussianBlur(grayImage, (3, 3), 0)
        dst = cv.Laplacian(gaussian, cv.CV_16S, ksize=3)
        LOG = cv.convertScaleAbs(dst)

        # Scharr算子
        x = cv.Scharr(binary, cv.CV_32F, 1, 0)
        y = cv.Scharr(binary, cv.CV_32F, 0, 1)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Scharr = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 用来正常显示中文标签
        # plt.rcParams["font.sans-serif"] = ["SimHei"]

        # 显示图形
        images = [
            rgbImg,
            binary,
            Sobel,
            Roberts,
            Laplacian,
            Canny,
            Prewitt,
            LOG,
            Scharr,
        ]
        titles = [
            "origin",
            "binary",
            "Sobel",
            "Roberts",
            "Laplacian",
            "Canny",
            "Prewitt",
            "LOG",
            "Scharr",
        ]
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.title(titles[i])
            if i == 0:
                plt.imshow(images[i])
            else:
                plt.imshow(images[i], plt.cm.gray)
            plt.xticks([]), plt.yticks([])
        plt.suptitle("fixed Edge")
        plt.show()

    def contour_segment(self):
        # 读取图像
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)))

        # 灰度化处理图像
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 阈值化处理
        ret, binary = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # 边缘检测
        contours, hierarchy = cv.findContours(
            binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        # 轮廓绘制
        cv.drawContours(img, contours, -1, (0, 255, 0), 1)

        # 显示图像
        cv.imshow("gray", binary)
        cv.imshow("res", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
