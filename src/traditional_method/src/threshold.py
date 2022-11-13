# -*- coding: utf-8 -*-
"""
Author: 梁超 1466858359@qq.com
Date: 2022-11-06 21:26:59
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:15:30
FilePath: \Machined:\CV\test\src\threshold.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os


class thresh_Segmentation:
    def __init__(self, path):
        self.path = path + "/dog.png"
        # self.path = path + "/lindau_000058_000019_leftImg8bit_resize.png"

    # imshow
    def show(self, thresh):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        ret, threshold = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
        images = [img, threshold]
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
        plt.show()

    # opencv自带threshold函数不同type对比
    def imgThresh(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)))
        # grayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # 阈值设定
        thresh = 145
        # 阈值处理
        # thresholdType：Int类型的，方法如下：
        # 	THRESH_BINARY 二进制阈值化 -> 大于阈值为1 小于阈值为0
        # 	THRESH_BINARY_INV 反二进制阈值化 -> 大于阈值为0 小于阈值为1
        # 	THRESH_TRUNC 截断阈值化 -> 大于阈值为阈值，小于阈值不变
        # 	THRESH_TOZERO 阈值化为0 -> 大于阈值的不变，小于阈值的全为0
        # 	THRESH_TOZERO_INV 反阈值化为0 -> 大于阈值为0，小于阈值不变
        ret, thresh1 = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(img, thresh, 255, cv.THRESH_BINARY_INV)
        ret, thresh3 = cv.threshold(img, thresh, 255, cv.THRESH_TRUNC)
        ret, thresh4 = cv.threshold(img, thresh, 255, cv.THRESH_TOZERO)
        ret, thresh5 = cv.threshold(img, thresh, 255, cv.THRESH_TOZERO_INV)
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

        # 显示结果
        titles = ["Gray Image", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.title(titles[i])
            plt.imshow(images[i])
            plt.xticks([]), plt.yticks([])
        plt.suptitle("fixed threshold")
        plt.show()

    # opencv自带adaptiveThreshold函数
    def imgAdaptThresh(self):

        # 使用自动阈值时，只能使用单通道图片
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)

        # adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
        # maxValue：Double类型的，阈值的最大值
        # adaptiveMethod：Int类型，选择如下
        # 	ADAPTIVE_THRESH_MEAN_C（通过平均取得平均值）
        # 	ADAPTIVE_THRESH_GAUSSIAN_C(通过高斯取得高斯值)
        # blockSize：Int类型，决定像素的邻域块尺寸。
        # C：偏移值调整量，计算adaptiveMethod用到的参数。
        thresh1 = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10
        )
        thresh2 = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10
        )

        images = [img, thresh1, thresh2]
        self.show(images)

    # 自行实现均值自适应阈值化分割
    def myAdapt_mean(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        win_size = (5, 5)
        ratio = 0.02

        # 对图像矩阵进行均值平滑
        image_mean = cv.blur(img, win_size)

        # 原图像矩阵与平滑结果做差
        threshImg = img - (1.0 - ratio) * image_mean

        # 当差值大于或等于0时，输出值为255，反之输出值为0
        threshImg[threshImg >= 0] = 255
        threshImg[threshImg < 0] = 0
        threshImg = threshImg.astype(np.uint8)

        images = [img, threshImg]
        self.show(images)

    # 自行实现高斯自适应阈值化分割
    def myAdapt_gauss(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        win_size = (5, 5)
        ratio = 0.02

        # 对图像矩阵进行高斯平滑
        image_mean = cv.GaussianBlur(img, win_size, 10)

        # 原图像矩阵与平滑结果做差
        threshImg = img - (1.0 - ratio) * image_mean

        # 当差值大于或等于0时，输出值为255，反之输出值为0
        threshImg[threshImg >= 0] = 255
        threshImg[threshImg < 0] = 0
        threshImg = threshImg.astype(np.uint8)

        images = [img, threshImg]
        self.show(images)

    # opencv自带Otsu阈值法
    def OtsuThresh(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        threshod = 145
        ret, thresh = cv.threshold(
            img, threshod, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        images = [img, thresh]
        self.show(images)

    # 灰度直方图的计算
    def calgrayHist(self, img):
        height, width = img.shape[:2]
        grayHist = np.zeros([256], np.uint64)
        for i in range(height):
            for j in range(width):
                grayHist[img[i][j]] += 1
        return grayHist

    # 自行实现Otsu阈值法(求阈值)
    def myOtsu(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        rows, cols = img.shape[:2]

        # 计算灰度直方图
        grayHist = self.calgrayHist(img)

        # 归一化灰度直方图
        norm_hist = grayHist / float(rows * cols)

        # 计算零阶累积矩, 一阶累积矩
        zero_cumu_moment = np.zeros([256], np.float32)
        one_cumu_moment = np.zeros([256], np.float32)
        for i in range(256):
            if i == 0:
                zero_cumu_moment[i] = norm_hist[i]
                one_cumu_moment[i] = 0
            else:
                zero_cumu_moment[i] = zero_cumu_moment[i - 1] + norm_hist[i]
                one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]

        # 计算方差，找到最大的方差对应的阈值
        mean = one_cumu_moment[255]
        thresh = 0
        sigma = 0
        for i in range(256):
            if zero_cumu_moment[i] == 0 or zero_cumu_moment[i] == 1:
                sigma_tmp = 0
            else:
                sigma_tmp = math.pow(
                    mean * zero_cumu_moment[i] - one_cumu_moment[i], 2
                ) / (zero_cumu_moment[i] * (1.0 - zero_cumu_moment[i]))
            if sigma < sigma_tmp:
                thresh = i
                sigma = sigma_tmp

        # print(thresh)
        # 阈值处理
        self.show(thresh)

    # opencv自带直方图阈值法
    def triangle(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        threshod = 145
        ret, thresh = cv.threshold(
            img, threshod, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE
        )
        images = [img, thresh]
        self.show(images)

    # 自行实现直方图阈值法【双峰】(求阈值)
    def myTriangle(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)

        # 计算灰度直方图
        histogram = self.calgrayHist(img)

        # 找到灰度直方图的最大峰值对应的灰度值
        maxLoc = np.where(histogram == np.max(histogram))
        firstPeak = maxLoc[0][0]

        # 寻找灰度直方图的第二个峰值对应的灰度值
        measureDists = np.zeros([256], np.float32)
        for k in range(256):
            measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
        maxLoc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = maxLoc2[0][0]

        # 找到两个峰值之间的最小值对应的灰度值，作为阈值
        thresh = 0
        if firstPeak > secondPeak:
            temp = histogram[int(secondPeak) : int(firstPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = secondPeak + minLoc[0][0] + 1
        else:
            temp = histogram[int(firstPeak) : int(secondPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = firstPeak + minLoc[0][0] + 1

        # 阈值处理
        self.show(thresh)

    # 自行实现信息熵阈值法(求阈值)
    def myEntropy(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        rows, cols = img.shape

        # 求灰度直方图
        grayHist = self.calgrayHist(img)

        # 归一化灰度直方图，即概率直方图
        normGrayHist = grayHist / float(rows * cols)

        # 1.计算累加直方图
        zeroCumuMoment = np.zeros([256], np.float32)
        for i in range(256):
            if i == 0:
                zeroCumuMoment[i] = normGrayHist[i]
            else:
                zeroCumuMoment[i] = zeroCumuMoment[i - 1] + normGrayHist[i]

        # 2.计算各个灰度级的熵
        entropy = np.zeros([256], np.float32)
        for i in range(256):
            if i == 0:
                if normGrayHist[i] == 0:
                    entropy[i] = 0
                else:
                    entropy[i] = -normGrayHist[i] * math.log10(normGrayHist[i])
            else:
                if normGrayHist[i] == 0:
                    entropy[i] = entropy[i - 1]
                else:
                    entropy[i] = entropy[i - 1] - normGrayHist[i] * math.log10(
                        normGrayHist[i]
                    )

        # 3.找阈值
        fT = np.zeros([256], np.float32)
        ft1, ft2 = 0, 0
        totalEntropy = entropy[255]
        for i in range(255):
            # 找最大值
            maxFront = np.max(normGrayHist[0 : i + 1])
            maxBack = np.max(normGrayHist[i + 1 : 256])
            if (
                maxFront == 0
                or zeroCumuMoment[i] == 0
                or maxFront == 1
                or zeroCumuMoment[i] == 1
                or totalEntropy == 0
            ):
                ft1 = 0
            else:
                ft1 = (
                    entropy[i]
                    / totalEntropy
                    * (math.log10(zeroCumuMoment[i]) / math.log10(maxFront))
                )
            if (
                maxBack == 0
                or 1 - zeroCumuMoment[i] == 0
                or maxBack == 1
                or 1 - zeroCumuMoment[i] == 1
            ):
                ft2 = 0
            else:
                if totalEntropy == 0:
                    ft2 = math.log10(1 - zeroCumuMoment[i]) / math.log10(maxBack)
                else:
                    ft2 = (1 - entropy[i] / totalEntropy) * (
                        math.log10(1 - zeroCumuMoment[i]) / math.log10(maxBack)
                    )
            fT[i] = ft1 + ft2

        # 找最大值的索引，作为得到的阈值
        threshLoc = np.where(fT == np.max(fT))
        thresh = threshLoc[0][0]

        # 阈值处理
        self.show(thresh)

    # 自行实现迭代阈值法(求阈值)
    def myIteraThre(self):
        img = cv.imread(os.path.abspath(os.path.join(__file__, self.path)), 0)
        rows, cols = img.shape
        hisData = self.calgrayHist(img)

        T0 = 0
        for i in range(256):
            T0 += i * hisData[i]

        T0 /= cols * rows
        T0 = int(T0)

        T1, T2, num1, num2, T = 0, 0, 0, 0, 0

        while 1:
            for i in range(int(T0 + 1)):
                T1 += i * hisData[i]
                num1 += hisData[i]
            if num1 == 0:
                continue

            index = int(256 - (T0 + 1))
            for i in range(index):
                j = (T0 + 1) + i
                T2 += j * hisData[j]
                num2 += hisData[j]
            if num2 == 0:
                continue

            T = int((T1 / num1 + T2 / num2) / 2)

            if T == T0:
                break
            else:
                T0 = T

        self.show(T)
