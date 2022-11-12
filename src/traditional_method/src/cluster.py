# -*- coding: utf-8 -*-
'''
Author: 梁超 1466858359@qq.com
Date: 2022-11-08 18:30:19
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:06:48
FilePath: \Machined:\CV\test\src\cluster.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class cluster_Segmentation:

    def __init__(self, path):
        self.path = path


    def kmeans_segment(self, n = 6):

        # 读取原始图像
        path = self.path + '/test_mini.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 图像二维像素转换为一维
        data = img.reshape((-1,3))
        data = np.float32(data)

        # 定义中心 (type,max_iter,epsilon)
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # 设置标签
        flags = cv.KMEANS_RANDOM_CENTERS

        # K-Means聚类 聚集成n类
        compactness, labels, centers = cv.kmeans(data, n, None, criteria, 10, flags)

        # 图像转换回uint8二维类型
        centers = np.uint8(centers)

        # 生成最终图像
        res = centers[labels.flatten()]
        dst = res.reshape((img.shape))
        # dst = labels.reshape((img.shape[0], img.shape[1]))

        # 图像转换为RGB显示
        b, g, r = cv.split(img)
        rgbImg = cv.merge([r, g, b])
        b, g, r = cv.split(dst)
        rgbDst = cv.merge([r, g, b])

        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif']=['SimHei']

        # 显示图像
        titles = [u'原始图像', u'聚类图像']  
        images = [rgbImg, rgbDst]  

        for i in range(2):  
            plt.subplot(1,2,i + 1)
            plt.imshow(images[i], 'gray'), 
            plt.title(titles[i])  
            plt.xticks([]), plt.yticks([])  
        plt.show()


    def meanshift_segment(self, n = 20):

        # 读取原始图像
        path = self.path + '/test_mini.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 决定性影响因素，漂移空间半径
        spatialRad = n   #空间窗口大小
        colorRad = n     #色彩窗口大小
        maxPyrLevel = 2    #金字塔层数

        #图像均值漂移分割
        dst = cv.pyrMeanShiftFiltering(img, spatialRad, colorRad, maxPyrLevel)

        #显示图像
        cv.imshow('src', img)
        cv.imshow('dst', dst)
        cv.waitKey()
        cv.destroyAllWindows()


    def SLIC_Superpixel(self):

        # 读取原始图像
        path = self.path + '/superpixel.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
        slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler = 20.0) 

        # 迭代次数，越大效果越好
        slic.iterate(10)     

        # 获取Mask，超像素边缘Mask==1
        mask_slic = slic.getLabelContourMask() 

        # 获取超像素标签
        label_slic = slic.getLabels()        

        # 获取超像素数目
        number_slic = slic.getNumberOfSuperpixels()  
        mask_inv_slic = cv.bitwise_not(mask_slic)  

        # 在原图上绘制超像素边界
        img_slic = cv.bitwise_and(img,img,mask =  mask_inv_slic) 

        cv.imshow("img_slic", img_slic)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def SEEEDS_Superpixel(self):

        # 读取原始图像
        path = self.path + '/superpixel.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # 初始化seeds项，注意图片长宽的顺序
        seeds = cv.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 2000, 15, 3, 5, True)

        # 输入图像大小必须与初始化形状相同，迭代次数为10
        seeds.iterate(img, 10)  

        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()
        mask_inv_seeds = cv.bitwise_not(mask_seeds)
        img_seeds = cv.bitwise_and(img, img, mask =  mask_inv_seeds)

        cv.imshow("img_seeds", img_seeds)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def LSC_Superpixel(self):

        # 读取原始图像
        path = self.path + '/superpixel.jpg'
        img = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        lsc = cv.ximgproc.createSuperpixelLSC(img)

        # 迭代次数，越大效果越好
        lsc.iterate(10)

        mask_lsc = lsc.getLabelContourMask()
        label_lsc = lsc.getLabels()
        number_lsc = lsc.getNumberOfSuperpixels()
        mask_inv_lsc = cv.bitwise_not(mask_lsc)
        img_lsc = cv.bitwise_and(img, img, mask = mask_inv_lsc)

        cv.imshow("img_lsc", img_lsc)
        cv.waitKey(0)
        cv.destroyAllWindows()


