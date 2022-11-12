# -*- coding: utf-8 -*-
'''
Author: 梁超 1466858359@qq.com
Date: 2022-11-07 17:21:17
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:11:12
FilePath: \Machined:\CV\test\src\grabcut.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


'''
===============================================================================
交互行为：
在程序运行之后会出现input和output两个界面;
在input界面中首先会出现输入的图像, output首先为全黑图像;
首先在受到提示之后可以使用鼠标右键画一个矩形区域, 框柱想要分割出的目标;
按n后3秒内继续按n可以记录并更新这个区域,
然后通过选择按下0/1/2/3, 分别用黑色、白色、绿色、红色的画笔画到你认为的绝对背景、绝对前景、可能背景、可能前景,
最后同样通过n进行保存更新, 通过output界面会实时输出结果;
r和s分别对应重置和保存, 保存的图像会有两张, 分别对应结果图像和过程图像;
===============================================================================
'''


class grab_Segmentation:

    def __init__(self, path):
        self.path = path


    def onmouse(self, event, x, y, flags, param):

        # 绘制矩形，对应分割大区域
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.rsvImg.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x,y), self.BLUE, 2)
                self.rect = (self.ix, self.iy, abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x,y), self.BLUE, 2)
            self.rect = (self.ix, self.iy, abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # 绘制曲线，对应自行标注的前景背景
        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
    

    def grab_cut(self):

        # 颜色对应
        self.BLUE = [255, 0, 0]   
        RED = [0, 0, 255]         
        GREEN = [0, 255, 0]       
        BLACK = [0, 0, 0]         
        WHITE = [255, 255, 255]   

        # 将数字与颜色对应，方便以后的显示
        DRAW_BG = {'color' : BLACK, 'val' : 0}      # sure background
        DRAW_FG = {'color' : WHITE, 'val' : 1}      # sure foreground
        DRAW_PR_FG = {'color' : GREEN, 'val' : 3}   # probable background
        DRAW_PR_BG = {'color' : RED, 'val' : 2}     # probable foreground

        # 生成需要的指示变量
        self.rect = (0, 0, 1, 1)
        self.drawing = False         # 绘制曲线的指示
        self.rectangle = False       # 绘制矩形的指示
        self.rect_over = False       # 检查矩形是否绘制完毕
        self.rect_or_mask = 100      # 检查是否选择矩形/蒙版
        self.value = DRAW_FG         # 初始化绘画前景
        self.thickness = 2           # 画笔粗细

        # 输出文本
        print(__doc__)

        # 读取初始图像
        path = self.path + '/grab.png'
        self.img = cv.imread(os.path.abspath(os.path.join(__file__, path)))
        # 保存初始图像(备份)
        self.rsvImg = self.img.copy()   
        # mask初始化为可能的背景(先全置0)                            
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) 
        # 定义输出图像(起始为全黑)
        output = np.zeros(self.img.shape, np.uint8)           

        # 打开窗口
        cv.namedWindow('output')
        cv.namedWindow('input')
        # 捕捉鼠标操作，对应callback函数
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1] + 10, 90)

        # 基础提示
        print(" Instructions : \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        # 主循环：负责实时操作与更新
        while(1):

            # 展示图像
            cv.imshow('output', output)
            cv.imshow('input', self.img)
            k = 0xFF & cv.waitKey(1)

            # 键位绑定
            if k == 27:                 # esc键退出
                break
            elif k == ord('0'):         # 0对应背景绘制
                print(" mark background regions with left mouse button \n")
                self.value = DRAW_BG
            elif k == ord('1'):         # 1对应前景绘制
                print(" mark foreground regions with left mouse button \n")
                self.value = DRAW_FG
            elif k == ord('2'):         # 2对应可能背景绘制
                self.value = DRAW_PR_BG
            elif k == ord('3'):         # 3对应可能前景绘制
                self.value = DRAW_PR_FG
            elif k == ord('s'):         # s对应保存图像
                # bar为输出结果图像
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                # res为总流程图像，存在拼接
                res = np.hstack((self.rsvImg, bar, self.img, bar, output))
                # 保存图像到指定位置
                path = self.path + '/grabcut_output.png'
                cv.imwrite(os.path.abspath(os.path.join(__file__, path)), output)
                path = self.path + '/grabcut_output_combined.png'
                cv.imwrite(os.path.abspath(os.path.join(__file__, path)), res)
                print(" Result saved as image \n")
            elif k == ord('r'):         # r对应重置全部操作
                print("resetting \n")
                self.rect = (0, 0, 1, 1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = DRAW_FG
                self.img = self.rsvImg.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) 
                output = np.zeros(self.img.shape, np. uint8)       
            elif k == ord('n'):         # n对应更新前景背景，执行切割操作，ouput实时更新
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                if (self.rect_or_mask == 0):         # 使用矩形进行grabcut
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv.grabCut(self.rsvImg, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                    self.rect_or_mask = 1
                elif self.rect_or_mask == 1:         # 使用自行标注的mask进行grabcut
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    cv.grabCut(self.rsvImg, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

            midMask = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            output = cv.bitwise_and(self.rsvImg, self.rsvImg, mask = midMask)

        cv.destroyAllWindows()
