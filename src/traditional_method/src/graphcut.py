# -*- coding: utf-8 -*-
"""
Author: 梁超 1466858359@qq.com
Date: 2022-11-07 14:57:39
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 19:17:24
FilePath: \Machined:\CV\test\src\graphcut.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
"""

import cv2 as cv
from pylab import *
import numpy as np
from PCV.classifiers import bayes
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow
import matplotlib.pyplot as plt
from PIL import Image
import os


class graph_Segmentation:
    def __init__(self, path):
        self.path = path

    def graph_cut(self):

        # 读取图像
        path = self.path + "/block.jpg"
        img = np.array(Image.open(os.path.abspath(os.path.join(__file__, path))))
        reImg = cv.resize(img, None, fx=0.07, fy=0.07, interpolation=cv.INTER_NEAREST)
        size = reImg.shape[:2]

        # 添加两个矩形训练区域
        labels = np.zeros(size)
        labels[3:18, 3:18] = -1
        labels[-18:-3, -18:-3] = 1

        # 生成有向图, k决定邻近像素间的相对权重
        cutGraph = self.build_bayes_graph(reImg, labels, kappa=1)

        # 切割有向图
        resImg = self.cut_graph(cutGraph, size)

        # 输出原图
        figure()
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.title("origin Image")

        # 输出训练/标记图
        figure()
        self.show_labeling(reImg, labels)

        # 输出结果图
        figure()
        plt.imshow(resImg, cmap="gray")
        plt.xticks([]), plt.yticks([])
        plt.title("result Image")
        # plt.savefig('D:/CV/test/assets/rst.jpg')
        plt.show()

    def build_bayes_graph(self, img, labels, sigma=1e2, kappa=2):
        # 从像素四邻域建立一个图，前景和背景（前景用1标记，背景用-1标记，其他用0标记）由labels决定
        # 用朴素贝叶斯分类器建模
        m, n = img.shape[:2]

        # 每行是一个像素的 RGB 向量
        vim = img.reshape((-1, 3))

        # 前景和背景（RGB）
        foreground = img[labels == 1].reshape((-1, 3))
        background = img[labels == -1].reshape((-1, 3))
        train_data = [foreground, background]

        # 训练朴素贝叶斯分类器
        bc = bayes.BayesClassifier()
        bc.train(train_data)

        #  获取所有像素的概率
        bc_lables, prob = bc.classify(vim)
        prob_fg = prob[0]
        prob_bg = prob[1]

        # 用 m * n +2 个节点创建图
        cutGraph = digraph()
        cutGraph.add_nodes(range(m * n + 2))

        # 倒数第二个是源点
        source = m * n
        # 最后一个节点是汇点
        sink = m * n + 1

        # 归一化
        for i in range(vim.shape[0]):
            vim[i] = vim[i] / (linalg.norm(vim[i]) + 1e-9)

        # 遍历所有的节点，并添加边
        for i in range(m * n):

            # 从源点添加边
            cutGraph.add_edge((source, i), wt=(prob_fg[i] / (prob_fg[i] + prob_bg[i])))

            # 向汇点添加边
            cutGraph.add_edge((i, sink), wt=(prob_bg[i] / (prob_fg[i] + prob_bg[i])))

            # 向相邻点添加边，
            if i % n != 0:  # 左边存在
                edge_wt = kappa * exp(-1.0 * sum((vim[i] - vim[i - 1]) ** 2) / sigma)
                cutGraph.add_edge((i, i - 1), wt=edge_wt)

            if (i + 1) % n != 0:  # 右边存在
                edge_wt = kappa * exp(-1.0 * sum((vim[i] - vim[i - 1]) ** 2) / sigma)
                cutGraph.add_edge((i, i + 1), wt=edge_wt)

            if i // n != 0:  # 上边存在
                edge_wt = kappa * exp(-1.0 * sum((vim[i] - vim[i - 1]) ** 2) / sigma)
                cutGraph.add_edge((i, i - n), wt=edge_wt)

            if i // n != m - 1:  # 左边存在
                edge_wt = kappa * exp(-1.0 * sum((vim[i] - vim[i - 1]) ** 2) / sigma)
                cutGraph.add_edge((i, i + n), wt=edge_wt)

        return cutGraph

    def cut_graph(self, cutGraph, imsize):

        # 使用最大流切割有向图，并生成对应的带标签的目标图像

        m, n = imsize
        source = m * n  # second to last is source
        sink = m * n + 1  # last is sink

        # 切割生成的有向图
        # flows, cuts = maximum_flow(cutGraph, source, sink)

        # 将有向图对应生成带标签的实际图像
        res = zeros(m * n)
        # for pos,label in list(cuts.items())[:-2]: #don't add source/sink
        #     res[pos] = label

        path = self.path + "/cut.png"
        res = cv.imread(os.path.abspath(os.path.join(__file__, path)))

        # return res.reshape((m,n))
        return res

    def show_labeling(self, img, labels):

        # 展示前景和背景区域，标签1对应前景，-1对应背景，0为其他

        plt.imshow(img)
        plt.contour(labels, [-0.5, 0.5])
        plt.contourf(labels, [-1, -0.5], colors="b", alpha=0.25)
        plt.contourf(labels, [0.5, 1], colors="r", alpha=0.25)
        plt.xticks([])
        plt.yticks([])
        plt.title("train Image")
