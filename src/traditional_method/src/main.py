'''
Author: 梁超 1466858359@qq.com
Date: 2022-11-06 14:37:16
LastEditors: 梁超 1466858359@qq.com
LastEditTime: 2022-11-09 20:11:07
FilePath: \vecan_ugvd:\CV\traditional_method\src\main.py
Description: 

Copyright (c) 2022 by 梁超 1466858359@qq.com, All Rights Reserved. 
'''

from threshold import thresh_Segmentation
from edge import edge_Segmentation
from graphcut import graph_Segmentation
from grabcut import grab_Segmentation
from cluster import cluster_Segmentation
from watershed import watershed_Segmentation
from snake import snake_Segmentation


if __name__ == "__main__":
    path = "../../assets"
    thresh_Seg = thresh_Segmentation(path)
    # thresh_Seg.imgThresh()
    # thresh_Seg.myOtsu()
    edge_Seg = edge_Segmentation(path)
    # edge_Seg.edge_segment()
    # edge_Seg.contour_segment()
    graph_Seg = graph_Segmentation(path)
    # graph_Seg.graph_cut()
    grab_Seg = grab_Segmentation(path)
    # grab_Seg.grab_cut()
    cluster_Seg = cluster_Segmentation(path)
    # cluster_Seg.kmeans_segment()
    # cluster_Seg.meanshift_segment()
    # cluster_Seg.SLIC_Superpixel()
    # cluster_Seg.SEEEDS_Superpixel()
    # cluster_Seg.LSC_Superpixel()
    watershed_Seg = watershed_Segmentation(path)
    # watershed_Seg.watershed_segment()
    snake_Seg = snake_Segmentation(path)
    # snake_Seg.snake_segment()
