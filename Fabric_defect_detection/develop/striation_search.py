#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 12.25.2020
Updated on 12.25.2020

Author: haoshaui@handaotech.com
'''

import os
import sys
import cv2
import numpy as np
import glob as gb
from PIL import Image
from matplotlib import pyplot as plt
from PascalVocParser import PascalVocXmlParser


pvoc = PascalVocXmlParser()


def striation_height_histogram(ann_dir):
    """Plot the histogram for the striation height positions
    
    Args:
        ann_dir: annotation path
    
    """
    ann_list = list()
    if isinstance(ann_dir, list):
        for ann_path in ann_dir:
            ann_list += gb.glob(ann_path + r"/*.xml")
    elif isinstance(ann_dir, str):
        ann_list = gb.glob(ann_dir + r"/*.xml")
    else:
        raise ValueError("Invalid ann_dir data type: must be string or list.")
        
    heights = list()
    for ann_file in ann_list:
        boxes = pvoc.get_boxes(ann_file)
        labels = pvoc.get_labels(ann_file)
        
        for label, box in zip(labels, boxes):
            if label == "striation":
                h = (box[1]+box[3]) / 2
                #h_rev = 540 - h
                heights.append(h)
                #heights.append(h_rev)
            
    plt.hist(heights, bins=54, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlim(0, 540)
    plt.xlabel("Striation height")
    plt.ylabel("Striation number")
    plt.title("Striation height histogram")
    plt.show()
    
    
def flip_image(img_file):
    img = Image.open(img_file)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(img_file)
    
    
if __name__ == "__main__":
    """
    ann_dir = [r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\lightgray",
               r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\darkgray",
               r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\white"]
    striation_height_histogram(ann_dir)
    """
    
    img_dir = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\valid_rev'
    img_list = gb.glob(img_dir+r"/*.bmp")
    for img_file in img_list:
        flip_image(img_file)