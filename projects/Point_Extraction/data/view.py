#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 12.29.2020
Updated on 12.29.2020

Author: haoshaui@handaotech.com
'''

import os
import sys
import cv2
import glob as gb
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def show_mask(mask_file):
    img = Image.open(mask_file)
    _, filename = os.path.split(mask_file)
    plt.imshow(img), plt.title(filename)
    plt.show()
    
    
if __name__ == "__main__":
    mask_path = r"E:\Projects\Integrated_Camera\point_extraction\mini_pet\annotations\trimaps"
    mask_list = gb.glob(mask_path + r"/*.png")
    
    for mask_file in mask_list: 
        show_mask(mask_file)