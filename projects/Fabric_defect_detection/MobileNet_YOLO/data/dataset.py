#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 12.09.2020
Updated on 12.09.2020

Author: haoshaui@handaotech.com
'''

import os
import sys
import cv2
from PIL import Image
import numpy as np
import glob as gb
from sklearn.utils import shuffle


def write_into_txt(file_path, suffix=".xml", save_path=None, save_name="List", is_shuffle=True): 
    file_list = gb.glob(file_path + r"/*"+suffix)
    if is_shuffle: file_list = shuffle(file_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for file in file_list:
            _, filename = os.path.split(file)
            fname, _ = os.path.splitext(filename)
            f.write(fname)
            f.write("\n")
        f.close()
    
    print("Done")


class DatasetGenerator(object):
    """Generates the traininng and validation dataset

    Load the original input images, randomly crop the images,
    and do some augmentations.

    Attributes:
        train_paths: A list of directories which contain the training images.
        valid_paths: A list of directories which contain the validation images.
        suffix: Suffix of the images
        height_range: Height range of the crop window
        width_range: Width range of the crop window
    """
    def __init__(self,
                 train_paths,
                 valid_paths,
                 suffix=".bmp",
                 heihgt_range=(500,540),
                 width_range=(512,600),
                 bright_range=None,
                 contrast_range=None): 
        pass
        
        
if __name__ == "__main__":
    file_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\dataset_v1\valid"
    save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Fabric_defect_detection\MobileNet_YOLO"
    
    write_into_txt(file_path, save_path=save_path, save_name="valid")
    