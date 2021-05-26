# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
import glob as gb
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageEnhance


def get_average_value(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape[:2]
    average = img.sum() / (img_h*img_w)
    
    return average


def show_single_distort(img_file, mode, lower=0.9, upper=1.1, save_path=None):
    img = Image.open(img_file)
    img1, img2 = img.copy(), img.copy()
    mode = mode.lower()
    
    if mode == "brightness":
        img_lower = ImageEnhance.Brightness(img1).enhance(lower)
        img_upper = ImageEnhance.Brightness(img2).enhance(upper)
    elif mode == "contrast":
        img_lower = ImageEnhance.Contrast(img1).enhance(lower)
        img_upper = ImageEnhance.Contrast(img2).enhance(upper)
    elif mode == "sharpness":
        img_lower = ImageEnhance.Contrast(img1).enhance(lower)
        img_upper = ImageEnhance.Contrast(img2).enhance(upper)
    else:
        raise ValueError("Invalid mode.")
    
    if save_path is not None:
        if not os.path.exists(save_path): return
        _, filename = os.path.split(img_file)
        fname, suffix = os.path.splitext(filename)
        img.save(os.path.join(save_path, fname+"_"+mode+"_"+"1.0"+suffix))
        img_lower.save(os.path.join(save_path, fname+"_"+mode+"_"+str(lower)+suffix))
        img_upper.save(os.path.join(save_path, fname+"_"+mode+"_"+str(upper)+suffix))
    else:
        cv2.imshow(mode+"_"+str(lower), np.array(img_lower, dtype=np.uint8))
        cv2.waitKey(0)
        cv2.imshow(mode+"_"+"1.0", np.array(img, dtype=np.uint8))
        cv2.waitKey(0)
        cv2.imshow(mode+"_"+str(upper), np.array(img_upper, dtype=np.uint8))
        cv2.waitKey(0)
    
    
if __name__ == "__main__":
    img_file = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\data_samples\sample_lightgray\300u_12gain.bmp"
    save_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\data_samples\random_brightness_lightgray"
    show_single_distort(img_file, mode="brightness", lower=0.6, upper=1.5, save_path=save_path)
    #print(get_average_value(img_file))