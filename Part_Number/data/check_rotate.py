import os
import sys
import cv2
import time
import glob as gb
import numpy as np
from .utils import *


def check_folder(img_folder, save_folder="", save_suffix=None):
    img_list = load_image_files(img_folder)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, suffix = os.path.splitext(filename)
        if save_suffix is None: save_suffix = suffix
        save_name = os.path.join(save_folder, fname + save_suffix)
        
        image = cv2.imread(img_file, -1)
        cv2.namedWindow(filename,0);
        cv2.resizeWindow(filename, 640, 480);
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        
        dir = int(input("Direction id:"))
        
        if dir == 2:
            image = cv2.flip(image, -1)
        elif dir == 1:
            image = rotate_bound(image, -90)
        elif dir == 3:
            image = rotate_bound(image, 90)
        """
        cv2.namedWindow("Result",0);
        cv2.resizeWindow("Result", 640, 480);
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        """
        cv2.imwrite(save_name, image)
        print("Image file saved to", save_name)
        
        
if __name__ == "__main__":
    img_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA"
    save_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA_rotate"
    save_suffix = ".png"
    check_folder(img_folder, save_folder, save_suffix)
    # switch_suffix(save_folder, ".png", save_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA_rect")