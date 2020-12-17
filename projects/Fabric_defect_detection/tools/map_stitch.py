# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
import glob as gb

from PascalVocParser import PascalVocXmlParser


def draw_box(img, box, color=255, thickness=2):
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)
    return img
    
    
def get_time_info(img_file):
    _, filename = os.path.split(img_file)
    sec = filename[46:48]
    msec = filename[49:51]
    
    if sec[0] == "0": sec = sec[1]
    if msec[0] == "0": msec = msec[1:]
    if msec[0] == "0": msec = msec[1]
    
    return float(sec), float(msec)
    
    
def get_time_interval(img_file, pre_img_file) -> float:
    cur_sec, cur_msec = get_time_info(img_file)
    pre_sec, pre_msec = get_time_info(pre_img_file)
    
    if cur_sec < pre_sec: 
        intv_sec = 60.0 + cur_sec - pre_sec
    else: 
        intv_sec = cur_sec - pre_sec
        
    intv_msec = cur_msec - pre_msec
    intv = (intv_sec * 1000.0 + intv_msec) / 1000.0
    
    return intv


def get_ann_file(img_file, suffix=".xml"):
    filename, _ = os.path.split(img_file)
    return filename + suffix


def map_time_step(img_path, 
                  rpm=19.6, 
                  field=13, 
                  diameter=70, 
                  suffix=".bmp", 
                  save_path=None):
                  
    img_list = sorted(gb.glob(img_path + r"/*"+suffix), key=os.path.getmtime)
    pvoc = PascalVocXmlParser()
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]
    
    # Get the linear velocity of the rotating cloth
    perimeter = np.pi * diameter
    min_velo = rpm * perimeter
    velocity = min_velo / 60.0
    
    for i in range(1, len(img_list), 1):
        img_file = img_list[i]
        ann_file = get_ann_file(img_file)
        boxes = pvoc.get_boxes(ann_file)
        
        pre_img_file = img_list[i-1]
        pre_ann_file = get_ann_file(pre_img_file)
        pre_boxes = pvoc.get_boxes(pre_ann_file)
        
        intv = get_time_interval(img_file, pre_img_file)
        
        for pre_box in pre_boxes:
            pre_x1, pre_x2 = pre_box[0], pre_box[2]
            
        
    
        

        
if __name__ == "__main__":
    filename = "MER2-041-436U3M(FDL17100010)_2020-12-01_16_35_13_594-0.png"
    