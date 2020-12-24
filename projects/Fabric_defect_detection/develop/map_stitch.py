# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
import glob as gb

from PascalVocParser import PascalVocXmlParser


def draw_boxes(img, boxes, color=255, thickness=2):
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)
    return img
    
    
def get_time_info(img_file):
    _, filename = os.path.split(img_file)
    sec = filename[46:48]
    msec = filename[49:52]
    
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
    
    print("The current time interval is", intv, "second(s).")
    
    return intv


def get_ann_file(img_file, suffix=".xml"):
    filename, _ = os.path.splitext(img_file)
    return filename + suffix
    

def map_next_boxes(boxes, speed, intv, field, img_shape):
    img_h, img_w = img_shape
    distance = speed * intv # cm
    pix_dist = int(distance / field * img_w)
    
    print("The current pixel distance is", pix_dist)
    
    nboxes = []
    for box in boxes:
        nx1, nx2 = box[0] - pix_dist, box[2] - pix_dist
        if nx1 < 0: 
            if nx2 > 1: nboxes.append([0, box[1], nx2, box[3]])
        else: nboxes.append([nx1, box[1], nx2, box[3]])
            
    return nboxes


def map_time_step(img_path, 
                  rpm=19.6, 
                  field=14, 
                  diameter=70, 
                  suffix=".bmp", 
                  save_path=None):
                  
    img_list = sorted(gb.glob(img_path + r"/*"+suffix), key=os.path.getmtime)
    pvoc = PascalVocXmlParser()
    
    # Get the linear velocity of the rotating cloth
    perimeter = np.pi * diameter
    min_velo = rpm * perimeter
    speed = min_velo / 60.0
    
    for i in range(1, len(img_list), 1):
        img_file = img_list[i]
        print(img_file)
        ann_file = get_ann_file(img_file)
        if os.path.isfile(ann_file):
            boxes = pvoc.get_boxes(ann_file)
        else: boxes = []
        
        pre_img_file = img_list[i-1]
        pre_ann_file = get_ann_file(pre_img_file)
        if os.path.isfile(pre_ann_file): 
            pre_boxes = pvoc.get_boxes(pre_ann_file)
        else: pre_boxes = []
        
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        intv = get_time_interval(img_file, pre_img_file)
        nboxes = map_next_boxes(pre_boxes, speed, intv, field, (img_h, img_w))
            
        #print(boxes)
        #print(nboxes)
        img = draw_boxes(img, boxes, color=(0,0,255))
        img = draw_boxes(img, nboxes, color=(255,0,0))
        
        if save_path is not None:
            _, filename = os.path.split(img_file)
            cv2.imwrite(os.path.join(save_path, filename), img)
        else: 
            cv2.imshow("image", img)
            cv2.waitKey(0)
        

if __name__ == "__main__":
    img_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\dataset_v1\valid"
    map_time_step(img_path)
    