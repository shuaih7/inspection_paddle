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
from matplotlib import pyplot as plt
from lxml.etree import Element, SubElement, tostring, ElementTree, XMLParser, parse


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
    
    
def random_select_list(file_list, file_path, ratio=1.0):
    new_file_list = []
    file_list = shuffle(file_list)
    slc_len = int(ratio*len(file_list))
    file_list = file_list[:slc_len]
    
    for file in file_list:
        _, filename = os.path.split(file)
        fname, _ = os.path.splitext(filename)
        item = os.path.join(file_path, fname)
        new_file_list.append(item)

    return new_file_list
    
    
def create_txt_file(path_dict, save_path=None, save_name='List', is_shuffle=True):
    data_path = path_dict['data_path']
    file_paths = path_dict['file_paths']
    # suffixes = path_dict['suffixes']
    
    full_path_list = []
    for path_info in file_paths:
        file_path = path_info[0]
        ratio = min(1, max(0, path_info[1]))
        full_path = os.path.join(data_path, file_path)
        file_list = gb.glob(full_path + r'/*.xml')
        file_list = random_select_list(file_list, file_path, ratio)
        full_path_list += file_list
    
    if is_shuffle: full_path_list = shuffle(full_path_list)
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for fname in full_path_list:
            f.write(fname)
            f.write("\n")
        f.close()
    
    print('Done')
    
    
def show_histogram(boxes):
    heights = list()
    for box in boxes:
        h = (box[1]+box[3]) / 2
        h_rev = 540 - h
        heights.append(h)
        heights.append(h_rev)
            
    plt.hist(heights, bins=54, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlim(0, 540)
    plt.xlabel("Striation height")
    plt.ylabel("Striation number")
    plt.title("Striation height histogram")
    plt.show()
    
    
def set_vertical_boundary(img_h, striations, max_ver_offset):
    """ Set the vertical top and bottom position for cropping
    
    Args:
        img_h: image height
        striation: list for the striation defect bounding boxes
        max_ver_offset: maximum vertical pixel-wise offset
        
    Returns:
        top: top y position
        bottom: bottom y position
    
    """
    if len(striations) == 0: return None, None
    
    # Get the top and bottom margin
    top_margin, bottom_margin = img_h, img_h
    for striation in striations:
        cur_top_margin = striation[1]
        cur_bottom_margin = img_h - striation[3]
        
        if cur_top_margin < top_margin: 
            top_margin = cur_top_margin
        if cur_bottom_margin < bottom_margin:
            bottom_margin = cur_bottom_margin
    
    # Randomly select the top or bottom offset
    kind = np.random.randint(0,2)
    if kind == 0: # top
        bottom = img_h
        offset = np.random.randint(0, min(top_margin, max_ver_offset)+1)
        top = offset
    else:
        top = 0
        offset = np.random.randint(0, min(bottom_margin, max_ver_offset)+1)
        bottom = img_h - offset
        
    return top, bottom
        

def set_horizontal_boundary(img_w, defects, max_hor_offset, aspect_ratio=None):
    """ Set the horizontal left and right position for cropping
    
    Args:
        img_w: image width
        defects: list for the long defect bounding boxes
        max_hor_offset: maximum horizontal pixel-wise offset
        aspect_ratio: boolean value for whether to keep the aspect ratio
        
    Returns:
        left: left x position
        right: right x position
    
    """
    if len(defects) == 0: return None, None
    
    # Get the left and right margin
    left_margin, right_margin = img_w, img_w
    for defect in defects:
        cur_left_margin = defect[0]
        cur_right_margin = img_w - defect[2]
        
        if cur_left_margin < left_margin: 
            left_margin = cur_left_margin
        if cur_right_margin < right_margin:
            right_margin = cur_right_margin
    
    # Randomly select the left or right offset
    kind = np.random.randint(0,2)
    offset = np.random.randint(0, max_hor_offset+1)
    if kind == 0: # left
        right = img_w
        left = min(left_margin, offset)
    else:
        left = 0
        right = img_w - min(right_margin, offset)
        
    return left, right
    
    
def rearrange_boxes(roi, boxes, aspect_ratio=None):
    left, top, right, bottom = roi
    if aspect_ratio is not None:
        hor_ratio = aspect_ratio[0] / (right-left)
        ver_ratio = aspect_ratio[1] / (bottom-top)
    else:
        hor_ratio = 1.0
        ver_ratio = 1.0
    
    nboxes = []
    for box in boxes:
        x0, y0, x1, y1 = box
        nx0 = int(max(0, x0-left)*hor_ratio)
        ny0 = int(max(0, y0-top)*ver_ratio)
        nx1 = int((x1-left)*hor_ratio)
        ny1 = int((y1-top)*ver_ratio)
        nboxes.append([nx0, ny0, nx1, ny1])
    
    return nboxes
    
    
def random_crop(ann_file, 
                pos=0.2, 
                img_dir=None, 
                img_suffix=".bmp",
                max_ver_offset=50, 
                max_hor_offset=70, 
                keep_ratio=True):
    """ Random crop the input image based on the defects localities
        The detailed croppping procedures are: 
        
        1. Search for the defects closest to the image upper and lower boundaries
        2. Randomly select the vertical offset
        3. Search for the defects closest to the image left and right boundaries
        4. Randomly select the horizontal offset, if keep_ratio, then defined automatically
        5. Crop the image based on the offsets
        
    Args:
        ann_file: annotation file
        pos: posibility for the crop procedure
        img_dir: image directory
        img_suffix: image file suffix
        max_ver_offset: maximum vertical offset
        max_hor_offset: maximum horizontal offset
        keep_ratio: bool value for keeping the image w/h ratio
        
    Returns:
    
    """
    ann_dir, filename = os.path.split(ann_file)
    fname, _ = os.path.splitext(filename)
    
    if img_dir is None: img_dir = ann_dir
    img_file = os.path.join(img_dir, fname+img_suffix)
    node_root = parse(ann_file, XMLParser(remove_blank_text=True)).getroot()
    
    # 1. Get the image width, height, and aspect ratio
    height = int(node_root.findtext("./size/height"))
    width = int(node_root.findtext("./size/width"))
    ratio = width / height
    
    # 2. Gather and sort the defects' information
    defects = list()
    striations = list()
    for obj in node_root.iter("object"):
        xmin = float(obj.findtext("bndbox/xmin"))
        ymin = float(obj.findtext("bndbox/ymin"))
        xmax = float(obj.findtext("bndbox/xmax"))
        ymax = float(obj.findtext("bndbox/ymax"))
        name = obj.findtext("name")
        
        if name == "defect": defects.append([xmin, ymin, xmax, ymax])
        elif name == "striation": striations.append([xmin, ymin, xmax, ymax])
    
    top, bottom = set_vertical_boundary(height, striations, max_ver_offset)
    left, right = set_horizontal_boundary(width, defects, max_hor_offset, ratio)
    # boxes = rearrange_boxes([left, top, right, bottom], defects+striations)
    boxes = rearrange_boxes([left, top, right, bottom], striations, aspect_ratio=[width, height])
    
    return boxes

        
if __name__ == "__main__":
    file_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\valid"
    save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Fabric_defect_detection\MobileNet_YOLO"
    
    train_path_dict = {
        'data_path': r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.2.0\dataset\train',
        'file_paths': [
            [r'darkgray-300mus-12gain-horizontal_type2+vertical', 1.0],
            [r'white-300mus-12gain-horizontal_type2+vertical', 1.0],
            [r'train_v1.1.0', 0.5]
        ]
    }
    
    valid_path_dict = {
        'data_path': r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.2.0\dataset\valid',
        'file_paths': [
            [r'darkgray-300mus-8gain-horizontal_type2+vertical', 1.0],
            [r'darkgray-300mus-12gain-vertical', 1.0],
            [r'darkgray-300mus-14gain-horizontal_type2+vertical', 1.0],
            [r'white-300mus-8gain-horizontal_type2+vertical', 1.0],
            [r'white-300mus-12gain-vertical', 1.0],
            [r'white-300mus-14gain-horizontal_type2+vertical', 1.0]
        ]
    }
    
    create_txt_file(valid_path_dict, save_path, save_name='valid')
    # write_into_txt(file_path, save_path=save_path, save_name="valid")

    """
    num_file = 3000
    ann_dir_white = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\white"
    ann_dir_light = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\lightgray"
    ann_dir_dark = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\darkgray"
    ann_list = gb.glob(ann_dir_white + r"/*.xml") + gb.glob(ann_dir_light + r"/*.xml") + gb.glob(ann_dir_dark + r"/*.xml")
    file_index = 0
    boxes = []
    
    for ann_file in ann_list:
        while file_index < num_file:
            boxes += random_crop(ann_file, max_ver_offset=70, max_hor_offset=70)
            file_index += 1
    show_histogram(boxes)
    """
    