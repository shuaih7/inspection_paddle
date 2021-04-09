#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.09.2021
Updated on 04.09.2021

Author: haoshuai@handaotech.com
'''

import os
import numpy as np


def box_to_center_relative(box, img_height, img_width):
    """
    Convert COCO annotations box with format [x1, y1, w, h] to
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w - 1, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, img_height - 1)

    x = (x1 + x2) / 2 / img_width
    y = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return np.array([x, y, w, h])
    
    
def create_box_from_polygon(points, img_h, img_w, min_h=5, min_w=5):
    points = np.array(points, dtype=np.float32)
    
    xmin = min(points[:,0])
    xmax = max(points[:,0])
    ymin = min(points[:,1])
    ymax = max(points[:,1])
    
    if xmax - xmin < min_w:
        off_x = min_w / 2
        center_x = (xmin+xmax) / 2
        
        if center_x - off_x < 0:
            xmin, xmax = 0, min_w
        elif center_x + off_x >= img_w:
            xmin, xmax = img_w - min_w, img_w
        else:
            xmin, xmax = center_x - off_x, center_x + off_x
            
    if ymax - ymin < min_h:
        off_y = min_h / 2
        center_y = (ymin+ymax) / 2
        
        if center_y - off_y < 0:
            ymin, ymax = 0, min_h
        elif center_y + off_y >= img_h:
            ymin, ymax = img_h - min_h, img_h
        else:
            ymin, ymax = center_y - off_y, center_y + off_y     
    #bbox = box_to_center_relative([xmin, ymin, xmax-xmin, ymax-ymin], img_h, img_w)
    
    return [xmin, ymin, xmax, ymax]