#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2021
Updated on 04.08.2021

Author: haoshuai@handaotech.com
'''

import os
import xml
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


class PascalVocParser(object):
    def __init__(self, train_parameters):
        self.update(train_parameters)
        
    def update(self, train_parameters):
        self.train_parameters = train_parameters
        
    def __call__(self, ann_file, im_height, im_width):
        bbox_labels, bbox_sample = [], []
        root = xml.etree.ElementTree.parse(ann_file).getroot()
        for object in root.findall('object'):
            bbox_sample = []
            # start from 1
            bbox_sample.append(float(self.train_parameters['label_dict'][object.find('name').text]))
            bbox = object.find('bndbox')
            box = [float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text) - float(bbox.find('xmin').text), float(bbox.find('ymax').text)-float(bbox.find('ymin').text)]
            # print(box, img.size)
            difficult = float(object.find('difficult').text)
            bbox = box_to_center_relative(box, im_height, im_width)
            # print(bbox)
            bbox_sample.append(float(bbox[0]))
            bbox_sample.append(float(bbox[1]))
            bbox_sample.append(float(bbox[2]))
            bbox_sample.append(float(bbox[3]))
            bbox_sample.append(difficult)
            bbox_labels.append(bbox_sample)
        
        return bbox_labels, bbox_sample
        