#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2021
Updated on 04.09.2021

Author: haoshuai@handaotech.com
'''

import os
import xml
import json
import numpy as np
from .utils import box_to_center_relative, create_box_from_polygon


class LabelmeParser(object):
    def __init__(self, train_parameters):
        self.update(train_parameters)
        
    def update(self, train_parameters):
        self.train_parameters = train_parameters
        self.min_h = 20  # Minimum label height
        self.min_w = 20  # Minimum label width
        
    def __call__(self, ann_file):
        bbox_labels = []
        with open(ann_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            img_h = js_obj['imageHeight']
            img_w = js_obj['imageWidth']
            
            bbox_sample = []
            for elem in js_obj['shapes']:
                difficult = 0.0
                bbox_sample = []
                bbox_sample.append(float(self.train_parameters['label_dict'][elem['label']]))
                xmin, ymin, xmax, ymax = create_box_from_polygon(elem['points'], 
                    img_h, img_w, self.min_h, self.min_w)
                bbox = box_to_center_relative([xmin, ymin, xmax-xmin, ymax-ymin], img_h, img_w)
                bbox_sample.append(float(bbox[0]))
                bbox_sample.append(float(bbox[1]))
                bbox_sample.append(float(bbox[2]))
                bbox_sample.append(float(bbox[3]))
                bbox_sample.append(difficult)
                bbox_sample.append(elem['points'])
                bbox_labels.append(bbox_sample)
            f.close()
        
        return bbox_labels
    

class PascalVocParser(object):
    def __init__(self, train_parameters):
        self.update(train_parameters)
        
    def update(self, train_parameters):
        self.train_parameters = train_parameters
        
    def __call__(self, ann_file):
        bbox_labels = []
        root = xml.etree.ElementTree.parse(ann_file).getroot()
        # Fetch image information
        size = root.find('size')
        im_height = float(size.find('height').text)
        im_width = float(size.find('width').text)
        # Fetch bbox information
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
        
        return bbox_labels
        