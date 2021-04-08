#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2020
Updated on 04.08.2021

Author: haoshuai@handaotech.com
'''

import os
import sys
import cv2
import json
import glob as gb
import numpy as np
from matplotlib import pyplot as plt


class CheckDefectVariation(object):
    def __init__(self, params):
        self.updateParams(params)
        
    def updateParams(self, params):
        self.labels = params['labels']
        self.line_width = params['line_width']
        self.image_dir = params['image_dir']
        self.image_suffix = params['image_suffix']
        self.params = params
        
        self.center_id = 1
        self.neighbor_id = 2
        
    def check(self, json_file):
        if not os.path.isfile(json_file):
            raise ValueError('Cannot find the input json file.')
            
        image_file = self._getImageFile(json_file)
        if image_file is None:
            raise ValueError('Cannot find the corresponding image file.')
            
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        print('Processing file', json_file, '...')
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            img_info, lines, labels = self._fetchJsonObject(js_obj)
            
            for points, label in zip(lines, labels):
                if label in self.labels: self._process(image, points, label)
            
            f.close()
            
    def _process(self, image, points, label):
        top_points, bottom_points = self._createTBNeighbor(points)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = self._drawMask(mask, points, self.center_id)
        mask = self._drawMask(mask, top_points, self.neighbor_id)
        mask = self._drawMask(mask, bottom_points, self.neighbor_id)
        
        var = self._calVariation(image, mask, self.center_id)
        ref_var = self._calVariation(image, mask, self.neighbor_id)
        
        print('Variation =', var)
        print('Reference variation =', ref_var)
        print()
        
        plt.subplot(1,2,1), plt.imshow(image, cmap='gray'), plt.title('Image')
        plt.subplot(1,2,2), plt.imshow(mask), plt.title('Mask')
        plt.show()
        
    def _calVariation(self, image, mask, id):
        roi = image[mask==id]
        average = sum(roi) / len(roi)
        
        var = 0
        for item in roi:
            var += (item - average)**2
        var /= len(roi)
        
        return var
        
    def _drawMask(self, mask, points, color_index):
        for i in range(len(points)-1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            mask = cv2.line(mask,(int(x0),int(y0)), (int(x1),int(y1)), color_index, self.line_width)
        
        return mask
            
    def _getImageFile(self, json_file):
        json_path, filename = os.path.split(json_file)
        fname, _ = os.path.splitext(filename)
        
        if self.image_dir is None: image_path = json_path
        else: image_path = self.image_dir
        
        image_file = None        
        for suffix in self.image_suffix:
            temp_image_file = os.path.join(image_path, fname+suffix)
            if os.path.isfile(temp_image_file):
                image_file = temp_image_file
                break
                
        return image_file
        
    def _fetchJsonObject(self, js_obj):
        img_file = js_obj['imagePath']
        img_h = js_obj['imageHeight']
        img_w = js_obj['imageWidth']
        img_info = {
            'img_file': img_file,
            'img_h': img_h,
            'img_w': img_w
        }
        
        lines, labels = [], []
        for elem in js_obj['shapes']:
            lines.append(elem['points'])
            labels.append(elem['label'])
        
        return img_info, lines, labels
        
    def _createTBNeighbor(self, points):
        top_points, bottom_points = [], []
        
        for point in points:
            x, y = point
            top_points.append([x, y-1.2*self.line_width])
            bottom_points.append([x, y+1.2*self.line_width])
            
        return top_points, bottom_points
    
    
if __name__ == "__main__":
    params = {
        'image_dir': None,
        'image_suffix': ['.bmp', '.png', '.jpg', '.tif'],
        'labels': ['s', 'striation', 'spandex'],
        'line_width': 10
    }
    
    json_file = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\label_test\MER2-041-436U3M(FDL21010006)_2021-03-25_15_36_37_955-3.json'
    cdv = CheckDefectVariation(params)
    cdv.check(json_file)
  
    