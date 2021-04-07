#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.06.2020
Updated on 04.06.2021

Author: haoshuai@handaotech.com
'''

import os
import sys
import json
import glob as gb
import numpy as np
from lxml.etree import Element, SubElement, tostring, ElementTree, XMLParser, parse


class LabelmePascalVOC(object):
    def __init__(self, params):
        self.kind = params['kind']
        self.det_params = params['det']
        self.seg_params = params['seg']
        self.params = params
        
    def transferDetFile(self, json_file):
        print('Processing file', json_file, '...')
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            img_info, boxes, labels = self._fetchJsonObject(js_obj)
            xml_tree = self._createPvoc(img_info, boxes, labels)
            if xml_tree is None: return # Only write into xml if there exists defect
            
            json_dir, filename = os.path.split(json_file)
            fname, _ = os.path.splitext(filename)
            if self.params['save_dir'] is None:
                save_name = os.path.join(json_dir, fname+".xml")
            else:
                save_name = os.path.join(self.params['save_dir'], fname+".xml")
            self._save(xml_tree, save_name)
            
            f.close()
        
    def transferSegFile(self, json_file):
        print('Processing file', json_file, '...')
        
    def transferDetPath(self, path):
        file_list = gb.glob(path + r'/*.json')
        for file in file_list:
            self.transferDetFile(file)
        
    def transferSegPath(self, path):
        file_list = gb.glob(path + r'/*.json')
        for file in file_list:
            self.transferSegFile(file)
        
    def __call__(self, input):
        if os.path.exists(input):
            if self.kind == 'det':
                self.transferDetPath(input)
            elif self.kind == 'seg':
                self.transferSegPath(input)
            else:
                raise ValueError('Invalid kind.')
        elif os.path.isfile(input):
            self.transferFile(input)
        else:
            raise ValueError('Invalid input.')
            
    def _fetchJsonObject(self, js_obj):
        img_file = js_obj['imagePath']
        img_h = js_obj['imageHeight']
        img_w = js_obj['imageWidth']
        img_info = {
            'img_file': img_file,
            'img_h': img_h,
            'img_w': img_w
        }
        
        boxes, labels = [], []
        for elem in js_obj['shapes']:
            boxes.append(self._createBox(elem['points'], img_info))
            labels.append(elem['label'])
        
        return img_info, boxes, labels
        
    def _labelShift(self, label):
        if label not in self.params['label_shift']: 
            return label
        else:
            return self.params['label_shift'][label]
        
    def _createBox(self, points, img_info):
        img_h = img_info['img_h']
        img_w = img_info['img_w']
        min_h = self.params['det']['min_h']
        min_w = self.params['det']['min_w']
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
        
        return [xmin, ymin, xmax, ymax]
         
    def _createPvoc(self, img_info, boxes, labels):
        if len(boxes) == 0: return None
        else: 
            img_file = img_info['img_file']
            img_h = img_info['img_h']
            img_w = img_info['img_w']
            
        node_root = Element('annotation')
         
        # Folder, filename, and path
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = ''
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_file
        node_path = SubElement(node_root, 'path')
        node_path.text = ''
        
        # Data source
        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = "Unknown"
         
        # Image size and segmented
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(img_w)
        node_height = SubElement(node_size, 'height')
        node_height.text = str(img_h)
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = "1"
        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = "0"
         
        for box, label in zip(boxes, labels): 
            self._createPvocObject(node_root, box, label)
        xml_tree = ElementTree(node_root)
        
        return xml_tree
        
    def _createPvocObject(self, node_root, bbx, label,
                           pose="Unspecified",
                           truncated="1",
                           difficult="0"):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = self._shiftlabel(label)
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = pose
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = truncated
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = difficult
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(bbx[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(bbx[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(bbx[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(bbx[3]))
        
    def _save(self, xml_tree, save_name):
        xml_tree.write(save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
            

if __name__ == "__main__":
    params = {
        'kind': 'det',
        'save_dir': None,
        'label_shift': {
            's': 'striation',
            'defect': 'defect'
        },
        'det': {
            'min_h': 20,
            'min_w': 20
        },
        'seg': {
        }
    }
    
    transfer = LabelmePascalVOC(params)
    
    label_folder = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\label_test'
    transfer(label_folder)

        