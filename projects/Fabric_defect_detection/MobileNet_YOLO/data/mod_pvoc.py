#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.07.2020
Updated on 04.07.2021

Author: haoshuai@handaotech.com
'''

import os
import sys
import glob as gb
import numpy as np
from xml.etree import ElementTree as ET


class ModPVOC(object):
    def __init__(self, params):
        self.updateParams(params)
        
    def updateParams(self, params):
        self.kind = params['kind']
        self.params = params
        
    def modifyPath(self, path): 
        xml_list = gb.glob(path + r'/*.xml')
        if self.kind == 'det':
            for xml_file in xml_list: self.modifyDetFile(xml_file)
        elif self.kind == 'seg':
            for xml_file in xml_list: self.modifySegFile(xml_file)
        else:
            raise ValueError('Invalid kind.')
                
    def modifyFile(self, xml_file):
        if self.kind == 'det':
            self.modifyDetFile(xml_file)
        elif self.kind == 'seg':
            self.modifySegFile(xml_file)
        else:
            raise ValueError('Invalid kind.')
            
    def modifyDetFile(self, xml_file):
        print('Processing file', xml_file, '...')
        xml_tree = self._getTree(xml_file)
        
        for elem in xml_tree.findall('object'):
            subelem = elem.find('name')
            self._shiftLabels(subelem)
        
        xml_dir, filename = os.path.split(xml_file)
        fname, _ = os.path.splitext(filename)
        if self.params['save_dir'] is None:
            save_name = os.path.join(xml_dir, fname+".xml")
        else:
            save_name = os.path.join(self.params['save_dir'], fname+".xml")
        self._save(xml_tree, save_name)
        
    def modifySegFile(self, xml_file):  
        print('Processing file', xml_file, '...')
        
    def __call__(self, input):
        if os.path.exists(input):
            self.modifyPath(input)
        elif os.path.isfile(input):
            self.modifyFile(input)
        else:
            raise ValueError('Invalid input.')
            
    def _shiftLabels(self, element):
        text = element.text
        for label in self.params['general']['shift_labels']:
            if text == label[0]: element.text = label[1]
            
    def _getTree(self, xml_file):
        return ET.parse(xml_file)
    
    def _getRoot(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return root
            
    def _save(self, xml_tree, save_name):
        xml_tree.write(save_name, xml_declaration=False, encoding='utf-8')
        
        
if __name__ == "__main__":
    params = {
        'kind': 'det',
        'save_dir': None,
        'general': {
            'folder': None,
            'filename': None,
            'path': None,
            'source': None,
            'shift_labels': [('s', 'striation')]
        }
    }
    
    xml_path = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\label_test'
    modifier = ModPVOC(params)
    modifier(xml_path)