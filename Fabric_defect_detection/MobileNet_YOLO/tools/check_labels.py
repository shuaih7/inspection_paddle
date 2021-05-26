#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2020
Updated on 04.08.2021

Author: haoshuai@handaotech.com
'''

import os
import sys
import json
import glob as gb
import numpy as np
from PascalVocParser import PascalVocXmlParser


class CheckLabels(object):
    def __init__(self, params):
        self.pvoc = PascalVocXmlParser()
        self.updateParams(params)
        
    def updateParams(self, params):
        self.format = params['format']
        self.is_alert = params['is_alert']
        self.collection = params['collection']
        self.params = params
        self.matrix = {}
        self.abnormal_matrix = {}
        
    def __call__(self, input):
        if os.path.isfile(input):
            _, suffix = os.path.splitext(input)
            if suffix[1:] == 'json':
                self.checkJson(input)
            elif suffix[1:] == 'xml':
                self.checkXml(input)
            else:
                raise ValueError('Unsupported input format.')
        elif os.path.exists(input):
            self.checkDir(input)
        else:
            raise ValueError('Invalid input.')
        
    def checkDir(self, dir):
        if self.format.lower() in ['labelme', 'json']:
            file_list = gb.glob(dir + r'/*.json')
            for json_file in file_list: 
                self.checkJson(json_file, disp=False)
                
        elif self.format.lower() in ['labelimg', 'xml', 'pascalvoc', 'voc']:
            file_list = gb.glob(dir + r'/*.xml')
            for xml_file in file_list:
                self.checkXml(xml_file, disp=False)
        
        else:
            raise ValueError('Invalid format.')
        
        self._display()
               
    def checkJson(self, json_file, disp=True):
        print('Processing file', json_file, '...')
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            for elem in js_obj['shapes']:
                self._register(elem['label'])
                if self.is_alert: 
                    self._register_abnormal(elem['label'], json_file)
            f.close()
        
        if disp: self._display()
     
    def checkXml(self, xml_file, disp=True):  
        print('Processing file', xml_file, '...')
        labels = self.pvoc.get_labels(xml_file)
        for label in labels:
            self._register(label)
            if self.is_alert: 
                self._register_abnormal(label, xml_file)
            
        if disp: self._display()
        
    def _register(self, label):
        if label not in self.matrix:
            self.matrix[label] = 1
        else:
            self.matrix[label] += 1
            
    def _register_abnormal(self, label, ann_file):
        if label in self.collection: return
        
        _, filename = os.path.split(ann_file)
        if label not in self.abnormal_matrix:
            self.abnormal_matrix[label] = [filename]
        else:
            self.abnormal_matrix[label].append(filename)
        
    def _display(self):
        print('\nDisplaying the results:\n')
        for label in self.matrix:
            print(label + ': ' + str(self.matrix[label]))
        
        if not self.is_alert: return
        for ab_label in self.abnormal_matrix:
            print('Abnormal label: ' + ab_label + '\n')
            for filename in self.abnormal_matrix[ab_label]:
                print('  ' + filename + '\n')
        
        
if __name__ == "__main__":
    params = {
        'format': 'labelme',
        'is_alert': True,
        'collection': ['defect', 'spandex', 's', 'striation']
    }
    
    data_dir = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\label_test\MER2-041-436U3M(FDL21010006)_2021-03-25_15_36_37_717-0.xml'
    check_labels = CheckLabels(params)
    check_labels(data_dir)
        
    