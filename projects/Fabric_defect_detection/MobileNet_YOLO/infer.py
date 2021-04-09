#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.09.2021
Updated on 04.09.2021

Author: haoshuai@handaotech.com
'''

import os
import sys
import cv2
import xml
import time
import json
import shutil
import glob as gb
import numpy as np
import paddle.fluid as fluid

from PIL import Image
from PIL import ImageDraw

import config
from data import create_box_from_polygon


train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
path = train_parameters['freeze_dir']  # 'model/freeze_model'
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe, model_filename='__model__', params_filename='params')
    
    
class Evaluation(object):
    def __init__(self, params):
        self.update(params)
    
    def update(self, params)
        self.iou_thresh = params['iou_thresh']
        self.label_format = params['label_format']
        self.label_dir = params['label_dir']
        self.save_dir = params['save_dir']
        self._init_matrix()
        self.params = params
        
    def __call__(self, input):
        if os.path.isfile(input):
            results = self.infer(input)
            self._process_results(input, results)
        elif os.path.exists(input):
            image_list = self._load_image_list(input)
            for image_file in image_list:
                results = self.infer(input)
                self._process_results(image_file, results)
        else:
            raise ValueError('Invalid input.')
         
    def infer(self, image_file):
        print('Processing file', image_file, '...')
        image = Image.open(image_file)
        origin, tensor_img, resized_img = self._read_image(image)
        input_w, input_h = origin.size[0], origin.size[1]
        image_shape = np.array([input_h, input_w], dtype='int32')
        # print("image shape high:{0}, width:{1}".format(input_h, input_w))
        t1 = time.time()
        batch_outputs = exe.run(inference_program,
                                feed={feed_target_names[0]: tensor_img,
                                      feed_target_names[1]: image_shape[np.newaxis, :]},
                                fetch_list=fetch_targets,
                                return_numpy=False)
        period = (time.time() - t1)*1000
        print("predict cost time:{0}".format("%2.2f ms" % period))
        bboxes = np.array(batch_outputs[0])
        #print(bboxes)
        if bboxes.shape[1] != 6:
            # print("No object found")
            return [img, [], [], [], period]
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')
        return [img, boxes, labels, scores, period]
        
    def _process_results(self, image_file, results):
        img, boxes, labels, scores, period = results
        label_file = self._get_label_file(image_file)
        self.matrix['num_images'] += 1
        
        gt_boxes, gt_labels, gt_scores = list(), list(), list()
        if os.path.isfile(label_file): 
            self.matrix['num_labeled_images'] += 1
            if self.label_format.lower() in ['labelme', 'json']:
                gt_results = self._load_labelme(label_file)
            elif self.label_format.lower() in ['xml', 'voc', 'pascalvoc']
                gt_results = self._load_pascalvoc(label_file)
            self._parse_results(results, gt_boxes, gt_labels, gt_scores)
        
        if self.params['is_write_image']:
            img = self._draw_bbox_image(img, boxes, labels, scores)
            img = self._draw_bbox_image(img, gt_boxes, gt_labels, gt_scores, gt=True)
            self._save_img(img, image_file)
        
    def _init_matrix(self):
        self.label_list = list()
        for label in train_parameters['label_dict']:
            self.label_dict.append(label)
        
        self.matrix = {
            'num_images': 0,
            'num_labeled_images': 0,
            'details': {}
        }
        
        self.element = {
            'num': 0,
            'match': 0,
            'missing': 0,
            'false_alarm': 0
        }
        
    def _parse_results(self, results, gt_results):
        img, boxes, labels, scores, period = results
        gt_boxes, gt_labels, gt_scores = gt_results
        self._register_gt_results(gt_results)
        
        img_w, img_h = img.size
        for box, label in zip(boxes, labels):
            iou = 0
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                if label != gt_label: continue
                cur_iou = self._box_iou(box, gt_box)
                elif cur_iou > self.iou_thresh:
                    pass
    
    def _register_gt_results(self, gt_results)
        gt_boxes, gt_labels, gt_scores = gt_results
        for gt_label in gt_labels:
            text_label = self.label_list[gt_label]
            if text_label not in self.matrix['details']:
                self.matrix['details'][text_label] = self.element.copy()
            self.matrix['details'][text_label]['num'] += 1
            
    def _box_iou(self, box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        inter_x1 = max(xmin1, xmin2)
        inter_x2 = min(xmax1, xmax2)
        inter_y1 = max(ymin1, ymin2)
        inter_y2 = min(ymax1, ymax2)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2: return 0
        inter_area = (inter_x2-inter_x1) * (inter_y2-inter_y1)
        area = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2)
        iou = inter_area / (area - inter_area)
        
        return iou
        
    def _load_image_list(self, image_path):
        image_list = list()
        
        suffices = self.params['supported_images']
        for suffix in suffices:
            image_list += gb.glob(image_path + r'/*'+suffix)
            
        return image_list
        
    def _get_label_file(self, image_path):
        image_path, filename = os.path.split(image_path)
        fname, _ = os.path.splitext(image_path)
        
        if self.label_format.lower() in ['json', 'labelme']:
            label_suffix = '.json'
        elif self.label_format.lower() in ['xml', 'voc', 'pascalvoc']
            label_suffix = '.xml'
        else:
            raise ValueError('Unsupported label type.')
            
        if self.label_dir is not None:
            label_path = self.label_dir
        else:
            label_path = image_path
        label_file = os.path.join(label_path, fname+label_suffix)
        
        return label_file
        
    def _load_labelme(self, label_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            img_h = js_obj['imageHeight']
            img_w = js_obj['imageWidth']
            
            boxes, labels, scores = list(), list(), list()
            for elem in js_obj['shapes']:
                bbox = create_box_from_polygon(elem['points'], img_h, img_w, **self.params['labelme'])
                boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                labels.append(int(train_parameters['label_dict'][elem['label']]))
            f.close()    
        
        return [boxes, labels, scores]
        
    def _load_pascalvoc(self, label_file):
        boxes, labels, scores = list(), list(), list()
        root = xml.etree.ElementTree.parse(ann_file).getroot()
        
        for object in root.findall('object'):
            labels.append(int(self.train_parameters['label_dict'][object.find('name').text]))
            bbox = object.find('bndbox')
            boxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text), 
                int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
                
        return [boxes, labels, scores]
        
    def _read_image(self, img):
        if img.mode != 'RGB': 
            img = img.convert('RGB')
            
        origin = img
        img = self._resize_img(origin, yolo_config["input_size"])
        resized_img = img.copy()
        img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
        img -= 127.5
        img *= 0.007843
        img = img[np.newaxis, :]
        return origin, img, resized_img
        
    def _resize_img(self, img, target_size):
        img = img.resize(target_size[1:], Image.BILINEAR)

        return img
        
    def _draw_bbox_image(self, img, boxes, labels, scores, gt=False):
        if len(boxes)==0 or len(boxes)==0 or len(scores)==0: return img
        
        color = ['red', 'green']
        draw = ImageDraw.Draw(img)
        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            if not gt: c = color[0]
            else: c = color[1]
            draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
            draw.text((xmin, ymin), str(score), (255, 255, 0))
        return img
        
    def _save_img(self, img, image_file):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        _, filename = os.path.split(image_file)
        save_name = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_name, img)
        
    def _save_results(self):
        save_name = os.path.join(self.save_dir, 'results.json')


if __name__ == '__main__':
    params = {
        'supported_images': ['.bmp', '.png', '.jpg', '.tif'],
        'label_format': 'labelme',
        'label_dir': None,
        'save_dir': r'',
        'iou_thresh': 0.5,
        'is_write_image': True,
        'labelme': {
            'min_h': 20,
            'min_w': 20
        }
    }
    
    
    image_path = r'G:\threeguns_data\20210322-20210326_data_collection\single\single_white_vertical_9gain_300mus_19.8rpm_headlight_on'
    #label_path = r'E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO\v1.1\valid'
    save_path  = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.2.0\dataset\valid_322"
    image_list = gb.glob(image_path + r"/*.bmp")
    total_time = 0.
    
    pvoc = PascalVocXmlParser()
    
    for image_file in image_list:
        img = cv2.imread(image_file)
        _, filename = os.path.split(image_file)
        fname, _ = os.path.splitext(filename)
        save_name = os.path.join(save_path, filename)
        #label_file = os.path.join(label_path, fname+".xml")
        
        flag, box, label, scores, bboxes, period = infer(img)
        total_time += period
        
        if flag:
            img = draw_bbox_image(img, box, label, scores)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            print('Defect detected at image', image_file)
            cv2.imwrite(save_name, img)
        else:
            print(image_path, "No defect detected.")
            shutil.copy(image_file, save_name)
        #print('infer one picture cost {} ms'.format(period))
    
    if len(image_list) > 0:
        average_time = total_time / len(image_list)
        fps = int(1000/average_time)
        print("The avergae processing time for one image is", average_time)
        print("The fps is", fps)
        """
    
    # Check the result of a single image
    image_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid\valid_gray_1_1600.png"
    img = cv2.imread(image_file)
    flag, box, label, scores, bboxes, period = infer(img)
    print("The boxes are", box)
    print("The scores are", scores)
    """
