#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2021
Updated on 04.09.2021

Author: haoshuai@handaotech.com
'''

import os
import cv2
import random
import paddle
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

import config
from data import Augment, PascalVocParser, LabelmeParser

train_parameters = config.init_train_parameters()
aug = Augment(train_parameters)

if train_parameters['label_format'].lower() in ['voc', 'pascalvoc', 'labelimg', 'xml']:
    parser = PascalVocParser(train_parameters)
elif train_parameters['label_format'].lower() in ['labelme', 'json']:
    parser = LabelmeParser(train_parameters)
else:
    raise ValueError('Invalid label format.')


def draw_bbox_image(img, boxes, im_width, im_height, gt=False):
    color = ['red', 'blue']
    if gt:
        c = color[1]
    else:
        c = color[0]
        
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        xmin = max(0, int((x-w/2)*im_width))
        xmax = min(im_width, int((x+w/2)*im_width))
        ymin = max(0, int((y-h/2)*im_height))
        ymax = min(im_height, int((y+h/2)*im_height))
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
    return img
    
    
def preprocess(img, bbox_labels, input_size, mode):
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    gtlabels = sample_labels[:, 0]
    gtboxes = sample_labels[:, 1:5]
    
    if mode == 'train':
        if train_parameters['apply_distort']:
            img = aug.distort_image(img)
        img, gtboxes = aug.random_expand(img, gtboxes)
        img, gtboxes, gtlabels = aug.random_crop(img, gtboxes, gtlabels)
        img, gtboxes = aug.random_flip_left_right(img, gtboxes)
        img, gtboxes = aug.random_flip_top_bottom(img, gtboxes)
        gtboxes, gtlabels = aug.shuffle_gtbox(gtboxes, gtlabels)
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
    # img = resize_img(img, sample_labels, input_size)
    img = random_interp(img, input_size)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


def custom_reader(file_list, data_dir,input_size, mode):

    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                
                fname = line.replace("\n","")
                image_path = os.path.join(data_dir, fname+".bmp")
                label_path = os.path.join(data_dir, fname+".xml")
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # layout: label | xmin | ymin | xmax | ymax | difficult
                bbox_labels = parser(label_path)
                if len(bbox_labels) == 0:
                    continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
                #img, sample_labels = custom_preprocess(img, bbox_labels, input_size, mode)
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_parameters['max_box_num']
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]
                
                yield img, ret_boxes, ret_lbls, ret_difficults
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader


def single_custom_reader(file_path, data_dir, input_size, mode):
    """
    单线程数据读取
    :param file_path:
    :param data_dir:
    :param input_size:
    :param mode:
    :return:
    """
    #file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    reader = custom_reader(images, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size'])
    reader = paddle.batch(reader, train_parameters['train_batch_size'])
    return reader
    

def preprocess_test(image_path):
    data_dir, filename = os.path.split(image_path)
    fname, _ = os.path.splitext(filename)
    label_path = os.path.join(data_dir, fname+".json")
    if not os.path.isfile(image_path):
        raise NameError("Could not find image file", image_path)
    if not os.path.isfile(label_path):
        raise NameError("Could not find label file", label_path)
    
    img = Image.open(image_path)    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    im_width, im_height = img.size # Input image size (720x540)
    
    bbox_labels = parser(label_path)
    if len(bbox_labels) == 0:
        pass # Do something ...
        
    # Code from function "preprocess"
    sample_labels = np.array(bbox_labels)
    gtlabels = sample_labels[:, 0]
    gtboxes = sample_labels[:, 1:5]
    #if train_parameters['apply_distort']:
    #    img = distort_image(img)
    #img, gtboxes = aug.random_expand(img, gtboxes)
    img0 = img.copy()
    gtboxes0 = gtboxes.copy()
    gtlabels0 = gtlabels.copy()
    #img0, gtboxes0, gtlabels = aug.random_crop(img0, gtboxes0, gtlabels0)
    img0, gtboxes0 = aug.random_rotate(img0, gtboxes0)
    #img0, gtboxes0 = aug.random_flip_top_bottom(img0, gtboxes0)
    
    draw_img = draw_bbox_image(img, gtboxes, img.size[0], img.size[1])
    draw_img0 = draw_bbox_image(img0, gtboxes0, img.size[0], img.size[1])
    plt.subplot(1,2,1), plt.imshow(draw_img)
    plt.subplot(1,2,2), plt.imshow(draw_img0)
    plt.show()

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    image_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\label_test\MER2-041-436U3M(FDL21010006)_2021-03-25_15_36_37_717-0.bmp"
    preprocess_test(image_path)
    '''
    img = Image.open(image_path)
    img_r = img.rotate(10, fillcolor=127)
    plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1,2,2), plt.imshow(img_r, cmap='gray'), plt.title('Rotated')
    plt.show()
    '''
    
    