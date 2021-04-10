#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04.08.2021
Updated on 04.09.2021

Author: haoshuai@handaotech.com
'''

import os
import cv2
import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from .utils import box_to_center_relative


class Augment(object):
    def __init__(self, train_parameters):
        """ Data augmentation for detection projects
        
        Attributes:
            train_parameters: Parameters loading from the configuration file
        
        """
        self.update(train_parameters)
        
    def update(self, train_parameters):
        self.train_parameters = train_parameters
        self.mask_color = (127,127,127)
        self.mask_width = 20
        
    def distort_image(self, img):
        ''' 
        Image random distortion
        '''
        def random_brightness(img):
            prob = np.random.uniform(0, 1)
            if prob < self.train_parameters['image_distort_strategy']['brightness_prob']:
                brightness_delta = self.train_parameters['image_distort_strategy']['brightness_delta']
                delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
                img = ImageEnhance.Brightness(img).enhance(delta)
            return img

        def random_contrast(img):
            prob = np.random.uniform(0, 1)
            if prob < self.train_parameters['image_distort_strategy']['contrast_prob']:
                contrast_delta = self.train_parameters['image_distort_strategy']['contrast_delta']
                delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
                img = ImageEnhance.Contrast(img).enhance(delta)
            return img
        
        def random_saturation(img):
            prob = np.random.uniform(0, 1)
            if prob < self.train_parameters['image_distort_strategy']['saturation_prob']:
                saturation_delta = self.train_parameters['image_distort_strategy']['saturation_delta']
                delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
                img = ImageEnhance.Color(img).enhance(delta)
            return img

            prob = np.random.uniform(0, 1)
            if prob < self.train_parameters['image_distort_strategy']['hue_prob']:
                hue_delta = self.train_parameters['image_distort_strategy']['hue_delta']
                delta = np.random.uniform(-hue_delta, hue_delta)
                img_hsv = np.array(img.convert('HSV'))
                img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
                img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
            return img
            
        ops = [random_brightness, random_contrast, random_saturation, random_hue]    
        np.random.shuffle(ops) 
        img = ops[0](img)
        img = ops[1](img)
        #img = ops[2](img)
        #img = ops[3](img)
        return img
        
    def random_expand(self, img, gtboxes, keep_ratio=True):
        if np.random.uniform(0, 1) > self.train_parameters['image_distort_strategy']['expand_prob']:
            return img, gtboxes

        max_ratio = self.train_parameters['image_distort_strategy']['expand_max_ratio']
        w, h = img.size
        c = 3
        ratio_x = random.uniform(1, max_ratio)
        if keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, max_ratio)
            
        oh = int(h * ratio_y)
        ow = int(w * ratio_x)
        off_x = random.randint(0, ow -w)
        off_y = random.randint(0, oh -h)

        out_img = np.zeros((oh, ow, c), np.uint8)
        for i in range(c):
            out_img[:, :, i] = self.train_parameters['mean_rgb'][i]

        out_img[off_y: off_y + h, off_x: off_x + w, :] = img
        gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
        gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
        gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
        gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

        return Image.fromarray(out_img), gtboxes
        
    def random_flip_left_right(self, img, gtboxes):
        if np.random.uniform(0, 1) > self.train_parameters['image_distort_strategy']['flip_lr_prob']:
            return img, gtboxes

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
        
        return img, gtboxes
        
        
    def random_flip_top_bottom(self, img, gtboxes):
        if np.random.uniform(0, 1) > self.train_parameters['image_distort_strategy']['flip_lr_prob']:
            return img, gtboxes
            
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        gtboxes[:, 1] = 1.0 - gtboxes[:, 1]
            
        return img, gtboxes
        
    def random_interp(self, img, size, interp=None):
        interp_method = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        img = np.asarray(img)
        if not interp or interp not in interp_method:
            interp = interp_method[random.randint(0, len(interp_method) - 1)]
        h, w, _ = img.shape
        im_scale_x = size[2] / float(w)
        im_scale_y = size[1] / float(h)
        img = cv2.resize(
            img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
        img = Image.fromarray(img)
        return img

    def shuffle_gtbox(self, gtbox, gtlabel):
        gt = np.concatenate(
            [gtbox, gtlabel[:, np.newaxis]], axis=1)

        idx = np.arange(gt.shape[0])
        np.random.shuffle(idx)
        gt = gt[idx, :]

        return gt[:, :4], gt[:, 4]
        
    def random_mask(self, img, boxes, labels, diffs, lines):
        # Only support labelme format 
        # Mask sure to do this augmentation before the random rotate
        # Because currently the random rotate will not rotate the lines
        # Will remain at lease one defect label
        if self.train_parameters['label_format'] not in ['labelme', 'json']:
            return img, boxes, labels, diffs
        elif np.random.uniform(0, 1) > self.train_parameters['image_distort_strategy']['mask_prob']:
            return img, boxes, labels, diffs
        elif len(boxes) == 1:
            return img, boxes, labels, diffs
            
        max_mask_num = min(self.train_parameters['image_distort_strategy']['max_mask_num'], len(boxes)-1)
        mask_num = random.randint(1, max_mask_num)
        index_list = random.sample(range(0, len(boxes)), mask_num)
        index_list = sorted(index_list, reverse=True)
        draw = ImageDraw.Draw(img)
        
        for index in index_list:
            pts = self._arrange_points(lines[index])
            draw.line(pts, fill=self.mask_color, width=self.mask_width)
            boxes = np.delete(boxes, index, axis=0)
            labels = np.delete(labels, index)
            diffs = np.delete(diffs, index)
            # lines.pop(index)
            
        return img, boxes, labels, diffs
        
    def random_rotate(self, img, boxes):
        # Please do this augmentation after the random mask
        if np.random.uniform(0, 1) > self.train_parameters['image_distort_strategy']['rotate_prob']:
            return img, boxes
            
        angle_delta = min(self.train_parameters['image_distort_strategy']['rotate_delta'], 8)
        angle = np.random.uniform(-1*angle_delta, angle_delta)
        
        img_w, img_h = img.size
        img = img.rotate(angle, fillcolor=self.mask_color)
        boxes = self._rotate_boxes(boxes, img_h, img_w, angle)
        
        return img, boxes
        
    def random_crop(self, img, boxes, labels, 
                    scales=[0.8, 1.0], 
                    max_ratio=1.25, 
                    constraints=None, 
                    max_trial=50):
                    
        if random.random() > 0.6:
            return img, boxes, labels
        if len(boxes) == 0:
            return img, boxes, labels

        if not constraints:
            constraints = [
                    (0.1, 1.0),
                    (0.3, 1.0),
                    (0.5, 1.0),
                    (0.7, 1.0),
                    (0.9, 1.0),
                    (0.0, 1.0)]

        w, h = img.size
        crops = [(0, 0, w, h)]
        for min_iou, max_iou in constraints:
            for _ in range(max_trial):
                scale = random.uniform(scales[0], scales[1])
                aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                              min(max_ratio, 1 / scale / scale))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                crop_box = np.array([[
                    (crop_x + crop_w / 2.0) / w,
                    (crop_y + crop_h / 2.0) / h,
                    crop_w / float(w),
                    crop_h /float(h)
                    ]])

                iou = self._box_iou_xywh(crop_box, boxes)
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_w, crop_h))
                    break

        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, box_num = self._box_crop(boxes, labels, crop, (w, h))
            if box_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                            crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
            return img, crop_boxes, crop_labels
        return img, boxes, labels
        
        
    def _box_crop(self, boxes, labels, crop, img_shape):
        x, y, w, h = map(float, crop)
        im_w, im_h = map(float, img_shape)

        boxes = boxes.copy()
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

        crop_box = np.array([x, y, x + w, y + h])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

        boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
        boxes[:, :2] -= crop_box[:2]
        boxes[:, 2:] -= crop_box[:2]

        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
        boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
        labels = labels * mask.astype('float32')
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

        return boxes, labels, mask.sum()
        
    def _box_iou_xywh(self, box1, box2):
        assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
        assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        inter_w = inter_x2 - inter_x1 + 1
        inter_h = inter_y2 - inter_y1 + 1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0

        inter_area = inter_w * inter_h
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        return inter_area / (b1_area + b2_area - inter_area)
        
    def _rotate_boxes(self, boxes, img_h, img_w, angle):
        ''' Rotate the normalizd boxes '''
        rot_boxes = []
        for box in boxes:
            x, y, w, h = box
            # Map back the bounding box
            xmin = max(0, int((x-w/2)*img_w))
            xmax = min(img_w, int((x+w/2)*img_w))
            ymin = max(0, int((y-h/2)*img_h))
            ymax = min(img_h, int((y+h/2)*img_h))
            
            # Initialize the points
            xs, ys, pts_list = list(), list(), list()
            pts_list.append([xmin, ymin])
            pts_list.append([xmax, ymin])
            pts_list.append([xmax, ymax])
            pts_list.append([xmin, ymax])
            
            # Mapping the points
            for pt in pts_list:
                xr, yr = self._rotate_point(pt, img_h, img_w, angle)
                xs.append(xr)
                ys.append(yr)
            xmin = max(0, min(xs))
            xmax = min(img_w, max(xs))
            ymin = max(0, min(ys))
            ymax = min(img_h, max(ys))
            
            rot_box = box_to_center_relative([xmin, ymin, xmax-xmin, ymax-ymin], img_h, img_w)
            rot_boxes.append(rot_box)
        
        return rot_boxes
        
    def _rotate_point(self, point, img_h, img_w, angle):
        x1 = point[0] - img_w / 2
        y1 = img_h / 2 - point[1]
        angle = angle / 180 * math.pi
        x2 = x1*math.cos(angle) - y1*math.sin(angle)
        y2 = x1*math.sin(angle) + y1*math.cos(angle)
        
        rot_point = [x2+img_w/2, img_h/2-y2]
        
        return rot_point
        
    def _arrange_points(self, points):
        pts = list()
        for point in points:
            pt = (point[0], point[1])
            pts.append(pt)
        return pts
            
        
        


