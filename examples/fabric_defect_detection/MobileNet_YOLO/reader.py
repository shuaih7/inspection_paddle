# -*- coding: UTF-8 -*-
"""
reader
"""
import numpy as np
import config
import random
import paddle
import os
from PIL import Image, ImageEnhance
import xml
import cv2

train_parameters = config.init_train_parameters()


"""
Utility functions to rotate the gtboxes
"""
def rotate_gtboxes(gtboxes, w=1.0, h=1.0, d="+"): # + means counter-clockwise
    nboxes = []
    for gtbox in gtboxes:
        x, y, width, height = gtbox
        x, y = get_rot90_pos(x, y, w=w, h=h, d=d)
        nboxes.append([x, y, height, width])
    
    return nboxes


"""
Utility functions to rotate the position
"""
def get_rot90_pos(x, y, w=1.0, h=1.0, d="+"): # + means counter-clockwise
    if d == "+":
        nx = y
        ny = h - x
        return nx, ny
    elif d == "-":
        nx = w - y
        ny = x
        return nx, ny
    else: raise ValueError("Invalid d value.")


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


def resize_img(img, sampled_labels, input_size):
    """
    缩放图像
    :param img:
    :param sampled_labels:
    :param input_size:
    :return:
    """
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def box_iou_xywh(box1, box2):
    """
    计算iou
    :param box1:
    :param box2:
    :return:
    """
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


def box_crop(boxes, labels, crop, img_shape):
    """
    box crop
    :param boxes:
    :param labels:
    :param crop:
    :param img_shape:
    :return:
    """
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


def distort_image(img):
    """
    图像增强
    :param img:
    :return:
    """
    
    def random_brightness(img):
        """
        随机亮度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['brightness_prob']:
            brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
            delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        return img

    def random_contrast(img):
        """
        随机对比度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['contrast_prob']:
            contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
            delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        return img
    
    def random_saturation(img):
        """
        随机饱和度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['saturation_prob']:
            saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        return img

    def random_hue(img):
        """
        随机色调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['hue_prob']:
            hue_delta = train_parameters['image_distort_strategy']['hue_delta']
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


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    """
    随机裁剪
    :param img:
    :param boxes:
    :param labels:
    :param scales:
    :param max_ratio:
    :param constraints:
    :param max_trial:
    :return:
    """
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

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels


def random_expand(img, gtboxes, keep_ratio=True):
    """
    随机扩张
    :param img:
    :param gtboxes:
    :param keep_ratio:
    :return:
    """
    if np.random.uniform(0, 1) < train_parameters['image_distort_strategy']['expand_prob']:
        return img, gtboxes

    max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']
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
        out_img[:, :, i] = train_parameters['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes


def random_flip(img, gtboxes, thresh=0.5):
    """
    随机翻转
    :param img:
    :param gtboxes:
    :param thresh:
    :return:
    """
    if random.random() > thresh:
        # img = img[:, ::-1, :]
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes
    
    
def random_rotate(img, gtboxes, thresh=0.5):
    """
    随机旋转
    :param img:
    :param gtboxes:
    :param thresh:
    :return:
    """
    if random.random() > thresh:
        value = random.random()
        
        if value > 0.5: # Counter-clockwise rotation
            img = img.rotate(90)
            gtboxes = rotate_gtboxes(gtboxes, d="+")
        else: # Clockwise rotation
            img = img.rotate(-90)
            gtboxes = rotate_gtboxes(gtboxes, d="-")
            
    return img, gtboxes       

def random_interp(img, size, interp=None):
    """
    随机差值
    :param img:
    :param size:
    :param interp:
    :return:
    """
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


def shuffle_gtbox(gtbox, gtlabel):
    """
    shuffle gt
    :param gtbox:
    :param gtlabel:
    :return:
    """
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)

    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]

    return gt[:, :4], gt[:, 4]


def custom_preprocess(img, bbox_labels, input_size, mode):
    """
    preprocess
    :param img:
    :param bbox_labels:
    :param input_size:
    :param mode:
    :return:
    """
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        if train_parameters['apply_distort']: 
            img = distort_image(img)
            
        gtboxes = sample_labels[:, 1:5] # Define the ground true bounding box
        gtlabels = sample_labels[:, 0]  # Define the ground true labels
        
        img, gtboxes = random_flip(img, gtboxes, thresh=0.5)
        img, gtboxes = random_rotate(img, gtboxes, thresh=0.5)
        gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)
        
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
        
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels
    
    
def preprocess(img, bbox_labels, input_size, mode):
    """
    preprocess
    :param img:
    :param bbox_labels:
    :param input_size:
    :param mode:
    :return:
    """
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        if train_parameters['apply_distort']:
            img = distort_image(img)
        img, gtboxes = random_expand(img, sample_labels[:, 1:5])
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])
        img, gtboxes = random_flip(img, gtboxes, thresh=0.5)
        gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)
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
    """
    定义reader
    :param file_list:
    :param data_dir:
    :param input_size:
    :param mode:
    :return:
    """
    def reader():
        """
        reader实现
        :return:
        """
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                
                fname = line.replace("\n","")
                image_path = os.path.join(data_dir, fname+".png")
                label_path = os.path.join(data_dir, fname+".xml")
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # layout: label | xmin | ymin | xmax | ymax | difficult
                bbox_labels = []
                root = xml.etree.ElementTree.parse(label_path).getroot()
                for object in root.findall('object'):
                    bbox_sample = []
                    # start from 1
                    bbox_sample.append(float(train_parameters['label_dict'][object.find('name').text]))
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
                if len(bbox_labels) == 0:
                    continue
                #img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
                img, sample_labels = custom_preprocess(img, bbox_labels, input_size, mode)
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


# def multi_process_custom_reader(file_path, data_dir, num_workers, mode):
#     file_path = os.path.join(data_dir, file_path)
#     readers = []
#     images = [line.strip() for line in open(file_path)]
#     n = int(math.ceil(len(images) // num_workers))
#     image_lists = [images[i: i + n] for i in range(0, len(images), n)]
#     for l in image_lists:
#         readers.append(paddle.batch(custom_reader(l, data_dir, mode),
#                                           batch_size=train_parameters['train_batch_size'],
#                                           drop_last=True))
#     return paddle.reader.multiprocess_reader(readers, False)
#
#
# def create_eval_reader(file_path, data_dir, input_size, mode):
#     file_path = os.path.join(data_dir, file_path)
#     images = [line.strip() for line in open(file_path)]
#     return paddle.batch(custom_reader(images, data_dir, input_size, mode),
#                                     batch_size=train_parameters['train_batch_size'],
#                                     drop_last=True)
                                    

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