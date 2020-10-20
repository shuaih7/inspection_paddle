import os
import numpy as np
from PIL import Image
from .PascalVocParser import PascalVocXmlParser


def get_reader(img_path, ann_path, txt_file, max_box_num=5, mode="train"):
    pvoc = PascalVocXmlParser()
    def reader():
        f = open(txt_file, "r")
        lines = f.readlines()
        
        for l in lines:
            fname = l.replace("\n", "")
            img_file = os.path.join(img_path, fname+".png")
            ann_file = os.path.join(ann_path, fname+".xml")
            
            # Load the training image data
            image = np.array(Image.open(img_file), dtype=np.float32)
            image = np.expand_dims(np.squeeze(image), 0) # Reshape the image into [c, h, w]
            image /= 255.0  # Normalization
            
            # Load the training annotation data
            boxes, labels = [], []
            img_h, img_w = image.shape[:2]
            bbxs = pvoc.get_boxes(ann_file).astype(np.float32)
            
            for bbx in bbxs:
                x = ((bbx[2]+bbx[0])/2)/img_w
                y = ((bbx[3]+bbx[1])/2)/img_h
                w = (bbx[2]-bbx[0])/img_w
                h = (bbx[3]-bbx[1])/img_h
                boxes.append([x,y,w,h])
                labels.append(1)
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
                
            cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
            ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
            ret_lbls = np.zeros((max_box_num), dtype=np.int32)   
            ret_boxes[0: cope_size] = boxes[0: cope_size]
            ret_lbls[0: cope_size] = labels[0: cope_size]

            yield image, ret_boxes, ret_lbls
            
    return reader


def preprocess(img, bbox_labels, input_size, mode):
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        if train_parameters['apply_distort']:
            img = distort_image(img)
        img, gtboxes = random_expand(img, sample_labels[:, 1:5])
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
    img = resize_img(img, sample_labels, input_size)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


def custom_reader(file_list, data_dir, input_size, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                ######################  以下可能是需要自定义修改的部分   ############################
                parts = line.split('\t')
                image_path = parts[0]
                img = Image.open(os.path.join(data_dir, image_path))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # bbox 的列表，每一个元素为这样
                # layout: label | x-center | y-cneter | width | height | difficult
                bbox_labels = []
                for object_str in parts[1:]:
                    if len(object_str) <= 1:
                        continue
                    bbox_sample = []
                    object = json.loads(object_str)
                    bbox_sample.append(float(train_parameters['label_dict'][object['value']]))
                    bbox = object['coordinate']
                    box = [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                    bbox = box_to_center_relative(box, im_height, im_width)
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    difficult = float(0)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                ######################  可能需要自定义修改部分结束   ############################
                if len(bbox_labels) == 0: continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
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
                yield img, ret_boxes, ret_lbls
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader