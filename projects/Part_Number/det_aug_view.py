import os
import sys
sys.path.append(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Part_Number\PaddleOCR-release-1.1")

import json
import cv2
from matplotlib import pyplot as plt
from ppocr.data.det.db_process import DBProcessTrain
from utils import draw_polylines


def show_det_augment(img_set_dir, 
                     image_shape, 
                     label_file,
                     is_aug=False,
                     is_crop=False,
                     is_shrink=False,
                     is_border=False):
    params = {}
    params['img_set_dir'] = img_set_dir
    params['image_shape'] = image_shape
    
    
    with open(label_file, "rb") as fin:
        label_infor_list = fin.readlines()
    
    dataset = DBProcessTrain(params)
    for label_infor in label_infor_list:
        data = dataset.get_data(label_infor, is_aug, is_crop, is_shrink, is_border)
        image = data['image']
        polys = data['polys']
        #texts = data['texts']
        image = draw_polylines(image, polys)
        
        label_infor = label_infor.decode()
        label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
        substr = label_infor.strip("\n").split("\t")
        img_file = os.path.join(params['img_set_dir'], substr[0])
        ori_img = cv2.imread(img_file, -1)
        #labels = json.loads(substr[1])
        labels = eval(substr[1])
        boxes = []
        for label in labels:
            boxes.append(label['points'])
        ori_img = draw_polylines(ori_img, boxes)
        
        plt.subplot(1,2,1), plt.imshow(ori_img), plt.title("Original image")
        plt.subplot(1,2,2), plt.imshow(image), plt.title("Augment result")
        plt.show()
        
     
if __name__ == "__main__":
    img_set_dir = r"E:\Projects\Part_Number\dataset"
    image_shape = [3, 640, 640]
    label_file = r"E:\Projects\Part_Number\dataset\20210112\train.txt"
    show_det_augment(img_set_dir, image_shape, label_file, 
                     is_crop=True)