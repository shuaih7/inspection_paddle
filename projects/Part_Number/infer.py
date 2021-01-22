import os
import sys
sys.path.append(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Part_Number\PaddleOCR-release-1.1")

import cv2
import glob as gb
import numpy as np
from paddleocr import PaddleOCR
from utils import *
from matplotlib import pyplot as plt


def image_loader(img_dir):
    suffixs = [".bmp", ".png", ".jpg", ".tif"]
    img_list = []
    
    for suf in suffixs:
        img_list += gb.glob(img_dir + r"/*"+suf)
    return img_list


def parse_labels(labels):
    texts, points = [], []
    
    for label in labels:
        texts.append(label['transcription'])
        points.append(label['points'])
        
    return points, texts


def load_data(data_dir, label_infor):
    label_infor = label_infor.decode()
    label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
    substr = label_infor.strip("\n").split("\t")
    
    img_path = os.path.join(data_dir, substr[0])
    labels = eval(substr[1].replace("false", "False"))
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None: 
        raise Exception("Could not find image", img_path)
    points, texts = parse_labels(labels)
            
    return image, points, texts
    
    
def infer_det(img_dir, ocr, label_file=None):
    img_list = image_loader(img_dir)
    
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        result = ocr.ocr(image, rec=False, det=True, cls=False)
        print(result)
        sys.exit()


if __name__ == "__main__":
    params = {
        "use_angle_cls": True,
        "lang": "en",
        "cls_model_dir": None,
        "det_model_dir": r"E:\Projects\Part_Number\model\det\saved_model\best_accuracy",
        "rec_model_dir": None
    }
    ocr = PaddleOCR(**params)
    
    img_dir = r"E:\Projects\Part_Number\dataset\valid\20210113"
    infer_det(img_dir, ocr)
    
    