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
    
    
def infer_rec(img_dir, ocr, label_file=None):
    img_list = image_loader(img_dir)
    
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        result = ocr.ocr(image, rec=True, det=False, cls=False)
        text = result[0][0].upper()
        conf = str(round(result[0][1], 3))
        
        title = text + " - " + conf
        plt.imshow(image, cmap="gray"), plt.title(title)
        plt.show()


if __name__ == "__main__":
    params = {
        "use_gpu": True,
        "gpu_mem": 2000,
        "use_angle_cls": True,
        "lang": "en",
        "cls_model_dir": r"E:\Projects\Part_Number\model\demo\cls",
        "det_model_dir": r"E:\Projects\Part_Number\model\demo\det",
        "rec_model_dir": r"E:\Projects\Part_Number\model\demo\rec\en"
    }
    ocr = PaddleOCR(**params)
    
    img_dir = r"E:\Projects\Part_Number\dataset\rec_valid\20210122"
    infer_rec(img_dir, ocr)
    
    