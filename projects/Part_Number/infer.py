import os
import sys
sys.path.append(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Part_Number\PaddleOCR-release-1.1")

import cv2
import numpy as np
from paddleocr import PaddleOCR
from utils import *
from matplotlib import pyplot as plt


ocr = PaddleOCR(use_angle_cls=True, lang="en")


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
    

def infer_rec(data_dir, label_file):
    with open(label_file, "rb") as fin:
        label_infor_list = fin.readlines()
            
        for label_infor in label_infor_list:
            image, points, texts = load_data(data_dir, label_infor)
            
            for pt, txt in zip(points, texts):
                dst_img = get_rotate_crop_image(image, np.array(pt, dtype=np.float32))
                if txt == "O":  
                    dst_img = flip_image_left_right(dst_img)
                elif txt == "A":
                    dst_img = flip_image_top_bottom(dst_img)
                    
                result = ocr.ocr(dst_img, rec=True, det=False, cls=False) # List of ["result", confidence]
                
                img_show = image.copy()
                img_show = draw_polylines(img_show, [pt])
                plt.subplot(1,2,1), plt.imshow(img_show), plt.title("Input image")
                plt.subplot(1,2,2), plt.imshow(dst_img), plt.title(result[0][0]+" - "+str(round(result[0][1], 3)))
                plt.show()
                img_show = None
        
        fin.close()

if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset\20210113"
    label_file = r"E:\Projects\Part_Number\dataset\20210113\machining_shank\Label.txt"
    
    infer_rec(data_dir, label_file)
    