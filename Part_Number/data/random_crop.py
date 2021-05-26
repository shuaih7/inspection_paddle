import os
import sys

file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.append(main_path)

import cv2
import math
import glob as gb
import numpy as np
from utils import draw_polylines
from matplotlib import pyplot as plt


TOP = 400
OFFY = 45

LEFT = 342
OFFX = 35

WIDTH = 683
HEIGHT = 905
OFFW = 50
OFFH = 75


def parse_labels(labels):
    texts, points = [], []
    
    for label in labels:
        texts.append(label['transcription'])
        points.append(label['points'])
        
    return [points, texts]
    
    
def get_centers(points):
    centers = list()
    
    for pts in points:
        x, y = 0, 0
        
        for pt in pts:
            x += pt[0]
            y += pt[1]
        centers.append([x/max(1,len(pts)), y/max(1,len(pts))])
        
    return centers
    
    
def pair_labels(points, texts, centers, _pairs=None, _texts=None):
    if len(points)%2 > 0:
        raise ValueError("The input data length must be even.")
    elif len(points) != len(centers) or len(points) != len(texts):
        raise ValueError("Points length and center length or texts length do not match.")
    elif len(points) == 0:
        raise ValueError("Empty input array.")
      
    if _pairs is None: _pairs = []    
    if _texts is None: _texts = []    
    
    if len(points) == 2:
        _pairs.append([points[0], points[1]])
        _texts.append([texts[0], texts[1]])
        return _pairs, _texts
    else:
        anchor = centers[0]
        index, dist = 0, 999999
        
        for i in range(1, len(centers)):
            center = centers[i]
            cur_dist = math.sqrt((center[0]-anchor[0])**2 + (center[1]-anchor[1])**2)
            if cur_dist < dist: 
                index = i
                dist = cur_dist
        
        _pairs.append([points[0], points[index]])
        _texts.append([texts[0], texts[index]])
        points.pop(index)
        points.pop(0)
        texts.pop(index)
        texts.pop(0)
        centers.pop(index)
        centers.pop(0)
        return pair_labels(points, texts, centers, _pairs, _texts)
        
   
def get_label_rois(pairs):
    rois = []
    
    for pair in pairs:
        xs, ys = [], []
        
        for pt in pair[0]+pair[1]: 
            xs.append(pt[0])
            ys.append(pt[1])
            
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        
        # Clock-wise         
        rois.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    
    return rois
    

def random_crop_pairs(data_dir, 
                      label_file, 
                      train_save_path="", 
                      valid_save_path="", 
                      ratio=0.8):
                      
    train_label_file = os.path.join(train_save_path, "label.txt")
    valid_label_file = os.path.join(valid_save_path, "label.txt")
    
    with open(label_file, "rb") as fin:
        data_lines = fin.readlines()
        
        train_labels = []
        valid_labels = []
        
        for data_line in data_lines:
            substr = data_line.decode('utf-8').strip("\n").split("\t")
            
            img_path = os.path.join(data_dir, substr[0])
            print("Processing image file", img_path, "...")
            _, filename = os.path.split(img_path)
            fname, suffix = os.path.splitext(filename)
            image = cv2.imread(img_path, -1)
            labels = eval(substr[1].replace("false", "False"))   
            
            points, texts = parse_labels(labels)
            centers = get_centers(points)
            pairs, texts = pair_labels(points, texts, centers)
            rois = get_label_rois(pairs)
            
            save_id = 0
            img_h, img_w = image.shape[:2]
            for pair, text_pair, roi in zip(pairs, texts, rois):
                center_x = int((roi[0][0]+roi[1][0])/2)
                center_y = int((roi[0][1]+roi[2][1])/2)
                
                offx = np.random.randint(-OFFX, OFFX)
                offy = np.random.randint(-OFFY, OFFY)
                offw = np.random.randint(-OFFW, OFFW)
                offh = np.random.randint(-OFFH, OFFH)
                
                xmin = min(int(roi[0][0]), max(0, center_x - LEFT + offx))
                xmax = max(int(roi[1][0]), min(img_w, xmin + WIDTH + offw))
                ymin = min(int(roi[0][1]), max(0, center_y - TOP + offy))
                ymax = max(int(roi[2][1]), min(img_h, ymin + HEIGHT + offh))
                
                label = []
                img = image[ymin:ymax, xmin:xmax]
                for points, txt in zip(pair, text_pair):
                    for pt in points:
                        pt[0] -= xmin
                        pt[1] -= ymin
                    label.append({"transcription": txt, "points": points})
            
                img_name = fname+"_"+str(save_id)+suffix
                if np.random.rand() < ratio:
                    _, prefix = os.path.split(train_save_path)
                    img_save_name = os.path.join(train_save_path, img_name)
                    item = os.path.join(prefix, img_name) + "\t" + str(label) + "\n"
                    train_labels.append(item.replace("False", "false"))
                else:
                    _, prefix = os.path.split(valid_save_path)
                    img_save_name = os.path.join(valid_save_path, img_name)
                    item = os.path.join(prefix, img_name) + "\t" + str(label) + "\n"
                    valid_labels.append(item.replace("False", "false"))
                    
                cv2.imwrite(img_save_name, img)     
                save_id += 1
            
        fin.close()
        
        with open(train_label_file, "w") as fout:
            for item in train_labels:
                fout.write(item)
            fout.close()
            
        with open(valid_label_file, "w") as fout:
            for item in valid_labels:
                fout.write(item)
            fout.close()
        
       
if __name__ == "__main__":
    ratio = 0.85
    data_dir = r"E:\Projects\Part_Number\dataset"
    label_file = r"E:\Projects\Part_Number\dataset\20210122\label.txt"
    train_save_path = r"E:\Projects\Part_Number\dataset\det_train\20210122"
    valid_save_path = r"E:\Projects\Part_Number\dataset\det_valid\20210122"
    random_crop_pairs(data_dir, label_file, train_save_path, valid_save_path, ratio=ratio)