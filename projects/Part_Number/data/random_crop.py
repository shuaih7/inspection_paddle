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
    
    
def pair_labels(points, centers, _pairs=None):
    if len(points)%2 > 0:
        raise ValueError("The input data length must be even.")
    elif len(points) != len(centers):
        raise ValueError("Points length and center length do not match.")
    elif len(points) == 0:
        raise ValueError("Empty input array.")
      
    if _pairs is None: _pairs = []     
    if len(points) == 2:
        _pairs.append([points[0], points[1]])
        return _pairs.copy()
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
        points.pop(index)
        points.pop(0)
        centers.pop(index)
        centers.pop(0)
        return pair_labels(points, centers, _pairs)
        
   
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
    

def random_crop_pairs(data_dir, label_file):
    with open(label_file, "rb") as fin:
        data_lines = fin.readlines()
        
        for data_line in data_lines:
            substr = data_line.decode('utf-8').strip("\n").split("\t")
            
            img_path = os.path.join(data_dir, substr[0])
            # if img_path != r"E:\Projects\Part_Number\dataset\train\20210113/2021-01-13_12_54_46_257.png": continue
            image = cv2.imread(img_path, -1)
            labels = eval(substr[1].replace("false", "False"))   
            
            points, texts = parse_labels(labels)
            centers = get_centers(points)
            pairs = pair_labels(points, centers)
            
            rois = get_label_rois(pairs)
            image = draw_polylines(image, rois, color=255)
            plt.imshow(image, cmap="gray"), plt.show()
            
        fin.close()
        
       
if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset\train"
    label_file = r"E:\Projects\Part_Number\dataset\train\20210113\Label.txt"
    random_crop_pairs(data_dir, label_file)