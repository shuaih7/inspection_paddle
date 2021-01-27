import os
import sys
import cv2
import json
import copy
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


def process(image, json_file, label):
    img_patches = []
    with open (json_file, "r") as f: 
        json_obj = json.load(f)
    points = json_obj["shapes"][0]["points"]
    label = copy.deepcopy(label[:len(label)-1]) # todo ...
    
    x0 = 0
    for pt in points:
        x = int(pt[0])
        img_patches.append(image[:,x0:x])
        x0 = x
    img_patches.append(image[:,x0:])
    img_patches, label = shuffle(img_patches, list(label))
    
    ann_concat = label[0]
    img_concat = img_patches[0]
    for i in range(1, len(img_patches), 1):
        img_concat = np.concatenate((img_concat, img_patches[i]), axis=1)
        ann_concat += label[i]
    
    return img_concat, ann_concat


def random_concat(data_dir, label_file):
    with open(label_file, "rb") as fin:
        label_infor_list = fin.readlines()
        
        for label_infor in label_infor_list:
            label_infor = label_infor.decode()
            label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
            
            substr = label_infor.strip("\n").split("\t")
            img_path = os.path.join(data_dir, substr[0])
            prefix, _ = os.path.splitext(img_path)
            json_path = prefix + ".json"
            if not os.path.isfile(json_path): continue
            print("Processing image file", img_path, "...")
            image = cv2.imread(img_path, -1)
            
            img_concat, ann_concat = process(image, json_path, substr[1])
            plt.imshow(img_concat, cmap="gray"), plt.title(ann_concat), plt.show()
            
        fin.close()
        
        
if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset\rec_train"
    label_file = r"E:\Projects\Part_Number\dataset\rec_train\20210122\label.txt"
    random_concat(data_dir, label_file)
    