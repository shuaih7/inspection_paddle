import os, sys
import numpy as np
import glob as gb
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from PascalVocParser import PascalVocXmlParser


pvoc = PascalVocXmlParser()

def cluster_anchors(path, k0=3, k1=3):
    dataset0, dataset1 = encode_dataset(path)
    km0 = KMeans(n_clusters=k0, random_state=9).fit(dataset0)
    km1 = KMeans(n_clusters=k1, random_state=9).fit(dataset1)
    ac0 = np.array(km0.cluster_centers_, dtype=np.int32)
    ac1 = np.array(km1.cluster_centers_, dtype=np.int32)
    
    return ac0, ac1

def encode_dataset(path, x_thres=300):
    dataset0 = [] # dataset that x_length larger than x_thres
    dataset1 = [] # dataset that x_length smaller than x_thres
    for xml_file in gb.glob(path + r"/*.xml"): 
        tree = ET.parse(xml_file)
 
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
 
        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin"))
            ymin = int(obj.findtext("bndbox/ymin"))
            xmax = int(obj.findtext("bndbox/xmax"))
            ymax = int(obj.findtext("bndbox/ymax"))
     
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin: print("Warning: xmin = xmax or ymin = ymax occurs at", xml_file)
            if xmax-xmin >= x_thres: dataset0.append([xmax - xmin, ymax - ymin])
            else: dataset1.append([xmax - xmin, ymax - ymin])
    return np.array(dataset0), np.array(dataset1)
    
    
if __name__ == "__main__":
    path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun_YOLO\train"
    ac0, ac1 = cluster_anchors(path, k0=3, k1=3)
    print(ac0)
    print(ac1)
