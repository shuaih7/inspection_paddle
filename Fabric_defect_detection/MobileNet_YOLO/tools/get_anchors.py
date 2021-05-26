import os, sys
import numpy as np
import glob as gb
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from PascalVocParser import PascalVocXmlParser


pvoc = PascalVocXmlParser()


def show_histogram(path):
    dataset = encode_dataset(path)
    widths = dataset[:,0]
    heights = dataset[:,1]
    plt.subplot(1,2,1), plt.hist(widths), plt.title("Width Distribution")
    plt.subplot(1,2,2), plt.hist(heights), plt.title("Heights Distribution")
    plt.show()
    
    
def get_xml_list(data_path, txt_file):
    f = open(txt_file, "r")
    lines = f.readlines()
    xml_list = []
    
    for item in lines:
        item = item.replace('\n', '') + '.xml'
        xml_list.append(os.path.join(data_path, item))
        
    return xml_list
    

def cluster_anchors(data_path, txt_file, k=6, label=None):
    xml_list = get_xml_list(data_path, txt_file)
    dataset = encode_dataset(xml_list, label=label)
    km = KMeans(n_clusters=k, random_state=9).fit(dataset)
    ac = np.array(km.cluster_centers_, dtype=np.int32)
    
    return ac


def encode_dataset(xml_list, label=None):
    dataset = [] # dataset that x_length larger than x_thres
    for xml_file in xml_list: 
        tree = ET.parse(xml_file)
 
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
 
        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin"))
            ymin = int(obj.findtext("bndbox/ymin"))
            xmax = int(obj.findtext("bndbox/xmax"))
            ymax = int(obj.findtext("bndbox/ymax"))
            name = obj.findtext("name")
     
            if label is None or name == label:
                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin: print("Warning: xmin = xmax or ymin = ymax occurs at", xml_file)
                dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)

    
if __name__ == "__main__":
    data_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\train"
    txt_file = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.3.0-double\dataset\train.txt'
    #show_histogram(path)
    label = "s"
    ac = cluster_anchors(data_path, txt_file, k=2, label=label)
    print(ac)
    
    #ac0, ac1 = cluster_anchors(path, k0=3, k1=3)
    #print(ac0)
    #print(ac1)
