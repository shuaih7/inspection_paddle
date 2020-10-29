import os
import glob as gb
from data import PascalVocXmlParser


def count_defects(ann_path):
    pvoc = PascalVocXmlParser()
    ann_list = gb.glob(ann_path + r"/*.xml")
    
    num_defects = 0
    for ann_file in ann_list:
        boxes = pvoc.get_boxes(ann_file)
        num_defects += boxes.shape[0]
        
    return num_defects
    
    
if __name__ == "__main__":
    ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun_YOLO\valid"
    num_defects = count_defects(ann_path)
    print("The total number of defects is", num_defects)
        