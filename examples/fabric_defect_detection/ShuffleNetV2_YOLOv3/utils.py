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
    
    
def check_defects(ann_path):
    pvoc = PascalVocXmlParser()
    ann_list = gb.glob(ann_path + r"/*.xml")
    
    num_defects = 0
    for ann_file in ann_list:
        boxes = pvoc.get_boxes(ann_file)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmin > xmax: print(ann_file, "xmin > xmax")
            if ymin > ymax: print(ann_file, "ymin > ymax")
        
    return num_defects
    
    
if __name__ == "__main__":
    ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train"
    num_defects = check_defects(ann_path)
    print("The total number of defects is", num_defects)
        