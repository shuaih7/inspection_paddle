import os, cv2, random
import numpy as np
import glob as gb
from lxml import etree
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from PascalVocParser import PascalVocXmlParser


pvoc = PascalVocXmlParser()


def get_rot90_box(box, w=352, h=352, d="+"):
    x1, y1, x2, y2 = box
    _nx1, _ny1 = get_rot90_pos(x1, y1, w=w, h=h, d=d)
    _nx2, _ny2 = get_rot90_pos(x2, y2, w=w, h=h, d=d)
    
    nx1, nx2 = min(_nx1, _nx2), max(_nx1, _nx2)
    ny1, ny2 = min(_ny1, _ny2), max(_ny1, _ny2)
    return [nx1, ny1, nx2, ny2]
    
    
def get_rot90_pos(x, y, w=352, h=352, d="+"): # + means counter-clockwise
    if d == "+":
        nx = y
        ny = h - x
        return nx, ny
    elif d == "-":
        nx = w - y
        ny = x
        return nx, ny
    else: raise ValueError("Invalid d value.")
    
    
def draw_boxes(image, boxes=[], color=(255,0,0), thickness=2):
    if len(boxes) == 0: return image
    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        image = cv2.rectangle(image, start_point, end_point, color=color, thickness=thickness)
    return image
    
    
def rotate_img_ann(ann_file, img_save_dir="", ann_save_dir="", d="+"):
    path, filename = os.path.split(ann_file)
    fname, _ = os.path.splitext(filename)
    img_file = os.path.join(path, fname+".png")
    print("Processing ann file", ann_file, "...")
    
    img = Image.open(img_file)
    tree = etree.parse(ann_file)
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    for elem in root:
        if elem.tag == "object":
            bndbox = elem.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            
            box = [xmin, ymin, xmax, ymax]
            box = get_rot90_box(box, w=width, h=width, d=d)
            xmin, ymin, xmax, ymax = box
            
            bndbox.find("xmin").text = str(xmin)
            bndbox.find("ymin").text = str(ymin)
            bndbox.find("xmax").text = str(xmax)
            bndbox.find("ymax").text = str(ymax)
            
    if d == "+": img = img.rotate(90)
    elif d == "-": img = img.rotate(-90)
    else: raise ValueError("Invalid d value.")
    
    img_save_name = os.path.join(img_save_dir, "r_"+fname+".png")
    ann_save_name = os.path.join(ann_save_dir, "r_"+fname+".xml")
    
    img.save(img_save_name)
    tree.write(ann_save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
    

def rotate_all(ann_dir, img_save_dir, ann_save_dir):
    ann_list = gb.glob(ann_dir +r"/*.xml")
    
    for ann_file in ann_list:
        value = random.random()
        if value < 0.5: rotate_img_ann(ann_file, img_save_dir, ann_save_dir, d="+")
        else: rotate_img_ann(ann_file, img_save_dir, ann_save_dir, d="-")
    print("Done")
    
    
if __name__ == "__main__":
    ann_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid"
    img_save_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\v_valid"
    ann_save_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\v_valid"
    
    rotate_all(ann_dir, img_save_dir, ann_save_dir)