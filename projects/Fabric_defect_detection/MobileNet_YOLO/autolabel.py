# -*- coding: UTF-8 -*-

import os, sys
import config
import shutil
import numpy as np
import glob as gb
import paddle.fluid as fluid
import json, cv2
from PIL import Image
from PIL import ImageDraw
from data import PascalVocXmlParser
from lxml.etree import Element, SubElement, tostring, ElementTree, XMLParser, parse

train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
path = train_parameters['freeze_dir']  # 'model/freeze_model'
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe, model_filename='__model__', params_filename='params')


def fetch_labels(img_path, label_path, suffix=".png"):
    img_list = gb.glob(img_path + r"/*"+suffix)
    
    for img_file in img_list:
        prefix, _ = os.path.splitext(img_file)
        if not os.path.isfile(prefix + ".xml"):
            _, fname = os.path.split(prefix)
            label_file = os.path.join(label_path, fname+".xml")
            save_name = os.path.join(img_path, fname+".xml")
            if os.path.isfile(label_file):
                print("Copying file", label_file, "...")
                shutil.copy(label_file, save_name)
    print("Done")


def create_pvoc_object(node_root, bbx, label,
                       pose="Unspecified",
                       truncated="1",
                       difficult="0"):
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = str(label_dict[label])
    node_pose = SubElement(node_object, 'pose')
    node_pose.text = pose
    node_truncated = SubElement(node_object, 'truncated')
    node_truncated.text = truncated
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = difficult
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(int(bbx[0]))
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(int(bbx[1]))
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(int(bbx[2]))
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(int(bbx[3]))


def create_pvoc(img_file, boxes, labels, origin):
    if len(boxes) == 0: return None
    
    img_path, filename = os.path.split(img_file)
    _, folder = os.path.split(img_path)
    img_w, img_h = origin.size[0], origin.size[1]
        
    node_root = Element('annotation')
     
    # Folder, filename, and path
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = folder
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename
    node_path = SubElement(node_root, 'path')
    node_path.text = img_file
    
    # Data source
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = "Unknown"
     
    # Image size and segmented
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_h)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = "1"
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = "0"
     
    for box, label in zip(boxes, labels): 
        create_pvoc_object(node_root, box, label)
    xml_tree = ElementTree(node_root)
    
    return xml_tree
    
    
def insert_pvoc(label_file, boxes, labels, names=[]):
    node_root = parse(label_file, XMLParser(remove_blank_text=True)).getroot()
    
    for box, label in zip(boxes, labels):
        if label_dict[label] in names:
            create_pvoc_object(node_root, box, label)
    xml_tree = ElementTree(node_root)
    
    return xml_tree
        

def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img


def read_image(img_file):
    """
    读取图片
    :param img_path:
    :return:
    """
    img = Image.open(img_file)
    origin = img
    
    img = resize_img(origin, yolo_config["input_size"])
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    
    return origin, img, resized_img
    
    
def scale_boxes(boxes, origin, resized_img):
    origin_w, origin_h = origin.size[0], origin.size[1]
    resized_w, resized_h = resized_img.size[0], resized_img.size[1]
    rh, rw = origin_h/resized_h, origin_w/resized_w
    
    nboxes = []
    for box in boxes:
        nbox = [box[0]*rw, box[1]*rh, box[2]*rw, box[3]*rh]
        nbox[2] = min(nbox[2], origin_w) # Check the width overflow
        nbox[3] = min(nbox[3], origin_h) # Check the height overflow
        nboxes.append(nbox)
    return nboxes


def infer(img_file):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(img_file)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([tensor_img.shape[-2], tensor_img.shape[-1]], dtype='int32')
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    bboxes = np.array(batch_outputs[0])

    if bboxes.shape[1] != 6:
        # print("No object found")
        return [], [], [], origin, resized_img
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    return boxes, labels, scores, origin, resized_img
    
    
def labelfile(img_file, save_dir):
    boxes, labels, scores, origin, resized_img = infer(img_file)
    boxes = scale_boxes(boxes, origin, resized_img) # Scale the boxes into the original size
    xml_tree = create_pvoc(img_file, boxes, labels, origin)
    if xml_tree is None: return # Only write into xml if there exits defect
    
    _, filename = os.path.split(img_file)
    fname, _ = os.path.splitext(filename)
    save_name = os.path.join(save_dir, fname+".xml")
    
    xml_tree.write(save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
    

def labelfile_insert(img_file, label_dir=None, save_dir=None, names=[]):
    boxes, labels, scores, origin, resized_img = infer(img_file)
    boxes = scale_boxes(boxes, origin, resized_img) # Scale the boxes into the original size
    
    img_dir, filename = os.path.split(img_file)
    if label_dir is None: label_dir=img_dir
    if save_dir is None: save_dir = label_dir
    
    fname, _ = os.path.splitext(filename)
    label_file = os.path.join(label_dir, fname+".xml")
    save_name = os.path.join(save_dir, fname+".xml")
    
    if not os.path.isfile(label_file):
        xml_tree = create_pvoc(img_file, boxes, labels, origin)
    else:
        xml_tree = insert_pvoc(label_file, boxes, labels, names)
    
    if xml_tree is None: return # Only write into xml if there exits defect
    xml_tree.write(save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
    
    
def labeldir(img_dir, save_dir, suffix=".png"):
    img_list = gb.glob(img_dir + r"/*"+suffix)
    
    for img_file in img_list:
        print("Labeling image file", img_file, "...")
        labelfile(img_file, save_dir)
        
        
def labeldir_insert(img_dir, label_dir=None, save_dir=None, names=[], suffix=".png"):
    img_list = gb.glob(img_dir + r"/*"+suffix)
    
    for img_file in img_list:
        print("Labeling image file", img_file, "...")
        labelfile_insert(img_file, label_dir, save_dir, names)
   

if __name__ == '__main__': 
    
    # Autolabeling ...
    image_path = r'E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\valid'
    save_path  = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\valid"
    suffix = ".bmp"
    
    #labeldir(image_path, save_path, suffix)
    labeldir_insert(image_path, names=["striation"], suffix=suffix)
    
    
    # Fetch labels ...
    # img_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\darkgray"
    # label_path = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.0.0\dataset\valid"
    # fetch_labels(img_path, label_path, suffix=".bmp")
    
