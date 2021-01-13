import os
import sys
import cv2
import json
import glob as gb
import numpy as np


def draw_polylines(image, polylines, texts=None, isClosed=True, size=0.6, color=(0,255,0), thickness=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    polylines = np.array(polylines, dtype=np.int32)#.reshape((-1,1,2))
    
    for i, line in enumerate(polylines):
        pt = (line[0][0], line[0][1])
        line = line.reshape((-1,1,2))
        image = cv2.polylines(image, [line], isClosed=isClosed, color=color, thickness=thickness)
        if texts is not None:
            image = cv2.putText(image, texts[i], pt, fontFace=font, 
                fontScale=size, color=color, thickness=max(1,thickness-1))
    return image   
    
    
def draw_initial_point(image, polylines, radius=3, color=(0,0,255), thickness=5):
    for line in polylines:
        point = (int(line[0][0]), int(line[0][1]))
        image = cv2.circle(image, point, radius=radius, color=color, thickness=thickness)
    return image
    
    
def flip_image_top_bottom(image):
    #flip_image = cv2.flip(image, 0)
    flip_image = cv2.flip(image, -1)
    return flip_image
    
    
def flip_label_top_bottom(labels, image_shape):
    img_h, img_w = image_shape
    
    flip_labels = []
    for label in labels:
        data = {}
        txt = label['transcription']
        pts = label['points']
        new_pts = [[img_w-pts[0][0], img_h-pts[0][1]], [img_w-pts[1][0], img_h-pts[1][1]],
                   [img_w-pts[2][0], img_h-pts[2][1]], [img_w-pts[3][0], img_h-pts[3][1]]]
                   
        if txt[0] == "O": txt = txt[1:]
        else: txt = "O" + txt
        data["transcription"] = txt
        data["points"] = new_pts
        data["difficult"] = False
        flip_labels.append(data)
        
    return flip_labels
    
    
def extend_data_by_flip(data_dir, label_file, new_label_file=None):
    label_ext_list = []
    
    print("Start laoding data and extending ...")
    with open(label_file, "rb") as fin:
        label_infor_list = fin.readlines()
        
        for label_infor in label_infor_list:
            label_infor = label_infor.decode()
            label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
            
            substr = label_infor.strip("\n").split("\t")
            img_path = os.path.join(data_dir, substr[0])
            labels = json.loads(substr[1])
            
            # Flip the input image
            image = cv2.imread(img_path, -1)
            image_shape = image.shape[:2]
            flip_image = flip_image_top_bottom(image)
            prefix, filename = os.path.split(substr[0])
            fname, suffix = os.path.splitext(filename)
            img_label = os.path.join(prefix, fname + "_flip" + suffix)
            save_name = os.path.join(os.path.join(data_dir, prefix), fname + "_flip" + suffix)
            cv2.imwrite(save_name, flip_image)
            
            flip_labels = flip_label_top_bottom(labels, image_shape)
            ori_label_item = os.path.join(prefix, filename) + "\t" + str(labels)
            new_label_item = img_label + "\t" + str(flip_labels)
            label_ext_list.append(ori_label_item.replace("'","\""))
            label_ext_list.append(new_label_item.replace("'","\""))
            
            fin.close()
    
    print("Start writing labels ...")
    if new_label_file is None: new_label_file = label_file
    with open(new_label_file, "w") as fout:
        for label_ext in label_ext_list:
            write_item = label_ext
            fout.write(write_item + "\n")
        fout.close()


if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset"
    label_file = r"E:\Projects\Part_Number\dataset\20210112_valid\Label.txt"
    new_label_file = r"E:\Projects\Part_Number\dataset\20210112_valid\valid.txt"
    
    extend_data_by_flip(data_dir, label_file, new_label_file)