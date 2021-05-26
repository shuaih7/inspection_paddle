import os
import sys
import cv2
import numpy as np
import glob as gb
from matplotlib import pyplot as plt


dictionary = "0123456789abcdefghijklmnopqrstuvwxyz"
dictionary = list(dictionary)


def clockwise_points(points):
    if len(points) != 4:
        raise ValueError("The points length should be 4.")
    
    clock_pts = []    
    xs = np.array([points[0][0], points[1][0], points[2][0], 
                    points[3][0]], dtype=np.float32)
    ys = np.array([points[0][1], points[1][1], points[2][1], 
                    points[3][1]], dtype=np.float32)

    # Get the left two points
    x_sort = np.argsort(xs)
    if points[x_sort[0]][1] < points[x_sort[1]][1]:
        clock_pts.append(points[x_sort[0]])
        last_pt = points[x_sort[1]]
    else:
        clock_pts.append(points[x_sort[1]])
        last_pt = points[x_sort[0]]
        
    if points[x_sort[2]][1] < points[x_sort[3]][1]:
        clock_pts.append(points[x_sort[2]])
        clock_pts.append(points[x_sort[3]])
    else:
        clock_pts.append(points[x_sort[3]])
        clock_pts.append(points[x_sort[2]])
    clock_pts.append(last_pt)
    
    return clock_pts
        

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
    
    
def generate_rec_dataset(data_dir, 
                         label_file, 
                         save_data_dir="",
                         label_save_name="label.txt"):
                         
    label_save_name = os.path.join(save_data_dir, label_save_name)
    with open(label_save_name, "w") as fout:    
        with open(label_file, "rb") as fin:
            data_lines = fin.readlines()
            fin.close()
            
        img_num = len(data_lines)
       
        for data_line in data_lines:
            substr = data_line.decode('utf-8').strip("\n").split("\t")
            _, filename = os.path.split(substr[0])
            fname, suffix = os.path.splitext(filename)
            
            img_path = os.path.join(data_dir, substr[0])
            print("Cropping image file", img_path, "...")
            image = cv2.imread(img_path, -1)
            labels = eval(substr[1].replace("false", "False"))
            
            for i, label in enumerate(labels):
                txt = label['transcription']
                
                skip_flag = False
                for char in txt:
                    if char.lower() not in dictionary: skip_flag = True
                if skip_flag: continue
                    
                box = np.array(clockwise_points(label['points']), dtype=np.float32)
                dst_img = get_rotate_crop_image(image, box)
                
                img_name = fname+"_"+str(i)+suffix
                _, prefix = os.path.split(save_data_dir)
                img_write_name = os.path.join(prefix, img_name)
                img_save_name = os.path.join(save_data_dir, img_name)
                cv2.imwrite(img_save_name, dst_img)
                
                item = img_write_name + "\t" + txt + "\n"
                fout.write(item)
                
        fout.close()
    print("Done")
    
if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset\det_valid"
    label_file = r"E:\Projects\Part_Number\dataset\det_valid\20210113\label.txt"
    save_data_dir = r"E:\Projects\Part_Number\dataset\rec_valid\20210113"
    generate_rec_dataset(data_dir, label_file, save_data_dir=save_data_dir, label_save_name="label.txt")