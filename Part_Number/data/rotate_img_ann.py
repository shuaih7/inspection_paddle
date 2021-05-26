import os
import sys
import cv2
import glob as gb
import numpy as np
from matplotlib import pyplot as plt
from utils import *


def rotate_img_ann(data_dir, label_file, save_path, angle=0):
    if angle not in [0, 90, -90, 180]:
        raise ValueError("Angle value only support 0, 90, -90, and 180.")
    
    new_label_file = os.path.join(save_path, "label_new.txt")
    with open(new_label_file, "w") as fout:
        with open(label_file, "rb") as fin:
            data_lines = fin.readlines()
            
            for data_line in data_lines:
                substr = data_line.decode('utf-8').strip("\n").split("\t")
                
                img_path = os.path.join(data_dir, substr[0])
                print("Processing image file", img_path, "...")
                _, filename = os.path.split(img_path)
                image = cv2.imread(img_path, -1)
                labels = eval(substr[1].replace("false", "False"))  
                img_shape = image.shape[:2]

                # Rotate the image matrix
                image_rotate = rotate_bound(image, angle)
                
                # Rotate the annotations
                results_rotate = []
                for label in labels:
                    text = label["transcription"]
                    points = label["points"]
                    points_rotate = rotate_points(points, img_shape, angle)
                    new_label = {"transcription": text, "points": points_rotate}
                    results_rotate.append(new_label)
                """    
                for res in results_rotate:
                    text = res["transcription"]
                    points = res["points"]
                    img_draw = draw_polylines(image_rotate, [points], [text], color=255, size=1.0)
                plt.imshow(img_draw, cmap="gray"), plt.show()
                """
                _, prefix = os.path.split(save_path)
                img_save_name = os.path.join(save_path, filename)
                item = os.path.join(prefix, filename) + "\t" + str(results_rotate) + "\n"
                cv2.imwrite(img_save_name, image_rotate)
                fout.write(item.replace("False", "false"))
                
        fin.close()
    fout.close()
    
    
if __name__ == "__main__":
    data_dir = r"E:\Projects\Part_Number\dataset"
    label_file = r"E:\Projects\Part_Number\dataset\20210122\Label.txt"
    save_path = r"E:\Projects\Part_Number\dataset\20210122"
    rotate_img_ann(data_dir, label_file, save_path, angle=90)
    