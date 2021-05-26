import os
import sys
from sklearn.utils import shuffle


def merge_labels(label_file_list, save_label_file, is_shuffle=True):
    data_lines = []
    for label_file in label_file_list:  
        print("Processing label file", label_file, "...")
        with open(label_file, "rb") as fin:
            data_lines += fin.readlines()
            fin.close()
    
    if is_shuffle:
        data_lines = shuffle(data_lines)
            
    with open(save_label_file, "wb") as fout:
        for item in data_lines:
            fout.write(item)
        fout.close()
        
        
if __name__ == "__main__":
    label_file1 = r"E:\Projects\Part_Number\dataset\det_valid\20210113\label.txt"
    label_file2 = r"E:\Projects\Part_Number\dataset\det_valid\20210122\label.txt"
    save_label_file = r"E:\Projects\Part_Number\dataset\det_valid\label.txt"
    
    merge_labels([label_file1, label_file2], save_label_file, is_shuffle=True)