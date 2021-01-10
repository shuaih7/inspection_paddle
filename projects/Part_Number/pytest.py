import os
import sys
import random
import numpy as np


def search_labels(label_file):
    data_lines = []
    
    with open(label_file, "rb") as f:
        lines = f.readlines()
        lines = random.sample(lines, 1)
        print(lines)
        data_lines.extend(lines)
    return data_lines
    

if __name__ == "__main__":
    label_file = r"E:\Projects\Part_Number\baidu\icdar2015\text_localization\test_icdar2015_label.txt"
    data_lines = search_labels(label_file)
    #print(data_lines)
