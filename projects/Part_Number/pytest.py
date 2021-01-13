import os
import sys
import json


def get_labels(label_file):
    with open(label_file, "rb") as fin:
        label_list = fin.readlines()
    return label_list
    
    
if __name__ == "__main__":
    label_file = r"E:\Projects\Part_Number\baidu\icdar2015\text_localization\test_icdar2015_label.txt"
    label_list = get_labels(label_file)
    label_infor = label_list[0]
    label_infor = label_infor.decode()
    label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
    substr = label_infor.strip("\n").split("\t")
    
    img_path = substr[0]
    label = json.loads(substr[1])
    
    print(type(label[0]))
    print(label[0])
    
