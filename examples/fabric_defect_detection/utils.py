import os, sys, cv2
import glob as gb
from sklearn.utils import shuffle


def write_pos_neg_into_txt(pos_path=None, neg_path=None, file_suffix=".jpg", save_path=None, save_name="List.txt", is_shuffle=True):
    if not os.path.exists(pos_path): print("Warning: the positive path does not exist.")
    if not os.path.exists(neg_path): print("Warning: the negative path does not exist.")
    
    pos_list = gb.glob(pos_path + r"/*"+file_suffix)
    neg_list = gb.glob(neg_path + r"/*"+file_suffix)
    
    for i, fname in enumerate(pos_list): 
        _, filename = os.path.split(fname)
        pos_list[i] = "pos_" + filename
        
    for i, fname in enumerate(neg_list): 
        _, filename = os.path.split(fname)
        neg_list[i] = "neg_" + filename
    
    file_list = pos_list + neg_list
    if is_shuffle: file_list = shuffle(file_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for fname in file_list:
            f.write(fname)
            f.write("\n")
        f.close()
    
    print("Done")
    

def read_lines_from_txt(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    item_list = [l.replace("\n", "") for l in lines if l is not ""]
    return item_list


if __name__ == "__main__":
    pos_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\pos"
    neg_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\neg"
    save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\human_classification"
    save_name = "test.txt"
    
    write_pos_neg_into_txt(pos_path, neg_path, save_path=save_path, save_name=save_name)