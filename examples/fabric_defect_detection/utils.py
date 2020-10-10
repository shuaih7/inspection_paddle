import os, sys, cv2
import glob as gb
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from data import AITEXData, NormalData, ImageRandomDistort


mask_path = r"E:\Projects\Fabric_Defect_Detection\oih-fabric-detect-detection\dataset\oih_stitch_skipping\mask"
json_path = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"


def random_valid_generator(normal_path, aitex_path, save_path_pos, save_path_neg, num=3000):
    normal = NormalData(mask_path)
    aitex  = AITEXData(json_path)
    aug    = ImageRandomDistort()
    
    normal_list = gb.glob(normal_path+r"/*.png")
    aitex_list  = gb.glob(aitex_path+r"/*.png")
    file_list   = normal_list + aitex_list
    file_list   = shuffle(file_list)
    
    i = 0
    while True:
        if i >= num: break
        for file in file_list:
            if i >= num: break 
            
            _, filename = os.path.split(file)
            fname, _ = os.path.splitext(filename)
            if fname[:3] == "MER":
                box, label = normal.generate(fname, pos=0.7)
                img_file   = os.path.join(normal_path, fname+".png")
            else: 
                box, label = aitex.generate(fname, pos=0.6)
                img_file   = os.path.join(aitex_path, fname+".png")
            
            img = Image.open(img_file)
            img = aug.crop_and_resize(img, box=box, size=[224, 224])
            img = aug.random_brightness(img, lower=0.7, upper=1.3)
            img = aug.random_contrast(img, lower=0.9, upper=1.1)
            img = aug.random_flip(img, pos=0.6)
            #img = aug.random_rotate(img, pos=0.6)
            img = aug.random_invert(img, pos=0.3)
            
            image = np.array(img, dtype=np.float32)
            if label == 1: img_file = os.path.join(save_path_pos, str(label)+"_"+str(i)+".png")
            else: img_file = os.path.join(save_path_neg, str(label)+"_"+str(i)+".png")
            print("Writing the valid image number:", i)
            cv2.imwrite(img_file, image)
            
            i += 1


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
    normal_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\x_train"
    aitex_path = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"
    save_path_pos = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\random_valid_pos"
    save_path_neg = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\random_valid_neg"
    random_valid_generator(normal_path, aitex_path, save_path_pos, save_path_neg, num=3000)

    # pos_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\pos"
    # neg_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\neg"
    # save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\human_classification"
    # save_name = "test.txt"
    
    # write_pos_neg_into_txt(pos_path, neg_path, save_path=save_path, save_name=save_name)
