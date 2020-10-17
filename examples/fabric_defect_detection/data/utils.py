import os, cv2, shutil, json
import glob as gb
import numpy as np
from sklearn.utils import shuffle
from skimage.measure import label, regionprops


def write_into_txt(normal_path, aitex_path=None, save_path=None, save_name="List", is_shuffle=True):
    normal_list = gb.glob(normal_path+r"/*.png")
    if aitex_path is not None: aitex_list = gb.glob(aitex_path+r"/*.png")
    else: aitex_list = []
    
    file_list   = normal_list + aitex_list
    if is_shuffle: file_list = shuffle(file_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for fname in file_list:
            _, temp_name = os.path.split(fname)
            filename, _ = os.path.splitext(temp_name)
            f.write(filename)
            f.write("\n")
        f.close()
    
    print("Done")
    
    
def rgb_to_gray(path, save_path, suf=".png"):
    img_list = gb.glob(path+r"/*"+suf)
    
    for img_file in img_list:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        _, filename = os.path.split(img_file)
        cv2.imwrite(os.path.join(save_path,filename), img)
        
    print("Done")
    

def region_proposals():
    mask_file = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\def_masks\0001_002_00_mask.png"
    json_file = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\def_images\0001_002_00.json"
    marking = {
      "label": "1",
      "line_color": None,
      "fill_color": None,
      "points": [],
      "shape_type": "rectangle",
      "flags": {}
    }
    
    mask_label = cv2.imread(mask_file, -1)   
    mask_label[mask_label > 0] = 1
    label_mask = label(mask_label, connectivity = 2)
    properties = regionprops(label_mask)
    
    for prop in properties:
        y1, x1, y2, x2 = prop.bbox
        marking["points"] = [[x1, y1],[x2, y2]]
    
    with open(json_file, "r", encoding="utf-8") as f:
        js_obj = json.load(f)
        
        for item in js_obj["shapes"]: # Check whether the defect has already been transferred
            if item["label"] == "1": 
                print("Mask file:", mask_file, "has already transferred into the corresponding json file.")
                f.close()
                return 
                
        js_obj["shapes"].append(marking)
        f.close()
        
    with open(json_file, "w", encoding="utf-8") as ff:
        res = json.dumps(js_obj, indent=4)
        ff.write(res)
        ff.close()
    
    print("Done")


def image_label_check():
    image_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
    label_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
    
    image_list = gb.glob(image_path + r"/*.bmp")
    for file in image_list:
        _, filename = os.path.split(file)
        fname, suf = os.path.splitext(filename)
        label_file = os.path.join(label_path, fname+".json")
        if not os.path.isfile(label_file): print("Could not find the mask file "+label_file)
        # else: 
            # shutil.copy(file, os.path.join(r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\def_images", filename))
            # shutil.copy(label_file, os.path.join(r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\def_masks", fname+"_mask"+suf))
    print("Done")


def sort_images():
    image_path = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\2608691-202020u"
    image_list = gb.glob(image_path+r"/*.png")

    src_path = os.path.join(image_path, "src")
    cpy_path = os.path.join(image_path, "cpy")

    for i, file in enumerate(image_list):
        _, filename = os.path.split(file)
        srcfile = os.path.join(src_path, filename)
        cpyfile = os.path.join(cpy_path, filename)
        
        if i % 2 == 0: shutil.move(file, cpyfile)
        else: shutil.move(file, srcfile)
        
    print("Done")
    
    
if __name__ == "__main__":
    #path = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"
    #channel_search(path)
    
    #path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\x_valid"
    #save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\x_valid"
    #rgb_to_gray(path, save_path)
    
    # For training ...
    # normal_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
    # aitex_path  = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"
    # save_path   = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection"
    
    normal_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\pos_neg_train"
    save_path   = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection"
    write_into_txt(normal_path, aitex_path=None, save_path=save_path, save_name="train", is_shuffle=True)

    
