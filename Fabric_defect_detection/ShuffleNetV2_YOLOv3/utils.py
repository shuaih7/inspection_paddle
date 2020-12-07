import os
import glob as gb
from data import PascalVocXmlParser
import paddlelite.lite as lite


def count_defects(ann_path):
    pvoc = PascalVocXmlParser()
    ann_list = gb.glob(ann_path + r"/*.xml")
    
    num_defects = 0
    for ann_file in ann_list:
        boxes = pvoc.get_boxes(ann_file)
        num_defects += boxes.shape[0]
        
    return num_defects
    
    
def check_defects(ann_path):
    pvoc = PascalVocXmlParser()
    ann_list = gb.glob(ann_path + r"/*.xml")
    
    num_defects = 0
    for ann_file in ann_list:
        boxes = pvoc.get_boxes(ann_file)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmin > xmax: print(ann_file, "xmin > xmax")
            if ymin > ymax: print(ann_file, "ymin > ymax")
        
    return num_defects
    
    
def model_compress(model_dir, save_dir=None, save_name="__model__", mode=""):
    a=lite.Opt()
    
    # 非combined形式
    if mode != "combined":
        a.set_model_dir(model_dir)

    # conmbined形式
    else:
        a.set_model_file(os.path.join(model_dir, "__model__"))
        a.set_param_file(os.path.join(model_dir, "params"))

    if save_dir is None: save_dir = model_dir
    a.set_optimize_out(os.path.join(save_dir, save_name))
    a.set_valid_places("x86")

    a.run()
    
    
if __name__ == "__main__":
    # ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train"
    # num_defects = check_defects(ann_path)
    # print("The total number of defects is", num_defects)
    
    model_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\ShuffleNetV2_s1.0\saved_model"
    save_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\ShuffleNetV2_s1.0\lite_model"
    model_compress(model_dir, save_dir)
        