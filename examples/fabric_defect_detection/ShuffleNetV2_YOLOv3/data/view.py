import os, cv2
import numpy as np
import glob as gb
from matplotlib import pyplot as plt
from PascalVocParser import PascalVocXmlParser


pvoc = PascalVocXmlParser()

def view_single(img_file, ann_file, label="missing_hole", color=(255,0,0), line_width=5):
    image = cv2.imread(img_file, -1)
    bbxs = pvoc.get_boxes(ann_file)
    
    for bbx in bbxs:
        x1, y1, x2, y2 = bbx[0], bbx[1], bbx[2], bbx[3]
        print("y1 =", y1, ", y2 =", y2, ", heihgt =", abs(y2-y1))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
    print("Showing image", img_file)
    plt.imshow(image, cmap="gray"), plt.title(label)
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    plt.show()
    
    
def view_batch(img_path, ann_path, label="missing_hole", color=(255,0,0), line_width=5):
    img_list = gb.glob(img_path + r"/*.png")
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        ann_file = os.path.join(ann_path, fname+".xml")
        view_single(img_file, ann_file, label=label, color=color, line_width=line_width)
        

if __name__ == "__main__":
    """
    label = r"Spurious_copper"
    img_path = os.path.join(r"E:\BaiduNetdiskDownload\PCB_DATASET\images", label)
    ann_path = os.path.join(r"E:\BaiduNetdiskDownload\PCB_DATASET\Annotations", label)
    
    view_batch(img_path, ann_path, label=label)
    """
    img_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.1\dataset\train"
    ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.1\dataset\train"
    view_batch(img_path, ann_path, label="defect", color=255, line_width=1)
    