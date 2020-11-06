import os, cv2
import numpy as np
import glob as gb
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
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
    
    
def view_batch(img_path, ann_path, label="missing_hole", color=(255,0,0), line_width=5, is_shuffle=True):
    img_list = gb.glob(img_path + r"/*.png")
    if is_shuffle: img_list = shuffle(img_list)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        ann_file = os.path.join(ann_path, fname+".xml")
        view_single(img_file, ann_file, label=label, color=color, line_width=line_width)
        
        
class view_preprocessing(object):
    def __init__(self):
        pass
        
    def view_brightness(self, img_file, vals=[]):
        if len(vals) == 0: return
        else: num_plots = len(vals)
        
        img = Image.open(img_file)
        for i, val in enumerate(vals):
            plot_img = ImageEnhance.Brightness(img).enhance(val)
            plt.subplot(1,num_plots,i+1), plt.imshow(plot_img, cmap='gray'), plt.title("Brightness "+str(val))
        plt.show()
        
    def view_contrast(self, img_file, vals=[]):
        if len(vals) == 0: return
        else: num_plots = len(vals)
        
        img = Image.open(img_file)
        for i, val in enumerate(vals):
            plot_img = ImageEnhance.Contrast(img).enhance(val)
            plt.subplot(1,num_plots,i+1), plt.imshow(plot_img, cmap='gray'), plt.title("Contrast "+str(val))
        plt.show()
        

if __name__ == "__main__":
    """
    img_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train\train_1407_1760.png"
    ann_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train\train_1407_1760.xml"
    view_single(img_file, ann_file, label="defect", color=255, line_width=1)
    """
    """
    label = r"Spurious_copper"
    img_path = os.path.join(r"E:\BaiduNetdiskDownload\PCB_DATASET\images", label)
    ann_path = os.path.join(r"E:\BaiduNetdiskDownload\PCB_DATASET\Annotations", label)
    
    view_batch(img_path, ann_path, label=label)
    """
    
    
    img_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train"
    ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train"
    view_batch(img_path, ann_path, label="defect", color=255, line_width=1)
    """
    
    img_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train\train_1392_1440.png"
    
    viewer = view_preprocessing()
    viewer.view_brightness(img_file, vals=[0.5, 1, 1.5])
    viewer.view_contrast(img_file, vals=[0.5, 1, 1.5])
    """

    