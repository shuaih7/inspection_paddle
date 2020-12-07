import os, cv2
import glob as gb
import numpy as np
from PIL import Image
from data import PascalVocXmlParser
import paddlelite.lite as lite
from matplotlib import pyplot as plt


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
    
    
def rotate_img_bbxs(img_file, ann_file):
    pvoc = PascalVocXmlParser()
    img = Image.open(img_file)
    boxes = pvoc.get_boxes(ann_file)
    
    img_orig = np.array(img, dtype=np.uint8)
    img_clock = np.array(img.rotate(-90), dtype=np.uint8)
    img_count = np.array(img.rotate(90), dtype=np.uint8)
    
    boxes_clock, boxes_count = [], []
    for box in boxes:
        boxes_clock.append(get_rot90_box(box, d="-"))
        boxes_count.append(get_rot90_box(box, d="+"))
    
    img_orig = draw_boxes(img_orig, boxes)
    img_clock = draw_boxes(img_clock, boxes_clock)
    img_count = draw_boxes(img_count, boxes_count)
    
    plt.subplot(1,3,1), plt.imshow(img_orig, cmap="gray"), plt.title("Original image")
    plt.subplot(1,3,2), plt.imshow(img_clock, cmap="gray"), plt.title("Clockwise Rotate")
    plt.subplot(1,3,3), plt.imshow(img_count, cmap="gray"), plt.title("Counter-clockwise rotate")
    plt.show()
    

def get_rot90_box(box, w=352, h=352, d="+"):
    x1, y1, x2, y2 = box
    _nx1, _ny1 = get_rot90_pos(x1, y1, w=w, h=h, d=d)
    _nx2, _ny2 = get_rot90_pos(x2, y2, w=w, h=h, d=d)
    
    nx1, nx2 = min(_nx1, _nx2), max(_nx1, _nx2)
    ny1, ny2 = min(_ny1, _ny2), max(_ny1, _ny2)
    return [nx1, ny1, nx2, ny2]
    
    
def get_rot90_pos(x, y, w=352, h=352, d="+"): # + means counter-clockwise
    if d == "+":
        nx = y
        ny = h - x
        return nx, ny
    elif d == "-":
        nx = w - y
        ny = x
        return nx, ny
    else: raise ValueError("Invalid d value.")
    
    
def draw_boxes(image, boxes=[], color=(255,0,0), thickness=2):
    if len(boxes) == 0: return image
    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        image = cv2.rectangle(image, start_point, end_point, color=color, thickness=thickness)
    return image
    
    
def crop_and_resize(img_file, save_dir, target_size=(512,512)):
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    offh, offw = 0, 0
    if h > w: offh = int((h-w)/2)
    else: offw = int((w-h)/2)
    
    image = image[offh:h-offh, offw:w-offw, :]
    
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    _, filename = os.path.split(img_file)
    fname, _ = os.path.splitext(filename)
    save_name = os.path.join(save_dir, fname+".png")
    cv2.imwrite(save_name, image)
    
    print("Done")
    
    
def crop_and_resize_pillow(img_file, save_dir, target_size=(352,352)):
    image = Image.open(img_file)
    w, h = image.size
    if image.mode != 'RGB':
        image = image.convert('RGB')
                    
    offh, offw = 0, 0
    if h > w: offh = int((h-w)/2)
    else: offw = int((w-h)/2)
    
    image = image.crop([offw, offh, w-offw, h-offh])
    
    image = image.resize(target_size, resample=Image.BILINEAR)
    
    _, filename = os.path.split(img_file)
    fname, _ = os.path.splitext(filename)
    save_name = os.path.join(save_dir, fname+".png")
    image.save(save_name)
    
    print("Done")
    
    
if __name__ == "__main__":
    # ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.2\dataset\train"
    # num_defects = check_defects(ann_path)
    # print("The total number of defects is", num_defects)
    
    # model_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO\pretrained_model"
    # save_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO"
    # model_compress(model_dir, save_dir, save_name="fast_yolo.nb")
        
    img_dir = r"E:\Projects\Fabric_Defect_Detection\model_dev\ThreeGun_Visit_1130\original_2"
    save_dir = r"E:\Projects\Fabric_Defect_Detection\model_dev\ThreeGun_Visit_1130\valid_2"
    img_list = gb.glob(img_dir+r"/*.bmp")
    
    for img_file in img_list:
        crop_and_resize_pillow(img_file, save_dir)