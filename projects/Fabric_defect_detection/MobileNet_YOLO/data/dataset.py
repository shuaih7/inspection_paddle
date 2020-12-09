import os, json, cv2, shutil, sys
import numpy as np
import glob as gb
from PIL import Image
from sklearn.utils import shuffle
from skimage.measure import label, regionprops
from lxml.etree import Element, SubElement, tostring, ElementTree
from augmentation import ImageRandomDistort


def random_split_train_valid(ann_path, train_save_path, valid_save_path, percent=0.8, is_shuffle=True):
    ann_list = gb.glob(ann_path + r"/*.json")
    num_files = len(ann_list)
    num_train = int(0.8 * num_files)
    if is_shuffle: ann_list = shuffle(ann_list)
    
    train_list = ann_list[:num_train]
    valid_list = ann_list[num_train:]
    
    for file in train_list:
        _, filename = os.path.split(file)
        save_name = os.path.join(train_save_path, filename)
        shutil.copy(file, save_name)
        
    for file in valid_list:
        _, filename = os.path.split(file)
        save_name = os.path.join(valid_save_path, filename)
        shutil.copy(file, save_name)
    
    print("Done")
    
    
def write_into_txt(file_path, suffix=".xml", save_path=None, save_name="List", is_shuffle=True): 
    file_list = gb.glob(file_path + r"/*"+suffix)
    if is_shuffle: file_list = shuffle(file_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for file in file_list:
            _, filename = os.path.split(file)
            fname, _ = os.path.splitext(filename)
            f.write(fname)
            f.write("\n")
        f.close()
    
    print("Done")
    
    
def write_into_txt_PascalVOC(img_path, ann_path, img_head="", ann_head="", save_path=None, save_name="List", is_shuffle=True): 
    ann_list = gb.glob(ann_path + r"/*.xml")
    if is_shuffle: ann_list = shuffle(ann_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for ann_file in ann_list:
            _, filename = os.path.split(ann_file)
            fname, _ = os.path.splitext(filename)
            img_file = os.path.join(img_path, fname+".png")
            
            assert os.path.isfile(img_file)
            img_item = os.path.join(img_head, fname+".png")
            ann_item = os.path.join(ann_head, fname+".xml")
            f.write(img_item + " " + ann_item)
            f.write("\n")
        f.close()
    
    print("Done")
    

class random_data_generator(object):
    def __init__(self, win_w=1600, win_h=1600, resize_w=352, resize_h=352, def_line_w=15, min_def_w=50, min_def_h=15, 
                 random_window=True, win_sizes=[1440, 1600, 1760]): 
        self.defects = []   # Update the defects info every image file
        
        self.width = win_w    # Width of the cropped window
        self.height = win_h   # Height of the cropped window
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.img_w = 2448
        self.img_h = 2048
        self.def_line_w = def_line_w
        self.min_def_w = min_def_w # Minimum defect width, referred to the resized image
        self.min_def_h = min_def_h # Minimum defect height, referred to the resized image
        self.random_window = random_window
        self.win_sizes = win_sizes
    
    def generate(self, save_name, img_file, img_save_path, ann_save_path, json_path=None):
        """
        Load the image, randomly crop the positive samples, get the corresponding bbxs, and save into PascalVOC xml annotation file.
        
        Args:
            save_name: The name to save the randomly cropped image
            img_file: Original image file
            img_save_path: Dir to save the cropped images
            ann_save_path: Dir to save the transferred PascalVOC xml files
            json_path: Dir holds the Labelme json files, None if those files saved in the img_path 
        
        Returns:

        """
        img_path, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        if json_path is None: json_path = img_path
        json_file = os.path.join(json_path, fname+".json")
        info = "" # indicate the crop window size
        
        if self.random_window:
            random_index = np.random.randint(0, len(self.win_sizes))
            self.width = self.win_sizes[random_index]
            self.height = self.win_sizes[random_index]
            info = str(self.win_sizes[random_index])
        
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            roi = self.generate_roi(js_obj) # Generate an ROI with contains at lease one defect
            img, xml_tree = self.generate_img_ann(img_file, roi, js_obj)
            img, xml_tree = self.preprocessing(img, xml_tree)
            self._save(img, xml_tree, save_name, img_save_path, ann_save_path, info)
            f.close()
        
    def generate_roi(self, js_obj):
        defects = []
        for item in js_obj["shapes"]: defects.append(item["points"])
        self.defects = defects # Pass the defects info to global variable
        num_defects = len(defects) # Number of defects
        defect = np.array(defects[np.random.randint(0,num_defects)], dtype=np.int32)
        
        # Get the left right position
        left_most, right_most = defect[0,0], defect[-1,0] - self.width
        if right_most > left_most: left = np.random.randint(left_most, right_most)
        else: left = left_most # Need to be further considered ... 
        if left + self.width >= self.img_w: left = self.img_w - self.width
        
        # Get the upper lower position
        y_line = self.linear_interpolate(defect)
        ys = y_line[left-left_most:left-left_most+self.width]
        
        upper_most, lower_most = max(0, ys.max()-self.height), min(self.img_h, ys.min())
        if lower_most <= upper_most: lower_most = upper_most + 1
        top = np.random.randint(upper_most, lower_most)
        if top + self.height >= self.img_h: top = self.img_h - self.height

        x0 = left
        y0 = top
        x1 = left + self.width
        y1 = top + self.height
        
        return [x0, y0, x1, y1]
        
    def generate_img_ann(self, img_file, roi, js_obj):
        img = Image.open(img_file)
        img = img.crop(roi) # Crop the img
        roi_x0, roi_y0, roi_x1, roi_y1 = roi
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        
        for defect in self.defects:
            for i in range(len(defect)-1):
                pt_start = (int(defect[i][0]), int(defect[i][1]))
                pt_end   = (int(defect[i+1][0]), int(defect[i+1][1]))
                mask = cv2.line(mask, pt_start, pt_end, color=255, thickness=self.def_line_w, lineType=8)
                
        mask = mask[roi_y0:roi_y1, roi_x0:roi_x1] # Crop the mask image
        label_mask = label(mask, connectivity = 2)
        properties = regionprops(label_mask)
        
        bbxs = []
        min_def_w = int(self.min_def_w * self.width / self.resize_w)
        for prop in properties:
            y1, x1, y2, x2 = prop.bbox
            if x2-x1 > min_def_w: bbxs.append([x1, y1, x2, y2])
            
        img, bbxs = self._resize_img_bbxs(img, bbxs)
        xml_tree = self.create_pascalvoc_xml_tree(img_file, bbxs)
        
        return img, xml_tree
        
    def preprocessing(self, img, xml_tree):
        aug = ImageRandomDistort()
        img = aug.random_brightness(img, lower=0.9, upper=1.1)
        img = aug.random_contrast(img, lower=0.9, upper=1.1)
        #img, xml_tree = aug.random_flip(img, xml_tree, pos=0.7)
        
        return img, xml_tree
        
    def linear_interpolate(self, points):
        array = []
        for i in range(len(points)-1):
            pt, pt_next = points[i], points[i+1]
            x1, y1, x2, y2 = pt[0], pt[1], pt_next[0], pt_next[1]
            for j in range(x2 - x1):
                value = int((y2-y1)/(x2-x1)*j+y1)
                array.append(value)
        array.append(points[-1,1])
        return np.array(array, dtype=np.int32)
        
    def create_pascalvoc_xml_tree(self, img_file, bbxs):
        img_path, filename = os.path.split(img_file)
        _, folder = os.path.split(img_path)
        
        node_root = Element('annotation')
     
        # Folder, filename, and path
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = folder
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = filename
        node_path = SubElement(node_root, 'path')
        node_path.text = img_file
        
        # Data source
        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = "Unknown"
         
        # Image size and segmented
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(self.resize_w)
        node_height = SubElement(node_size, 'height')
        node_height.text = str(self.resize_h)
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = "1"
        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = "0"
         
        for bbx in bbxs: self._create_pascalvoc_object(node_root, bbx)
        xml_tree = ElementTree(node_root)
        
        return xml_tree
        
    def _create_pascalvoc_object(self, node_root, bbx, 
                                 name="defect", 
                                 pose="Unspecified",
                                 truncated="0",
                                 difficult="0"):
                                 
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = name
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = pose
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = truncated
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = difficult
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbx[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbx[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbx[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbx[3])
        
    def _resize_img_bbxs(self, img, bbxs, resample=Image.BILINEAR):
        if self.width == self.resize_w and self.height == self.resize_h: return img ,bbxs
        img = img.resize((self.resize_w, self.resize_h), resample=resample)
        rw, rh, rbbxs = self.resize_w/self.width, self.resize_h/self.height, []
        
        for bbx in bbxs: 
            x1, y1, x2, y2 = int(bbx[0]*rw), int(bbx[1]*rh), int(bbx[2]*rw), int(bbx[3]*rh)
            # Correct the defect height by defining the minimum defect height
            def_h = abs(y2 - y1)
            if def_h < self.min_def_h:
                off_y = int((self.min_def_h-def_h)/2.0 + 0.5)
                
                if y2 >= y1:
                    y1, y2 = y1-off_y, y2+off_y
                    if y1 < 0: y1, y2 = 0, self.min_def_h
                    elif y2 > self.height: y1, y2 = self.height-self.min_def_h, self.height
                """
                else: 
                    y2, y1 = y2-off_y, y1+off_y
                    if y2 < 0: y2, y1 = 0, self.min_def_h
                    elif y1 > self.height: y2, y1 = self.height-self.min_def_h, self.height
                """
            rbbxs.append([x1, y1, x2, y2])
        
        return img, rbbxs
        
    def _save(self, img, xml_tree, save_name, img_save_path, ann_save_path, info=""):
        img_save_name = os.path.join(img_save_path, save_name+"_"+info+".png")
        ann_save_name = os.path.join(ann_save_path, save_name+"_"+info+".xml")
        img.save(img_save_name)
        xml_tree.write(ann_save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
        
        
if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    # 1. Generate the cropped image and the xml label
    # img_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\MER-502-79U3M(NR0190090349)_2020-10-13_13_35_28_592-18.bmp"
    # img_save_path = os.getcwd()
    # ann_save_path = os.getcwd()
    # data_gen = random_data_generator()
    # data_gen.generate("sample", img_file, img_save_path, ann_save_path)
    
    # 2. Display the bbxs
    # from PascalVocParser import PascalVocXmlParser
    # pvoc = PascalVocXmlParser()
    
    # img_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\sample.png"
    # ann_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\sample.xml"
    # image = cv2.imread(img_file, -1)
    # bbxs = pvoc.get_boxes(ann_file)
    
    # for bbx in bbxs:
        # x1, y1, x2, y2 = bbx[0], bbx[1], bbx[2], bbx[3]
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, 1)
    # print("Showing image", img_file)
    # plt.imshow(image, cmap="gray"), plt.title(label)
    # #manager = plt.get_current_fig_manager()
    # #manager.resize(*manager.window.maxsize())
    # plt.show()
        
    # 3. Split the train valid annotation files 
    # ann_path = r"E:\Projects\Fabric_Defect_Detection\ThreeGun_1013\sampling_1013_40Hz_white"
    # train_save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\train_json_white"
    # valid_save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid_json_white"
    # random_split_train_valid(ann_path, train_save_path, valid_save_path)
    
    # 4. Create the training and validation set
    # img_path = r"E:\Projects\Fabric_Defect_Detection\ThreeGun_1013\sampling_1013_40Hz_white"
    # ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\train_json_white"
    # img_save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\train"
    # ann_save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\train"
    
    # num = 2500 # Number of samples will be generated
    # ann_list = gb.glob(ann_path + r"/*.json")
    # data_gen = random_data_generator()

    # index = 0
    # while True:
        # if index >= num: break
        
        # for ann_file in ann_list:
            # _, filename = os.path.split(ann_file)
            # fname, _ = os.path.splitext(filename)
            # img_file = os.path.join(img_path, fname+".bmp")
            # save_name = "train_white_" + str(index)
            # data_gen.generate(save_name, img_file, img_save_path, ann_save_path, json_path=ann_path)
            # print("Finish generating image number:", index)
            # index += 1
            
            # if index >= num: break
            
    
    # 5. Create the training and validation txt file
    img_head = r"ThreeGun_Fabric_1013\Images\valid"
    ann_head = r"ThreeGun_Fabric_1013\Annotations\valid"
    
    img_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO\v1.1\valid"
    ann_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO\v1.1\valid"
    #save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun_Fabric_1013"
    save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\MobileNet_YOLO"
    write_into_txt(ann_path, save_path=save_path, save_name="valid")
    
    #write_into_txt_PascalVOC(img_path, ann_path, img_head, ann_head, save_path=save_path, save_name="valid")
    
    
        
     
    
    