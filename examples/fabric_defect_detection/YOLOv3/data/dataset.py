import os, json, cv2
import numpy as np
from PIL import Image
from augmentation import ImageRandomDistort
from skimage.measure import label, regionprops
from lxml.etree import Element, SubElement, tostring, ElementTree


class random_data_generator(object):
    def __init__(self): 
        self.defects = []   # Update the defects info every image file
        
        self.width = 672    # Width of the cropped window
        self.height = 672   # Height of the cropped window
        self.resize_w = 416
        self.resize_h = 416
        self.img_w = 2448
        self.img_h = 2048
    
    def generate(self, save_name, img_file, img_save_path, ann_save_path, json_path=None):
        """
        Load the image, randomly crop the positive samples, get the corresponding bbxs, and save into PascalVOC xml annotation file.
        
        Args:
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
        
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            roi = self.generate_roi(js_obj) # Generate an ROI with contains at lease one defect
            img, xml_tree = self.generate_img_ann(img_file, roi, js_obj)
            img, xml_tree = self.preprocessing(img, xml_tree)
            self._save(img, xml_tree, save_name, img_save_path, ann_save_path)
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
        else: left = left_most
        if left + self.width >= self.img_w: left = self.img_w - self.width
        
        # Get the upper lower position
        y_line = self.linear_interpolate(defect)
        ys = y_line[left-left_most:left-left_most+self.width]
        
        upper_most, lower_most = max(0, ys.max()-self.height), min(self.img_h, ys.min())
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
                mask = cv2.line(mask, pt_start, pt_end, color=255, thickness=2, lineType=8)
                
        mask = mask[roi_y0:roi_y1, roi_x0:roi_x1] # Crop the mask image
        label_mask = label(mask, connectivity = 2)
        properties = regionprops(label_mask)
        
        bbxs = []
        for prop in properties:
            y1, x1, y2, x2 = prop.bbox
            bbxs.append([x1, y1, x2, y2])
            
        img, bbxs = self._resize_img_bbxs(img, bbxs)
        xml_tree = self.create_pascalvoc_xml_tree(img_file, bbxs)
        
        return img, xml_tree
        
    def preprocessing(self, img, xml_tree):
        aug = ImageRandomDistort()
        img = aug.random_brightness(img, lower=0.9, upper=1.1)
        img = aug.random_contrast(img, lower=0.9, upper=1.1)
        
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
        for bbx in bbxs: rbbxs.append([int(bbx[0]*rw), int(bbx[1]*rh), int(bbx[2]*rw), int(bbx[3]*rh)])
        
        return img, rbbxs
        
    def _save(self, img, xml_tree, save_name, img_save_path, ann_save_path):
        img_save_name = os.path.join(img_save_path, save_name+".png")
        ann_save_name = os.path.join(ann_save_path, save_name+".xml")
        img.save(img_save_name)
        xml_tree.write(ann_save_name, pretty_print=True, xml_declaration=False, encoding='utf-8')
        
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # Gnerate the cropped image and the xml label
    # img_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\MER-502-79U3M(NR0190090349)_2020-10-13_13_35_28_592-18.bmp"
    # img_save_path = os.getcwd()
    # ann_save_path = os.getcwd()
    # data_gen = random_data_generator()
    # data_gen.generate("sample", img_file, img_save_path, ann_save_path)
    
    # Display the bbxs
    from PascalVocParser import PascalVocXmlParser
    pvoc = PascalVocXmlParser()
    
    img_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\sample.png"
    ann_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\YOLOv3\data\sample.xml"
    image = cv2.imread(img_file, -1)
    bbxs = pvoc.get_boxes(ann_file)
    
    for bbx in bbxs:
        x1, y1, x2, y2 = bbx[0], bbx[1], bbx[2], bbx[3]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, 1)
    print("Showing image", img_file)
    plt.imshow(image, cmap="gray"), plt.title(label)
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    plt.show()
    
        
     
    
    