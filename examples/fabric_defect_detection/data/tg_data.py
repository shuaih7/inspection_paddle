import os, json, sys
import numpy as np

class TGData(object):
    def __init__(self, json_path):
        self.json_path = json_path
        self.width = 672
        self.height = 672
        self.img_w = 2448
        self.img_h = 2048
        
    def generate(self, fname, val=None, pos=0.5): # pos: posibility for generating the positive sample
        json_file = os.path.join(self.json_path, fname+".json")
        if os.path.isfile(json_file): return self.generate_positive(json_file)
        else: return self.generate_negative()
        
    def generate_positive(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            defects = []
            
            for item in js_obj["shapes"]: defects.append(item["points"])
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
            f.close()

        x0 = left
        y0 = top
        x1 = left + self.width
        y1 = top + self.height
        
        return [x0, y0, x1, y1], 1
          
    def generate_negative(self):
        x0 = np.random.randint(0, self.img_w-self.width)
        y0 = np.random.randint(0, self.img_h-self.height)
        x1 = x0 + self.width
        y1 = y0 + self.height
        
        return [x0, y0, x1, y1], 0
        
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
        
if __name__ == "__main__":
    import cv2
    import glob as gb
    from matplotlib import pyplot as plt
    
    img_path  = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun"
    json_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun"
    img_list = gb.glob(img_path+r"/*.bmp")
    data = TGData(json_path)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        
        fname, _ = os.path.splitext(filename)
        json_file = os.path.join(json_path, fname+".json")
        
        box, label = data.generate(fname)
        
        img = cv2.imread(img_file, -1)
        img = img[box[1]:box[3],box[0]:box[2]]
        
        if label == 1: title = "Positive"
        else: title = "Negative"
        print(filename)
        plt.imshow(img, cmap="gray"), plt.title(title)
        plt.show()