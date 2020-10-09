import os, json
import numpy as np

class AITEXData(object):
    def __init__(self, json_path):
        self.json_path = json_path
        self.min_len = 200
        self.max_len = 256
        self.img_w   = 4096
        self.img_h   = 256
        
    def generate(self, fname, val=None, pos=0.2): # pos: posibility for generating the positive sample
        json_file = os.path.join(self.json_path, fname+".json")
        if val is NOne: val = np.random_uniform(0, 1)
        if val < pos: return self.generate_positive(json_file)
        else: return self.generate_negative(json_file)
        
    def generate_positive(self, json_file):
        min_len = self.min_len
        max_len = self.max_len
        img_w   = self.img_w
        img_h   = self.img_h
        left, xmin, xmax = 0, -1, -1
        
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            defects = []

            for item in js_obj["shapes"]: # Check whether the defect has already been transferred
                if item["label"] == "0": left = item["points"][0][0]
                elif item["label"] == "2": defects.append([item["points"][0][0], item["points"][1][0]])
            f.close()
            
        if len(defects) == 0: return self.generate_negative(json_file)
        
        # Set the length of the box edge
        length = np.random.randint(min_len, max_len)
        
        # Set the top left point as the anchor
        id = np.random.randint(0,len(defects))
        xmin, xmax = defects[id] 
        x0 = np.random.randint(xmax-length, xmin)
        y0 = np.random.randint(0, img_h-length)
        x1 = x0 + length
        y1 = y0 + length
        
        return [x0, y0, x1, y1], 1
                
    def generate_negative(self, json_file):
        min_len = self.min_len
        max_len = self.max_len
        img_w   = self.img_w
        img_h   = self.img_h
        left, xmin, xmax = 0, -1, -1
        
        with open(json_file, "r", encoding="utf-8") as f:
            js_obj = json.load(f)
            defects = []

            for item in js_obj["shapes"]: # Check whether the defect has already been transferred
                if item["label"] == "0": left = item["points"][0][0]
                elif item["label"] == "2": defects.append([item["points"][0][0], item["points"][1][0]])
            f.close()

        # Set the length of the box edge
        length = np.random.randint(min_len, max_len)
        
        # Set the top left point as the anchor
        if len(defects) == 0: x0 = np.random.randint(left, img_w-length)
        else: 
            neg_blocks = self.generate_negative_block(left, defects)
            print(neg_blocks)
            print()
            
            id = np.random.randint(0,len(neg_blocks))
            xmin, xmax = neg_blocks[id]
            x0 = np.random.randint(xmin, xmax-length)
        
        y0 = np.random.randint(0, img_h-length)
        x1 = x0 + length
        y1 = y0 + length
        
        return [x0, y0, x1, y1], 0
        
    def generate_negative_block(self, left, defects):
        neg_blocks, s = [], left
        for i in range(len(defects)):
            blk_len = defects[i][0] - s
            if blk_len > self.max_len: neg_blocks.append([s, defects[i][0]])
            s = defects[i][1]
        if self.img_w-s-1 > self.max_len: neg_blocks.append([s, self.img_w-1])
        
        return neg_blocks
        
if __name__ == "__main__":
    import cv2
    import glob as gb
    from matplotlib import pyplot as plt
    
    img_path  = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"
    json_path = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src"
    img_list = gb.glob(img_path+r"/*.png")
    data = AITEXData(json_path)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        json_file = os.path.join(json_path, fname+".json")
        
        pbox, plabel = data.generate_positive(json_file)
        nbox, nlabel = data.generate_negative(json_file)
        
        img = cv2.imread(img_file, -1)
        pimg = img[pbox[1]:pbox[3],pbox[0]:pbox[2]]
        nimg = img[nbox[1]:nbox[3],nbox[0]:nbox[2]]
        
        if plabel == 1:
            print(img_file)
            print("positive box:", pbox)
            print("negative box:", nbox)
            print()
            
            plt.subplot(1,2,1), plt.imshow(pimg, cmap="gray"), plt.title("Positive label "+str(plabel))
            plt.subplot(1,2,2), plt.imshow(nimg, cmap="gray"), plt.title("negative label "+str(nlabel))
            plt.show()