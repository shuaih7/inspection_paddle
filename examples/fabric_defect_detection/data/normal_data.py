import os, cv2, json
import numpy as np


class NormalData(object):
    def __init__(self, mask_path):
        self.mask_path = mask_path
        
    def generate(self, fname, val=None, pos=0.6): # pos: posibility for generating the positive sample
        mask_file = os.path.join(self.mask_path, fname+"_json_mask.png")
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: return self.generate_positive(mask_file)
        else: return self.generate_negative(mask_file)
        
    def generate_positive(self, mask_file):
        min_len = 256
        max_len = 512 # width, height of the mask image
        pixel_thres = 1000
        
        # Set the length of the box edge
        length = np.random.randint(min_len, max_len) 
        
        # Set the top left point as the anchor
        mask = cv2.imread(mask_file, -1)
        mask[mask>0] = 1 # Normalization
        mask_h, mask_w = mask.shape[:2]
        x0 = np.random.randint(0, mask_w-length)
        y0 = np.random.randint(0, mask_h-length)
        x1 = x0 + length
        y1 = y0 + length
        
        metric = mask[y0:y1,x0:x1]
        if metric.sum() > pixel_thres: return [x0, y0, x1, y1], 1
        else: return [x0, y0, x1, y1], 0
      
    def generate_negative(self, mask_file):
        min_len = 180
        mask_h  = 512
        
        temp_name, _ = os.path.splitext(mask_file)   
        json_file = temp_name + ".json"
        
        with open(json_file, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
            xmin, xmax, max_len = json_obj["xmin"], json_obj["xmax"], json_obj["length"]
            f.close()
        
        if max_len <= min_len: return self.generate_positive(mask_file)
        
        # Set the length of the box edge
        length = np.random.randint(min_len, max_len) 
        
        # Set the top left point as the anchor
        x0 = np.random.randint(xmin, xmax-length)
        y0 = np.random.randint(0, mask_h-length)
        x1 = x0 + length
        y1 = y0 + length
        
        return [x0, y0, x1, y1], 0
        
    def generate_negative_block(self, mask_file):
        mask = cv2.imread(mask_file, -1)#.astype(np.uint16)
        mask_h, mask_w = mask.shape[:2]
        #mask[mask>0] = 1
        mask_ysum = mask.sum(axis=0)
        json_obj = {}
        
        xmin, xmax, length = 0, 0, 0
        for i in range(mask_w-1):
            if mask_ysum[i]>0 and mask_ysum[i+1]==0: xmin = i+1
            if (mask_ysum[i]==0 and mask_ysum[i+1]>0) or (i==mask_w-2 and mask_ysum[-2]==0): 
                if i==mask_w-2 and mask_ysum[-1]==0: xmax = i + 1
                else: xmax = i
                
                if xmax - xmin > length: 
                    length = xmax - xmin
                    json_obj["xmin"] = xmin
                    json_obj["xmax"] = xmax
                    json_obj["length"] = length
        
        temp_name, _ = os.path.splitext(mask_file)   
        json_file = temp_name + ".json"
        with open(json_file, "w", encoding="utf-8") as f:
            res = json.dumps(json_obj, indent=4)
            f.write(res)
            f.close()
        
        """        
        xmin, xmax = json_obj["xmin"], json_obj["xmax"]
        print("xmin =",xmin, "xmax =",xmax, "length =",length)
        mask[253:258,xmin:xmax] = 255
        cv2.imshow("image", mask)
        cv2.waitKey(0)
        """
        
 
if __name__ == "__main__":
    import glob as gb
    from matplotlib import pyplot as plt
    
    img_path = r"E:\Projects\Fabric_Defect_Detection\oih-fabric-detect-detection\dataset\oih_stitch_skipping\image"
    mask_path = r"E:\Projects\Fabric_Defect_Detection\oih-fabric-detect-detection\dataset\oih_stitch_skipping\mask"
    #mask_list = gb.glob(mask_path+r"/*.png")
    img_list = gb.glob(img_path+r"/*.png")
    data = NormalData(mask_path)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        mask_file = os.path.join(mask_path, fname+"_json_mask.png")
        
        pbox, plabel = data.generate_positive(mask_file)
        nbox, nlabel = data.generate_negative(mask_file)
        
        img = cv2.imread(img_file, -1)
        pimg = img[pbox[1]:pbox[3],pbox[0]:pbox[2]]
        nimg = img[nbox[1]:nbox[3],nbox[0]:nbox[2]]
        print("positive length:", pbox[2]-pbox[0])
        print("negative length:", nbox[2]-nbox[0])
        print()
        
        plt.subplot(1,2,1), plt.imshow(pimg, cmap="gray"), plt.title("Positive label "+str(plabel))
        plt.subplot(1,2,2), plt.imshow(nimg, cmap="gray"), plt.title("negative label "+str(nlabel))
        plt.show()
        
        
        
    
        
        
        