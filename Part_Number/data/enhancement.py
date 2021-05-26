import os
import sys
import cv2
from scipy.ndimage import *
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
from utils import *


class SNPatch():
    '''
    Get SN patches for OCR recognition
    '''
    # offset = (500, 300)  #(r,c)
    # width = 905
    # height = 683
    # row_nbr, col_nbr = 4, 5

    offset = (0, 0)  #(r,c)
    width = 905
    height = 683
    row_nbr, col_nbr = 5, 5

    def __init__(self):
        self.image_patches = []
        self.image_filtered = None
        self.roi = []  # list of r0,c0,r1,c1
        self.set_roi()

    # todo, roi setting for config files
    def set_roi(self):
        ''''''
        for r in range(self.row_nbr):
            for c in range(self.col_nbr):
                r0, r1 = self.offset[0] + self.height * r, self.offset[0] + self.height * (r + 1)
                c0, c1 = self.offset[1] + self.width * c, self.offset[1] + self.width * (c + 1)
                self.roi.append((r0,c0,r1,c1))
                #print("r0,c0,r1,c1: ",r0, c0, r1, c1)

    # todo, multiprocessing
    def __call__(self, image, angle, engine=None, params={}, app=None):
        #self.image_filtered,_ = self.denoise(image,image)
        #return self.get_patches(self.image_filtered)
        #return self.rec_patches(image, engine, params, app)

        # todo , using numba.cuda
        # self.image_filtered = median_filter(image,8)
        # self.image_filtered = cv2.medianBlur(image,7)
        self.image_filtered = image
        return self.rec_patches(self.image_filtered, angle, engine, params, app)

    def get_patches(self,img_filtered):
        
        image_patches = []
        for i,roi in enumerate(self.roi):
            r0,c0,r1,c1 = roi
            img_patch= img_filtered[r0:r1, c0:c1]
            cv2.imwrite(f"output/{i}.png",img_patch)
            image_patches.append(img_patch)
        return image_patches
        
    def rec_patches(self, img_filtered, angle, engine=None, params={}, app=None):
        if engine is None: return []
        if angle not in [0, 90, -90, 180, -180]:
            raise ValueError("Rotate angle only support 0, 90, -90, 180.")
            
        row, col = 0, 0
        offy, offx = self.offset
        results = []
        img_patches = self.get_patches(img_filtered)
        for i, img in enumerate(img_patches):
            cur_result = []
            img = rotate_image(img, angle)
            img_shape = img.shape[:2]
            loc_result = engine.ocr(img, **params)

            for label in loc_result:
                points = rotate_points(label[0], img_shape, -1*angle)
                text = label[1][0].upper()
                confidence = label[1][1]

                r0, c0, r1, c1 = self.roi[i]
                # print("r0,c0,r1,c1:", r0, c0, r1, c1)
                for point in points:
                    point[0] += c0
                    point[1] += r0
                cur_result.append([points, [text, confidence]])
                
            # results += self.merge_results(cur_result)
            results += cur_result

            if app is not None: app.processEvents()

        return results
        
    def merge_results(self, results):
        if len(results) != 2:
            return results
        
        xs, ys = [], []
        for pt in results[0][0] + results[1][0]:
            xs.append(pt[0])
            ys.append(pt[1])
        
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        
        # Clock-wise         
        roi = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        if len(results[0][1][0]) >= len(results[1][1][0]):
            text = results[0][1][0] + results[1][1][0]
        else:
            text = results[1][1][0] + results[0][1][0]
        confidence = (results[0][1][1] + results[1][1][1])/2
        
        mresult = [[roi, [text, confidence]]]
        
        return mresult


if __name__ =="__main__":
    import glob as gb
    patcher = SNPatch()
    
    img_dir = r"E:\Projects\Part_Number\dataset\test"
    save_dir = r"E:\Projects\Part_Number\dataset\test_patches"
    img_list = load_image_files(img_dir)
    
    for img_file in img_list:
        print("Processing image file", img_file, "...")
        image = cv2.imread(img_file, -1)
        img_patches = patcher.get_patches(image)
        
        _, filename = os.path.split(img_file)
        fname, suffix = os.path.splitext(filename)
    
        for i, img in enumerate(img_patches):
            save_name = os.path.join(save_dir, fname+"_"+str(i)+suffix)
            cv2.imwrite(save_name, img)
        
        