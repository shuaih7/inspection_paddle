# https://blog.csdn.net/qq_36834959/article/details/79958446

import os, sys
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from matplotlib import pyplot as plt
#from lxml.etree import Element, SubElement, tostring, ElementTree


class ImageRandomDistort(object):
    def __init__(self):
        pass
        
    def random_crop(self, img, area=[]):
        pass
        
    def random_brightness(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(val)
        
    def random_contrast(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(val)
        
    def random_sharpness(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random.uniform(lower, upper)
        return ImageEnhance.Sharpness(img).enhance(val)
        
    def random_color(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(val)
        
    def random_rotate(self, img, xml_tree, val=None, pos=0.5): # Note: only for image while img_h == img_w
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: 
            kind = np.random.randint(0,2)
            if kind == 0: img = img.rotate(90)  # Todo: Need to consider about the label
            if kind == 1: img = img.rotate(270) # Yodo: Need to consider about the label
        return img
        
    def random_flip(self, img, xml_tree, val=None, pos=0.5):
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: 
            kind = np.random.randint(0,3)
            if kind in [1,3]: img, xml_tree = self.random_flip_left_right(img, xml_tree, val=0)      
            if kind in [2,3]: img, xml_tree = self.random_flip_top_bottom(img, xml_tree, val=0) 

        return img, xml_tree
        
    def random_flip_left_right(self, img, xml_tree, val=None, pos=0.5):
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: 
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            root = xml_tree.getroot()
            
            # Get the image width and height
            width, height = 0,0 
            for node in root.iter():
                if node.tag == "size":
                    for item in node.iter():
                        if item.tag == "width": width = int(item.text)
                        elif item.tag == "height": height = int(item.text)
            assert width > 0 and height > 0
            
            # Left right flip the image and the corresponding bbxs
            for node in root.iter():
                if node.tag == "object":
                    xs = []
                    for item in node.iter():
                        if item.tag in ["xmin", "xmax"]: xs.append(str(width - int(item.text)))
                    if len(xs) == 0: break
                    
                    for item in node.iter():
                        if item.tag == "xmin":   item.text = str(min(xs))
                        elif item.tag == "xmax": item.text = str(max(xs))
            
        return img, xml_tree
        
    def random_flip_top_bottom(self, img, xml_tree, val=None, pos=0.5):
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: 
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            root = xml_tree.getroot()
            
            # Get the image width and height
            width, height = 0,0 
            for node in root.iter():
                if node.tag == "size":
                    for item in node.iter():
                        if item.tag == "width": width = int(item.text)
                        elif item.tag == "height": height = int(item.text)
            assert width > 0 and height > 0
            
            # Top bottom flip the image and the corresponding bbxs
            for node in root.iter():
                if node.tag == "object":
                    ys = []
                    for item in node.iter():
                        if item.tag in ["ymin", "ymax"]: ys.append(str(height - int(item.text)))
                    if len(ys) == 0: break
                    
                    for item in node.iter():
                        if item.tag == "ymin":   item.text = str(min(ys))
                        elif item.tag == "ymax": item.text = str(max(ys))
            
        return img, xml_tree
        
    def random_invert(self, img, val=None, pos=0.3):
        if val is None: val = np.random.uniform(0, 1)
        if val < pos: return ImageOps.invert(img)
        else: return img
        
    def crop_and_resize(self, img, box=None, size=None, resample=Image.BILINEAR):
        if box is not None: img = img.crop(box)
        if size is not None: img = img.resize(size, resample=resample)
        return img
        
        
def pillow_test(img_file):
    img = Image.open(img_file)
    #img_sub = img.crop([128,128,512,512])
    #plt.imshow(img_sub, cmap="gray")
    
    aug = ImageRandomDistort()
    img_bright   = aug.random_brightness(img, 1.5)
    img_contrast = aug.random_contrast(img, 1.5)
    img_flip     = aug.random_rotate(img, 0)
    img_invert   = aug.random_invert(img, 0)
    plt.subplot(1,5,1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
    plt.subplot(1,5,2), plt.imshow(img_bright, cmap="gray"), plt.title("Bright Image")
    plt.subplot(1,5,3), plt.imshow(img_contrast, cmap="gray"), plt.title("Contrast Image")
    plt.subplot(1,5,4), plt.imshow(img_flip, cmap="gray"), plt.title("Flip Image")
    plt.subplot(1,5,5), plt.imshow(img_invert, cmap="gray"), plt.title("Invert Image")
    plt.show()
    
     

if __name__ == "__main__":
    img_file = "sample.png"
    pillow_test(img_file)

