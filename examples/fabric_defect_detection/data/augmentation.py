# https://blog.csdn.net/qq_36834959/article/details/79958446

import os, sys, cv2
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt


class ImageRandomDistort(object):
    def __init__(self):
        pass
        
    def random_crop(self, img, area=[]):
        pass
        
    def random_brightness(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random_uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(val)
        
    def random_contrast(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random_uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(val)
        
    def random_sharpness(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random_uniform(lower, upper)
        return ImageEnhance.Sharpness(img).enhance(val)
        
    def random_color(self, img, val=None, lower=0.9, upper=1.1):
        if val is None: val = np.random_uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(val)
        
    def random_flip_left_right(self, img, val=None, pos=0.5):
        if val is None: val = np.random_uniform(0, 1)
        if val < pos: return img.transpose(Image.FLIP_LEFT_RIGHT)
        else: return img
        
    def random_flip_top_bottom(self, img, val=None, pos=0.5):
        if val is None: val = np.random_uniform(0, 1)
        if val < pos: return img.transpose(Image.FLIP_TOP_BOTTOM)
        else: return img
        
    def crop_and_resize(self, img, box=None, size=None, resample=Image.BILINEAR):
        if box is not None: img = img.crop(box)
        if size is not None: img = Image.resize(size, resample=resample)
        return img
        
        
def pillow_test(img_file):
    img = Image.open(img_file)
    #img_sub = img.crop([128,128,512,512])
    #plt.imshow(img_sub, cmap="gray")
    
    aug = ImageRandomDistort()
    img_bright   = aug.random_brightness(img, 1.5)
    img_contrast = aug.random_contrast(img, 1.5)
    img_flip     = aug.random_flip_top_bottom(img, 0)
    plt.subplot(1,4,1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
    plt.subplot(1,4,2), plt.imshow(img_bright, cmap="gray"), plt.title("Bright Image")
    plt.subplot(1,4,3), plt.imshow(img_contrast, cmap="gray"), plt.title("Contrast Image")
    plt.subplot(1,4,4), plt.imshow(img_flip, cmap="gray"), plt.title("Flip Image")
    plt.show()
    
     

if __name__ == "__main__":
    img_file = "sample.png"
    pillow_test(img_file)
    #img = cv2.imread(img_file, -1)
    #aug = ImageRandomDistort()
    #img_aug = aug.random_brightness(img_file, 1.1)
    #print(type(img_aug))
    """
    plt.subplot(1,2,1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
    plt.subplot(1,2,2), plt.imshow(img_aug, cmap="gray"), plt.title("Brightness 1.1")
    plt.show()
    """

