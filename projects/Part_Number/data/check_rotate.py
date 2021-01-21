import os
import sys
import cv2
import time
import glob as gb
import numpy as np


SUFFIX = [".bmp", ".png", ".jpg", ".tif"]

def load_image_files(img_folder, key=None, suffix=None):
    img_list = []
    
    if suffix is None:
        for suf in SUFFIX:
            img_list += gb.glob(img_folder + r"/*"+suf)
    else:
        img_list = gb.glob(img_folder + r"/*"+suffix)
        
    if key == "time":
        img_list = sort(img_list, key=time.getmtime)
    
    return img_list
    
    
def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img


def check_folder(img_folder, save_folder="", save_suffix=None):
    img_list = load_image_files(img_folder)
    
    for img_file in img_list:
        _, filename = os.path.split(img_file)
        fname, suffix = os.path.splitext(filename)
        if save_suffix is None: save_suffix = suffix
        save_name = os.path.join(save_folder, fname + save_suffix)
        
        image = cv2.imread(img_file, -1)
        cv2.namedWindow(filename,0);
        cv2.resizeWindow(filename, 640, 480);
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        
        dir = int(input("Direction id:"))
        
        if dir == 2:
            image = cv2.flip(image, -1)
        elif dir == 1:
            image = rotate_bound(image, -90)
        elif dir == 3:
            image = rotate_bound(image, 90)
        """
        cv2.namedWindow("Result",0);
        cv2.resizeWindow("Result", 640, 480);
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        """
        cv2.imwrite(save_name, image)
        print("Image file saved to", save_name)
        

def switch_suffix(img_folder, suffix=".png", save_folder=None, orig_suffix=None):
    img_list = load_image_files(img_folder, key=None, suffix=orig_suffix)
    
    for img_file in img_list:
        print("Processing image file", img_file, "...")
        path, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        if save_folder is None: save_folder = path
        
        image = cv2.imread(img_file, -1)
        cv2.imwrite(os.path.join(save_folder, fname+suffix), image)
        
        
if __name__ == "__main__":
    img_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA"
    save_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA_rotate"
    save_suffix = ".png"
    # check_folder(img_folder, save_folder, save_suffix)
    switch_suffix(save_folder, ".png", save_folder = r"E:\Projects\Part_Number\dataset\20210113\casting_sideA_rect")