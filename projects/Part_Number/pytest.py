import os
import sys
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
    
    
if __name__ == "__main__":
    image_file = r"E:\Projects\Part_Number\baidu\icdar2015\text_localization\ch4_test_images\img_61.jpg"
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    box = np.array([[310, 104], [416, 141], [418, 216], [312, 179]], dtype=np.float32)
    dst_img = get_rotate_crop_image(image, box)
    plt.subplot(1,2,1), plt.imshow(image), plt.title("Original image")
    plt.subplot(1,2,2), plt.imshow(dst_img), plt.title("Cropped image")
    plt.show()
    