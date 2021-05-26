import os
import sys
import cv2
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
    
    
def draw_polylines(image, polylines, texts=None, isClosed=True, size=0.6, color=(0,255,0), thickness=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    polylines = np.array(polylines, dtype=np.int32)#.reshape((-1,1,2))
    
    for i, line in enumerate(polylines):
        pt = (line[0][0], line[0][1])
        line = line.reshape((-1,1,2))
        image = cv2.polylines(image, [line], isClosed=isClosed, color=color, thickness=thickness)
        if texts is not None:
            image = cv2.putText(image, texts[i], pt, fontFace=font, 
                fontScale=size, color=color, thickness=max(1,thickness-1))
    return image   
    
    
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
    
    
def rotate_points(points, shape, angle):
    if angle not in [0, 90, -90, 180]:
        raise ValueError("Angle value only support 0, 90, -90, and 180.")
    if angle == 0: return points
     
    img_h, img_w = shape
    points_rotated = []
    
    if angle == 180:
        for pt in points:
            rot_pt = [img_w-pt[0], img_h-pt[1]]
            points_rotated.append(rot_pt)
    elif angle == 90:
        for pt in points:
            rot_pt = [img_h-pt[1], pt[0]]
            points_rotated.append(rot_pt)
    elif angle == -90:
        for pt in points:
            rot_pt = [pt[1], img_w-pt[0]]
            points_rotated.append(rot_pt)
    
    return points_rotated
    
    
def switch_suffix(img_folder, suffix=".png", save_folder=None, orig_suffix=None):
    img_list = load_image_files(img_folder, key=None, suffix=orig_suffix)
    
    for img_file in img_list:
        print("Processing image file", img_file, "...")
        path, filename = os.path.split(img_file)
        fname, _ = os.path.splitext(filename)
        if save_folder is None: save_folder = path
        
        image = cv2.imread(img_file, -1)
        cv2.imwrite(os.path.join(save_folder, fname+suffix), image)
    
    