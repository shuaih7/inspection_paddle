import os
import cv2
import sys
import glob as gb
from PIL import Image
sys.path.append(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Part_Number\PaddleOCR-release-1.1")
#sys.path.append(r"C:\Users\shuai\Documents\PaddleOCR-release-2.0-rc1-0")
from paddleocr import PaddleOCR
from utils import draw_polylines, draw_initial_point
from matplotlib import pyplot as plt
from data.enhancement import SNPatch


def rec_patch(img_path, save_path=None, params={}):
    patcher = SNPatch()
    engine = PaddleOCR(**params)
    img_list = gb.glob(img_path + r"/*.bmp")
    
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        results = patcher(image, engine)
        
        for label in results:
            points = label[0]
            text = label[1][0]
            image = draw_polylines(image, [points], texts=[text], size=1.2, color=(255,0,0))
        plt.imshow(image), plt.show()
        

"""
image = cv2.imread(img_path, -1)
image = draw_polylines(image, result)
image = draw_initial_point(image, result)
plt.imshow(image), plt.show()
cv2.imwrite("result_flip.jpg", image)
"""

params = {
    "use_angle_cls": True,
    "use_gpu": True,
    "gpu_mem": 2048,
    "lang": "ch",
    "cls_model_dir": r"C:\Users\shuai\.paddleocr\1.1\cls",
    "det_model_dir": r"C:\Users\shuai\.paddleocr\1.1\det",
    "rec_model_dir": r"C:\Users\shuai\.paddleocr\1.1\rec\ch"
}


# For image dir test
ocr = PaddleOCR(**params)
img_dir = r"E:\Projects\Part_Number\dataset\test\img"
img_list = gb.glob(img_dir + r"/*.bmp")
if len(img_list) == 0: 
    raise Exception("There are no image file in the specified folder.")
    
for img_path in img_list:
    result = ocr.ocr(img_path, rec=False, det=True, cls=True)
    #print(result)

    image = cv2.imread(img_path, -1)
    image = draw_polylines(image, result)
    plt.imshow(image, cmap="gray"), plt.title("Detection result"), plt.show()
    #cv2.imwrite("result_flip.jpg", image)


