import os
import cv2
import sys
import glob as gb
sys.path.append(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\projects\Part_Number\PaddleOCR-release-1.1")
from paddleocr import PaddleOCR
from utils import draw_polylines, draw_initial_point
from matplotlib import pyplot as plt

# For single image test
"""
ocr = PaddleOCR(use_angle_cls=True, lang="en") # need to run only once to download and load model into memory
img_path = './sample_flip.jpg'
result = ocr.ocr(img_path, rec=False, det=True, cls=True)
print(result)

image = cv2.imread(img_path, -1)
image = draw_polylines(image, result)
image = draw_initial_point(image, result)
plt.imshow(image), plt.show()
cv2.imwrite("result_flip.jpg", image)
"""

# For image dir test
ocr = PaddleOCR(use_angle_cls=True, lang="en")
img_dir = r"E:\Projects\Part_Number\dataset\20210111"
img_list = gb.glob(img_dir + r"/*.bmp")
if len(img_list) == 0: 
    raise Exception("There are no image file in the specified folder.")
    
for img_path in img_list:
    result = ocr.ocr(img_path, rec=False, det=True, cls=True)
    #print(result)

    image = cv2.imread(img_path, -1)
    image = draw_polylines(image, result)
    plt.imshow(image), plt.title("Detection result"), plt.show()
    #cv2.imwrite("result_flip.jpg", image)


"""
for line in result:
    print(line)
 
# ????
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save("./results/sample_result.jpg")
"""
