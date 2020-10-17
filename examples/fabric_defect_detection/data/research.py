import os
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

img_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train\MER-502-79U3M(NR0190090349)_2020-10-13_13_35_29_260-25.bmp"

img = Image.open(img_file)

box = [800, 500, 1800, 1500]
img = img.crop(box)
img_en = ImageEnhance.Contrast(img).enhance(1.8)

plt.subplot(1,2,1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
plt.subplot(1,2,2), plt.imshow(img_en, cmap="gray"), plt.title("Enhanced Image")
plt.show()