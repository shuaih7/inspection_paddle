import os, cv2
import numpy as np


def image_crop_generator(img_file, size=224, return_dim=True):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    hdim, wdim = int(h/size), int(w/size)
    image = cv2.resize(image, (wdim*size,hdim*size), cv2.INTER_LINEAR)
    
    images = []
    for j in range(hdim):
        for i in range(wdim): images.append(image[j*size:(j+1)*size,i*size:(i+1)*size])
        
    images = np.array(images, dtype=np.float32) / 255.0
    images = np.expand_dims(np.squeeze(images),1)
    
    if return_dim: return images, [hdim, wdim]
    else: return images