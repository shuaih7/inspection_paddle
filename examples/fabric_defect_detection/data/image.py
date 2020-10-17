import os, cv2
import numpy as np


def image_crop_generator(img_file, size=224, resize=None, scale="down"):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    hdim, wdim = int(h/size), int(w/size)
    """
    if scale == "up": 
        if h % size > 0: hdim += 1
        if w % size > 0: wdim += 1
    #image = cv2.resize(image, (wdim*size,hdim*size), cv2.INTER_LINEAR)
    """
    off_y = int((h%(hdim*size))/2)
    off_x = int((w%(wdim*size))/2)
    
    images = []
    for j in range(hdim):
        for i in range(wdim): 
            img = image[off_y+j*size:off_y+(j+1)*size,off_x+i*size:off_x+(i+1)*size]
            if resize is not None: img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        
    images = np.array(images, dtype=np.float32) / 255.0
    if len(images) == 1: images = np.expand_dims(np.expand_dims(np.squeeze(images),0),0)
    else: images = np.expand_dims(np.squeeze(images),1)
    
    return images, [hdim, wdim]
    
    
def mark_image(img_file, matrix, size=224, color=(255,0,0), line_width=5, offset=5):
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    hdim, wdim = matrix.shape
    #image = cv2.resize(image, (wdim*size,hdim*size), cv2.INTER_LINEAR)
    
    off_y = int((h%(hdim*size))/2)
    off_x = int((w%(wdim*size))/2)
    
    for j in range(hdim):
        for i in range(wdim): 
            x1, y1 = off_x+i*size+offset, off_y+j*size+offset
            x2, y2 = off_x+(i+1)*size-offset, off_y+(j+1)*size-offset
            if matrix[j,i] > 0: image = cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
    return image
    
    
    