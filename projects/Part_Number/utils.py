import os
import sys
import cv2
import numpy as np


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
    
    
def draw_initial_point(image, polylines, radius=3, color=(0,0,255), thickness=5):
    for line in polylines:
        point = (int(line[0][0]), int(line[0][1]))
        image = cv2.circle(image, point, radius=radius, color=color, thickness=thickness)
    return image
    
    
def flip_image_top_bottom(image):
    return cv2.flip(image, 0)


if __name__ == "__main__":
    img_file = "sample.jpg"
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    flip_image = flip_image_top_bottom(image)
    cv2.imwrite("sample_flip.jpg", flip_image)