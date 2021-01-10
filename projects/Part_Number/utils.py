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
    
