import os
import sys
import cv2
import numpy as np


def draw_polylines(image, polylines, isClosed=True, color=(0,255,0), thickness=3):
    polylines = np.array(polylines, dtype=np.int32)#.reshape((-1,1,2))
    for line in polylines:
        line = line.reshape((-1,1,2))
        image = cv2.polylines(image, [line], isClosed=isClosed, color=color, thickness=thickness)
    return image   
    
