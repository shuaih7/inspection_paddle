# coding=utf-8
import os
import numpy as np

def linear_interpolate(points):
    array = []
    for i in range(len(points)-1):
        pt, pt_next = points[i], points[i+1]
        x1, y1, x2, y2 = pt[0], pt[1], pt_next[0], pt_next[1]
        for j in range(x2 - x1):
            value = int((y2-y1)/(x2-x1)*j+y1)
            array.append(value)
    array.append(points[-1,1])
    return array
    
    
points = np.array([[1,3],[5,12],[8,19]])
array = linear_interpolate(points)
print(array)