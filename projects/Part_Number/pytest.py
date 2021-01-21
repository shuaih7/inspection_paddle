import os
import sys
import json
import random
import numpy as np


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


w = np.zeros(28, dtype=np.uint32)

minx = 2
maxx = 5
w[minx:maxx] = 1
minx = 12
maxx = 15
w[minx:maxx] = 1
minx = 22
maxx = 25
w[minx:maxx] = 1

w_axis = np.where(w == 0)[0]
print(w_axis)
print()

w_regions = split_regions(w_axis)
print(w_regions)
print()

    
