import os, sys
import numpy as np

mask = (0,1,2)

anchors = [(188,15), (351,16), (351,30)]
anchors = [anchors[i] for i in mask]
anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])

print(anchors_tensor)


grid_w, grid_h = 11, 11
col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
grid = np.concatenate((col, row), axis=-1)

print(grid.shape)
#print(grid)


