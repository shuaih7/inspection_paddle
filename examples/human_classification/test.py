import numpy as np

a = np.zeros((4,4), dtype=np.uint8)
b = np.ones((4,4), dtype=np.uint8)
c = np.ones((4,4), dtype=np.uint8)*2

d = np.stack((a,b,c), axis=0)
print(d.shape)
print(d)