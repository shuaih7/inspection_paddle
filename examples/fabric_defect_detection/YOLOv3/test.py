import os

path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
dir, folder = os.path.split(path)
print(dir)
print(folder)

label = [1,2,3,4,5]
bbxs = [1,2,3,4,5]
assert len(label) == len(bbxs)