import os, cv2, paddle
import numpy as np
from PIL import Image
import paddle.fluid as fluid
from .tg_data import TGData 
from .aitex_data import AITEXData
from .normal_data import NormalData 
from .augmentation import ImageRandomDistort


txt_file = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\train.txt"
txt_file_valid = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\valid.txt"
txt_file_pos_valid = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\valid_pos.txt"
txt_file_neg_valid = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\valid_neg.txt"

mask_path   = r"E:\Projects\Fabric_Defect_Detection\oih-fabric-detect-detection\dataset\oih_stitch_skipping\mask"
json_path   = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
normal_path = r"E:\Projects\Fabric_Defect_Detection\oih-fabric-detect-detection\dataset\oih_stitch_skipping\image_gray"
aitex_path  = r"E:\BaiduNetdiskDownload\fabric_defects\AITEX\NODefect_images\src" 

train_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\pos_neg_train"
pos_valid_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\pos_valid"
neg_valid_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\neg_valid"


def train_generator():
    aug = ImageRandomDistort()
    
    f = open(txt_file, "r")
    lines = f.readlines()
    
    for l in lines:
        fname = l.replace("\n", "")
        img_file = os.path.join(train_path, fname+".png")
        
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image = np.expand_dims(np.squeeze(image), 0) # Reshape the image into [c, h, w]
        image /= 255.0  # Normalization
        
        if fname[0] == "1": label = 1
        else: label = 0
        
        yield [image, label]

"""
def train_generator():
    #normal = NormalData(mask_path)
    #aitex  = AITEXData(json_path)
    tg = TGData(json_path)
    aug = ImageRandomDistort()
    
    f = open(txt_file, "r")
    lines = f.readlines()
    
    for l in lines:
        fname = l.replace("\n", "")

        # if fname[:3] == "MER": 
            # box, label = normal.generate(fname, pos=0.7)
            # img_file   = os.path.join(normal_path, fname+".png")
        # else: 
            # box, label = aitex.generate(fname, pos=0.6)
            # img_file   = os.path.join(aitex_path, fname+".png")

        box, label = tg.generate(fname)
        img_file = os.path.join(train_path, fname+".bmp")
        
        img = Image.open(img_file)
        img = aug.crop_and_resize(img, box=box, size=[224, 224])
        img = aug.random_brightness(img, lower=0.9, upper=1.1)
        img = aug.random_contrast(img, lower=0.9, upper=1.1)
        #img = aug.random_flip(img, pos=0.6)
        #img = aug.random_rotate(img, pos=0.6)
        #img = aug.random_invert(img, pos=0.3)
        
        image = np.array(img, dtype=np.float32) # Cast the data type to float 32
        
        # Reshape the image into [c, h, w]
        image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        #print(fname)
        #print(box)
        #print()
        
        yield [image, label]
"""        
        
def valid_generator():
    normal = NormalData(mask_path)
    aitex  = AITEXData(json_path)
    aug    = ImageRandomDistort()
    
    f = open(txt_file_valid, "r")
    lines = f.readlines()
    
    for l in lines:
        fname = l.replace("\n", "")
        if fname[:3] == "MER": 
            box, label = normal.generate(fname, pos=0.7)
            img_file   = os.path.join(normal_path, fname+".png")
        else: 
            box, label = aitex.generate(fname, pos=0.4)
            img_file   = os.path.join(aitex_path, fname+".png")
        
        img = Image.open(img_file)
        img = aug.crop_and_resize(img, box=box, size=[224, 224])
        img = aug.random_brightness(img, lower=0.7, upper=1.3)
        img = aug.random_contrast(img, lower=0.9, upper=1.1)
        img = aug.random_flip(img, pos=0.6)
        #img = aug.random_rotate(img, pos=0.6)
        img = aug.random_invert(img, pos=0.3)
        
        image = np.array(img, dtype=np.float32) # Cast the data type to float 32
        
        # Reshape the image into [c, h, w]
        image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        #print(fname)
        #print(box)
        #print()
        
        yield [image, label]
        
        
def valid_pos_generator():
    f = open(txt_file_pos_valid, "r")
    lines = f.readlines()
    
    for l in lines:
        fname = l.replace("\n", "")
        img_file = os.path.join(pos_valid_path, fname+".png")
        image = cv2.imread(img_file, -1).astype(np.float32)
        
        # Reshape the image into [c, h, w]
        image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        
        yield [image, 1]
        
        
def valid_neg_generator():
    f = open(txt_file_neg_valid, "r")
    lines = f.readlines()
    
    for l in lines:
        fname = l.replace("\n", "")
        img_file = os.path.join(neg_valid_path, fname+".png")
        image = cv2.imread(img_file, -1).astype(np.float32)
        
        # Reshape the image into [c, h, w]
        image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        
        yield [image, 0]

        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    data = train_generator()
    for i in range(100):
        image, label = next(data)
        plt.imshow(image, cmap="gray"), plt.title(str(label))
        plt.show()