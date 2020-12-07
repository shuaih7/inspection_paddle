import os, cv2, paddle
import numpy as np
import paddle.fluid as fluid

"""
def reader():
    for i in range(10):
        yield i
        
batch_reader = fluid.io.batch(reader, batch_size=2)

for data in batch_reader():
    print(data)
"""

txt_file = r"train.txt"
pos_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Train\pos"
neg_path = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Train\neg"

txt_file_valid = r"test.txt"
pos_path_valid = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\pos"
neg_path_valid = r"E:\Projects\Engine_Inspection\VGG16CAM\INRIAPerson\Test\neg"


def train_generator():
    f = open(txt_file, "r")
    lines = f.readlines()
    
    for l in lines:
        fname, label = l.replace("\n", ""), 0
        if fname[:3] == "pos":   fname, label = os.path.join(pos_path, fname[4:]), 1
        elif fname[:3] == "neg": fname, label = os.path.join(neg_path, fname[4:]), 0
        
        image = cv2.imread(fname, -1).astype(np.float32)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Reshape the image into [c, h, w]
        if image.shape[-1] == 3: image = np.stack((image[:,:,0],image[:,:,1],image[:,:,2]), axis=0)
        else: image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        
        yield [image, label]
        
        
def valid_generator():
    f = open(txt_file_valid, "r")
    lines = f.readlines()
    
    for l in lines:
        fname, label = l.replace("\n", ""), 0
        if fname[:3] == "pos":   fname, label = os.path.join(pos_path_valid, fname[4:]), 1
        elif fname[:3] == "neg": fname, label = os.path.join(neg_path_valid, fname[4:]), 0
        
        image = cv2.imread(fname, -1).astype(np.float32)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Reshape the image into [c, h, w]
        if image.shape[-1] == 3: image = np.stack((image[:,:,0],image[:,:,1],image[:,:,2]), axis=0)
        else: image = np.expand_dims(np.squeeze(image), 0)
        image /= 255.0  # Normalization
        
        yield [image, label]
    

if __name__ == "__main__":
    #reader(txt_file, pos_path=pos_path, neg_path=neg_path)
    batch_reader = fluid.batch(train_generator(), batch_size=16)
    print(type(batch_reader))

    #for batch_id, data in enumerate(train_reader()): pass
    