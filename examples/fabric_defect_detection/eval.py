import cv2, os, sys
import paddle.fluid as fluid
import numpy as np
from data.image import image_crop_generator, mark_image
from matplotlib import pyplot as plt


use_cuda = False
model_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\saved_model"
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
[infer_program, feed_var_names, target_vars] = (fluid.io.load_inference_model(dirname=model_path, executor=exe))


def evaluate(img_file, size=224, resize=None, scale="down", is_display=True):
    images, dims = image_crop_generator(img_file, size=size, resize=resize, scale="down")
    result = exe.run(program=infer_program, feed={feed_var_names[0] :images}, fetch_list=target_vars)
    
    result = np.argmax(result[0], axis=1)
    if is_display:
        matrix = np.reshape(result, (dims[0], dims[1]))
        res_img = mark_image(img_file, matrix, size=size)
        plt.imshow(res_img)
        plt.show(), plt.title("Prediction Result")

   
if __name__ == "__main__":
    import glob as gb
    """
    # Displaying the result
    img_path = r"E:\Projects\Fabric_Defect_Detection\ThreeGun_1013\sampling_1013_40Hz_bright"
    train_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
    valid_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\valid"

    img_list = gb.glob(img_path + r"/*.bmp")
    train_list = gb.glob(train_path + r"/*.bmp")
    valid_list = gb.glob(valid_path + r"/*.bmp")
    full_list  = train_list + valid_list
    
    file_list = []
    for img_file in full_list:
        _, filename = os.path.split(img_file)
        file_list.append(filename)
    
    index = 1
    for img_file in img_list: 
        _, filename = os.path.split(img_file)
        if filename not in file_list: 
            print("Evaluating image ID:", index)
            evaluate(img_file, size=672, resize=224, scale="down")
            index += 1
    """
    
    # Counting the prediction speed
    import time
    pos_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\pos_valid"
    neg_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\neg_valid"
    pos_list = gb.glob(pos_path+r"/*.png")
    neg_list = gb.glob(neg_path+r"/*.png")
    img_list = pos_list + neg_list
    
    images = []
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image = np.expand_dims(np.squeeze(image),0)
        images.append(image)
    images = np.array(images, dtype=np.float32)
        
    batch_size = 64
    
    start = time.time()
    for i in range(int(images.shape[0]/batch_size)):
        imgs = images[i*batch_size:(i+1)*batch_size]
        result = exe.run(program=infer_program, feed={feed_var_names[0] :imgs}, fetch_list=target_vars)
        if (i+1)*batch_size >= images.shape[0]: break
    end = time.time()
    
    elapsed = (end - start)
    print("The running time is", elapsed, "second(s).")
         
    
    
        
    
        
    
    
    

    
