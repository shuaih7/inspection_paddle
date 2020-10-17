import cv2, os, sys
import paddle.fluid as fluid
import numpy as np
from data.image import image_crop_generator, mark_image
from matplotlib import pyplot as plt


use_cuda = True
model_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\saved_model"
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
[infer_program, feed_var_names, target_vars] = (fluid.io.load_inference_model(dirname=model_path, executor=exe))


def evaluate(img_file, size=224, resize=None, scale="down"):
    images, dims = image_crop_generator(img_file, size=size, resize=resize, scale="down")
    result = exe.run(program=infer_program, feed={feed_var_names[0] :images}, fetch_list=target_vars)
    
    result = np.argmax(result[0], axis=1)
    matrix = np.reshape(result, (dims[0], dims[1]))
    res_img = mark_image(img_file, matrix, size=size)
    plt.imshow(res_img)
    plt.show(), plt.title("Prediction Result")

   
if __name__ == "__main__":
    img_path = r"E:\Projects\Fabric_Defect_Detection\ThreeGun_1013\sampling_1013_40Hz_bright"
    train_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\train"
    valid_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\ThreeGun\valid"
    
    import glob as gb
    img_list = gb.glob(img_path + r"/*.bmp")
    train_list = gb.glob(train_path + r"/*.bmp")
    valid_list = gb.glob(valid_path + r"/*.bmp")
    full_list  = train_list + valid_list
    
    file_list = []
    for img_file in full_list:
        _, filename = os.path.split(img_file)
        file_list.append(filename)
    
    for img_file in img_list[:101]: 
        _, filename = os.path.split(img_file)
        if filename not in file_list: evaluate(img_file, size=672, resize=224, scale="down")
    """
    img_file = r"18_6rpm.jpg"
    evaluate(img_file, scale="down")
    """

    
