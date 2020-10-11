import cv2, os
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


def evaluate(img_file, scale="up"):
    images, dims = image_crop_generator(img_file, scale="up")
    result = exe.run(program=infer_program, feed={feed_var_names[0] :images}, fetch_list=target_vars)
    print(result[0])
    
    result = np.argmax(result[0], axis=1)
    matrix = np.reshape(result, (dims[0], dims[1]))
    res_img = mark_image(img_file, matrix, size=224)
    plt.imshow(res_img)
    plt.show(), plt.title("Prediction Result")

   
if __name__ == "__main__":
    img_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\dataset\random_valid_pos\1_51.png" 
    evaluate(img_file)
    
    
