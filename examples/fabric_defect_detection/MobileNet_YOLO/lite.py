'''
Paddle-Lite light python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time, cv2, shutil
import config
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from paddlelite.lite import *


train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
place = fluid.CUDAPlace(0) #if train_parameters['use_gpu'] else fluid.CPUPlace()
#exe = fluid.Executor(place)
path = train_parameters['pretrained_model_dir'] 


def get_predictor(model_dir):
    model_config = MobileConfig()
    model_config.set_model_from_file(model_dir)
    
    predictor = create_paddle_predictor(model_config)

    return predictor
    

def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img


def read_image(img):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = img
    img = resize_img(origin, yolo_config["input_size"])
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img
    
    
def infer(input_tensor, img):
    image_data = np.array(img).flatten().tolist()
    input_tensor.set_float_data(image_data)
    
    t1 = time.time()
    predictor.run()
    output_tensor = predictor.get_output(0)
    period = (time.time() - t1)*1000
    
    #print(output_tensor.shape())
    
    return period


if __name__ == '__main__':
    import sys
    import glob as gb
    model_dir = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNetV1_YOLOv3\Fast_YOLO\fast_yolo.nb"
    image_path = r'E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid'
    label_path = r'E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid'
    save_path  = r"E:\Projects\Fabric_Defect_Detection\model_proto\MobileNetV1_YOLOv3\Fast_YOLO\valid_output"
    image_list = gb.glob(image_path + r"/*.png")
    total_time = 0.

    
    predictor = get_predictor(model_dir)
    input_tensor = predictor.get_input(0)
    input_tensor.resize([1, 3, 352, 352])
    
    for image_file in image_list:
        img = cv2.imread(image_file)
        # _, filename = os.path.split(image_file)
        # fname, _ = os.path.splitext(filename)
        # save_name = os.path.join(save_path, filename)
        #label_file = os.path.join(label_path, fname+".xml")
        
        period = infer(input_tensor, img)
        total_time += period
        
        # if flag:
            # img = draw_bbox_image(img, box, scores)
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # print('Defect detected at image', image_file)
            # cv2.imwrite(save_name, img)
        # else:
            # print(image_path, "No defect detected.")
            # shutil.copy(image_file, save_name)
        # #print('infer one picture cost {} ms'.format(period))
        
    average_time = total_time / len(image_list)
    fps = int(1000/average_time)
    print("The avergae processing time for one image is", average_time)
    print("The fps is", fps)