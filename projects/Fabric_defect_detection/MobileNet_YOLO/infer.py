# -*- coding: UTF-8 -*-
"""
模型推理
"""
import os
import config
import cv2
import numpy as np
import time
import paddle.fluid as fluid
import json
from PIL import Image
from PIL import ImageDraw
import shutil
from data import PascalVocXmlParser

train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
path = train_parameters['freeze_dir']  # 'model/freeze_model'
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe, model_filename='__model__', params_filename='params')

"""
def draw_bbox_image(img, boxes, labels, gt=False):
    '''
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    '''
    color = ['red', 'blue']
    if gt:
        c = color[1]
    else:
        c = color[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0))
    return img
"""


def draw_bbox_image(img, boxes, labels, scores, gt=False):
    '''
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    '''
    color = ['red', 'orange']
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        if label == 0: 
            c = color[0]
        else: c = color[1]
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
        draw.text((xmin, ymin), str(score), (255, 255, 0))
    return img
    

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


def infer(image):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, tensor_img, resized_img = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = (time.time() - t1)*1000
    print("predict cost time:{0}".format("%2.2f ms" % period))
    bboxes = np.array(batch_outputs[0])
    #print(bboxes)
    if bboxes.shape[1] != 6:
        # print("No object found")
        return False, [], [], [], [], period
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    return True, boxes, labels, scores, bboxes, period
   

if __name__ == '__main__':
    import sys
    import glob as gb
    
    image_path = r'G:\threeguns_data\20210322-20210326_data_collection\single\single_white_vertical_9gain_300mus_19.8rpm_headlight_on'
    #label_path = r'E:\Projects\Fabric_Defect_Detection\model_proto\MobileNet_YOLO\Fast_YOLO\v1.1\valid'
    save_path  = r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.2.0\dataset\valid_322"
    image_list = gb.glob(image_path + r"/*.bmp")
    total_time = 0.
    
    pvoc = PascalVocXmlParser()
    
    for image_file in image_list:
        img = cv2.imread(image_file)
        _, filename = os.path.split(image_file)
        fname, _ = os.path.splitext(filename)
        save_name = os.path.join(save_path, filename)
        #label_file = os.path.join(label_path, fname+".xml")
        
        flag, box, label, scores, bboxes, period = infer(img)
        total_time += period
        
        if flag:
            img = draw_bbox_image(img, box, label, scores)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            print('Defect detected at image', image_file)
            cv2.imwrite(save_name, img)
        else:
            print(image_path, "No defect detected.")
            shutil.copy(image_file, save_name)
        #print('infer one picture cost {} ms'.format(period))
    
    if len(image_list) > 0:
        average_time = total_time / len(image_list)
        fps = int(1000/average_time)
        print("The avergae processing time for one image is", average_time)
        print("The fps is", fps)
        """
    
    # Check the result of a single image
    image_file = r"E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.1\dataset\valid\valid_gray_1_1600.png"
    img = cv2.imread(image_file)
    flag, box, label, scores, bboxes, period = infer(img)
    print("The boxes are", box)
    print("The scores are", scores)
    """
