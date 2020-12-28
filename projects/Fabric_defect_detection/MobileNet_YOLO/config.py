# -*- coding: UTF-8 -*-
"""
配置参数
"""
import os
import logging
import codecs

train_parameters = {
    "data_dir": r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\train",
    "val_dir": r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\dataset\valid",
    "train_list": "train.txt",
    "eval_list": "valid.txt",
    "use_filter": False,
    "class_dim": -1,
    "label_dict": {},
    "num_dict": {},
    "image_count": -1,
    "continue_train": True,     # 是否加载前一次的训练参数，接着训练
    "pretrained": False,
    "pretrained_model_dir": r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\pretrained_model",
    "save_model_dir": r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\saved_model",
    "model_prefix": "yolo-v3",
    "freeze_dir": r"E:\Projects\Fabric_Defect_Detection\model_dev\v1.1.0\freeze_model",
    "use_tiny": False,          # 是否使用 裁剪 tiny 模型
    "max_box_num": 10,          # 一幅图上最多有多少个目标
    "num_epochs": 100,
    "train_batch_size": 16,      # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉
    "use_gpu": True,
    "yolo_cfg": {
        "input_size": [3, 352, 352],    # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
        # "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], #416
        "anchors": [483, 25, 707, 30, 30, 537],#384
        # "anchors": [8, 10, 12, 23, 25, 18, 23, 47, 48, 35, 45, 92, 89, 69, 120, 152, 287, 251],#320
        "anchor_mask": [[0, 1, 2]]
    },
    "yolo_tiny_cfg": {
        "input_size": [3, 352, 352],
        "anchors": [96, 15, 157, 15, 240, 16, 303, 17, 351, 16, 351, 26],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "nms_top_k": 10,
    "nms_pos_k": 10,
    "valid_thresh": 0.005,
    "nms_thresh": 0.1,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 1.25,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.3,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.6
    },
    "sgd_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [10, 45, 80, 110, 135, 160, 180],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.025, 0.004, 0.001, 0.0005]
    },
    "early_stop": {
        "sample_frequency": 50,
        "rise_limit": 10,
        "min_loss": 0.00000005,
        "min_curr_map": 0.84
    }
}


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    #file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    #label_list = os.path.join(train_parameters['data_dir'], "label_list")
    file_list = train_parameters['train_list']
    label_list = "label_list"
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['num_dict'][index] = line.strip()
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)
    return train_parameters


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
    

if __name__ == "__main__":
    config_matrix = init_train_parameters()
    label_dict = config_matrix["label_dict"]
    
    print([elem for elem in label_dict.keys()])