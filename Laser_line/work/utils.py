import numpy as np
import logging
import os


def label2img(lab):
    x, y = lab.shape
    label_img = np.zeros((x, y, 3))
    label_dic = {1: [0.0, 0.0, 255.0], 2: [255.0, 0.0, 0.0], 3: [0.0, 255.0, 0.0], 4: [255.0, 255.0, 0.0],
                 5: [255.0, 0.0, 255.0], 6: [0.0, 255.0, 255.0], 7: [255.0, 255.0, 255.0], 0: [0.0, 0.0, 0.0],
                 8: [100, 100, 100]}
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            key = lab[i][j]
            if key < 0:
                key = 8
            label_img[i][j][0] = label_dic[key][0]
            label_img[i][j][1] = label_dic[key][1]
            label_img[i][j][2] = label_dic[key][2]
    return label_img


def init_log_config(log_nm):
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, log_nm)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
