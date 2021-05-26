import os
import numpy as np

data_path = "data/iccv09Data/"


def file_list(dirname, ext='.jpg'):
    """获取目录下所有特定后缀的文件
    @param dirname: str 目录的完整路径
    @param ext: str 后缀名, 以点号开头
    @return: list(str) 所有子文件名(不包含路径)组成的列表
    """
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))


image_path = os.path.join(data_path, 'images/')
label_path = os.path.join(data_path, "labels/")
# 对文件列表进行打乱
image_list = file_list(image_path)
np.random.shuffle(image_list)
# 取其中80% 作训练集
total_num = len(image_list)
train_num = int(total_num * 0.8)
# 创建训练文件
with open("data/train_list.txt", 'w') as train_f:
    for i in range(train_num):
        train_data_path = os.path.join('images/', image_list[i])
        train_label_path = os.path.join('labels/', image_list[i])
        train_label_path = train_label_path.replace("jpg", "regions.txt")
        lines = train_data_path + '\t' + train_label_path + '\n'
        train_f.write(lines)
with open("data/eval_list.txt", 'w') as eval_f:
    for i in range(train_num, total_num):
        eval_data_path = os.path.join('images/', image_list[i])
        eval_label_path = os.path.join('labels/', image_list[i])
        eval_label_path = eval_label_path.replace("jpg", "regions.txt")
        lines = eval_data_path + '\t' + eval_label_path + '\n'
        eval_f.write(lines)
