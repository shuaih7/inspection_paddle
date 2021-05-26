import numpy as np
from config import agrs
import cv2
import os
"""
首先要对数据进行解压和随机划分的处理，来获得以下几个文件
train_list.txt  这个用来存放训练集的路径
val_list.txt   这个用来存放验证集的路径
test_list.txt   这个用来存放测试集的路径

label_color.txt  这个存放标签和颜色对应的字典
"""


def slice_with_pad(a, s, value=0):
    pads = []
    slices = []
    for i in range(len(a.shape)):
        if i >= len(s):
            pads.append([0, 0])
            slices.append([0, a.shape[i]])
        else:
            l, r = s[i]
            if l < 0:
                pl = -l
                l = 0
            else:
                pl = 0
            if r > a.shape[i]:
                pr = r - a.shape[i]
                r = a.shape[i]
            else:
                pr = 0
            pads.append([pl, pr])
            slices.append([l, r])
    slices = list(map(lambda x: slice(x[0], x[1], 1), slices))
    a = a[slices]
    a = np.pad(a, pad_width=pads, mode='constant', constant_values=value)
    return a


def custom_reader(file_list, data_dir, mode):
    """
    数据读取
    :param file_list: 读取数据的路径列表
    :param data_dir:  数据所在的目录路径
    :param mode:      所读取数据的用途
    :return:
    """
    def reader():
        np.random.shuffle(file_list)
        for lines in file_list:
            if mode == 'train' or mode == 'eval':
                # 这里可根据自己的需要进行修改
                image_path, label_path = lines.split()
                image_path = os.path.join(data_dir, image_path)
                label_path = os.path.join(data_dir, label_path)

                # 数据读入进来后，类型为numpy.ndarray
                img = cv2.imread(image_path)

                # 读标签，因为该数据集的标签以txt存储，因此如果存储方式不同可以对这里进行修改
                lab = []
                with open(label_path, 'r') as OfData:
                    for line in OfData.readlines():
                        temp = line.strip().split(' ')
                        temp_lab = []
                        for i in range(len(temp)):
                            if int(temp[i]) == -1:
                                temp_lab.append(8)
                            else:
                                temp_lab.append(int(temp[i]))
                        lab.append(temp_lab)
                lab = np.array(lab)
                if not agrs['data_augmentation_config']['use_augmentation']:
                    yield img, lab
                else:
                    if np.random.rand() > 0.5:
                        range_l = 1
                        range_r = agrs['data_augmentation_config']['max_resize']
                    else:
                        range_l = agrs['data_augmentation_config']['min_resize']
                        range_r = 1
                    random_scale = np.random.rand(1) * (range_r - range_l) + range_l
                    crop_size = int(agrs['data_augmentation_config']['crop_size'] / random_scale)
                    bb = crop_size // 2
                    offset_x = np.random.randint(bb, max(bb + 1, img.shape[0] -
                                                         bb)) - crop_size // 2
                    offset_y = np.random.randint(bb, max(bb + 1, img.shape[1] -
                                                         bb)) - crop_size // 2
                    img_crop = slice_with_pad(img, [[offset_x, offset_x + crop_size],
                                                    [offset_y, offset_y + crop_size]], 128)
                    img = cv2.resize(img_crop, (agrs['image_shape'][0], agrs['image_shape'][1]))
                    label_crop = slice_with_pad(lab, [[offset_x, offset_x + crop_size],
                                                        [offset_y, offset_y + crop_size]],
                                                8)
                    lab = cv2.resize(
                        label_crop, (agrs['image_shape'][0], agrs['image_shape'][1]), interpolation=cv2.INTER_NEAREST)
                    yield img, lab
            if mode == 'test':
                image_path = os.path.join(data_dir, lines)
                yield cv2.imread(image_path)
    return reader()


def custom_batch_reader(batch_size, reader):
    batch_img = []
    batch_label = []
    for img, label in reader:
        if img.shape[0] == 240 and img.shape[1] == 320:
            batch_img.append(img)
            batch_label.append(label)
        if len(batch_img) == batch_size:
            yield batch_img, batch_label
            batch_img = []
            batch_label = []
    if len(batch_img) != 0:
        yield batch_img, batch_label



if __name__ == '__main__':
    import utils
    train_path = '../data/train_list.txt'
    file_list = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            lines = line.strip()
            file_list.append(lines)

    data_dir = '../data/iccv09Data/'
    reader = custom_reader(file_list, data_dir, mode='train')
    for img, label in reader:
        cv2.imshow('image', img)
        label = utils.label2img(label)
        cv2.imshow('label', label)
        cv2.waitKey(0)
