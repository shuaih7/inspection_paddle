### Script for loading variable test
### Reference: https://aistudio.baidu.com/aistudio/projectdetail/474305

import os, sys
#sys.path.append(r"C:\Users\shuai\Documents\PaddleDetection-release-0.4")

import paddle
from paddle import fluid
#from ppdet.modeling.backbones.resnet import ResNet
from resnet import ResNet


train_core1 = { 
    # 输入size大小,建议保持一致
    "input_size": [3, 224, 224], 
    # 使用这个项目是要把目标图像分成多少种类
    "class_dim": 12,  # 分类数,
    # 主要修改的超参学习率,可以试试0.001,0.005,0.01,0.02之类的
    "learning_rate":0.0002,
    # 建议使用GPU,否则训练时间会很长
    "use_gpu": True,
    # 前期的训练轮数,你可以尝试着去增加试试看
    "num_epochs": 5, #训练轮数
    # 当达到想要的准确率就立刻保存下来当时的模型
    "last_acc":0.4
} 
"""
train=open('train_split_list.txt','w')
val=open('val_split_list.txt','w')
with open('data/data10954/train_list.txt','r') as f:
    #with open('data_sets/cat_12/train_split_list.txt','w+') as train:
    lines=f.readlines()
    for line in lines:
        if random.uniform(0, 1) <= train_ratio: 
            train.write(line) 
        else: 
            val.write(line)

train.close()
val.close()
"""
#train_reader = paddle.batch(reader.train(), batch_size=32)
#test_reader = paddle.batch(reader.val(), batch_size=32)


# Constructing the modeling
image=fluid.layers.data(name='image',shape=train_core1["input_size"],dtype='float32')
label=fluid.layers.data(name='label',shape=[1],dtype='int64')

model = ResNet(layers=50)
net = model.net(input=image, class_dim=10)

print(type(net))


