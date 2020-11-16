import paddle, os, sys
import numpy as np
from paddle import fluid

#input层

image = fluid.layers.data(name='pixel', shape=[1,28,28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

#model
conv1 = fluid.layers.conv2d(input=image, filter_size=5, num_filters=20, stride=1)
relu1 = fluid.layers.relu(conv1)
pool1 = fluid.layers.pool2d(input=relu1, pool_size=2, pool_stride=2)
conv2 = fluid.layers.conv2d(input=pool1, filter_size=5, num_filters=50)
relu2 = fluid.layers.relu(conv2)
pool2 = fluid.layers.pool2d(input=relu2, pool_size=2, pool_stride=2)

predict = fluid.layers.fc(input=pool2, size=10, act='softmax')

#loss
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
batch_acc = fluid.layers.accuracy(input=predict, label=label)
#optimizer
opt = fluid.optimizer.AdamOptimizer()
opt.minimize(avg_cost)
#initialize
#place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
#模型训练之前要对参数进行初始化，且只需执行一次初始化操作
#stratup_program存储模型参数的初始化操作
#main_program存储模型网络结构
exe.run(fluid.default_startup_program())

#初始化后的参数存放在fluid.global_scope()中，可通过参数名从该scope中获取参数

train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=128)
model_save_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\scripts\model"

fluid.contrib.model_stat.summary(fluid.default_main_program())

for epoch_id in range(1):
    for batch_id, data in enumerate(train_reader()):
        img_data = np.array([x[0].reshape([1,28,28]) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).reshape([len(img_data),1]).astype('int64')
        loss, acc = exe.run(fluid.default_main_program(), feed={'pixel':img_data, 'label':y_data}, fetch_list=[avg_cost, batch_acc])
        print("epoch:%d, batch:%d, loss:%.5f, acc:%.5f"%(epoch_id, batch_id, loss, acc))
        
    fluid.io.save_inference_model(dirname=model_save_path, feeded_var_names=['pixel'], target_vars=[predict], executor=exe)

"""
# 多卡训练
from paddle.fluid import compiler
#将构建的Program转换为数据并行模式的Program
compiled_program = compiler.CompiledProgram(fluid.default_main_program())
compiled_program.with_data_parallel(loss_name=avg_cost.name)
"""

#训练

#exe.run(compiled_program(),...)
