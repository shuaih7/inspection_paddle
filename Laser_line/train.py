import work
import os
import numpy as np
import time
import paddle.fluid as fluid
from config import agrs
import argparse


parser = argparse.ArgumentParser(description='Short sample app')    #创建parser对象

parser.add_argument('--use_gpu', type=int, default=0, required = True, help="0: not use_gpu, 1: use_gpu")
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--continue_train', type=int, default=0, required = True, help="0: not continue_train, 1: continue_train")
parser.add_argument('--paddle_flag', type=int, default=0, required = True,
                    help="0: not use paddle model zoo's pretrained model, 1: use paddle model zoo's pretrained model")

mlh_args = parser.parse_args()

agrs['use_gpu'] = bool(mlh_args.use_gpu)
agrs['continue_train'] = bool(mlh_args.continue_train)
agrs['paddle_flag'] = bool(mlh_args.paddle_flag)

batch_size = agrs['batch_size']
use_gpu = agrs['use_gpu']
train_model_dir = agrs['train_model_dir']
pretrain_model_dir = agrs['pretrained_model_dir']
continue_train = agrs['continue_train']
paddle_flag = agrs['paddle_flag']
num_classes = agrs['num_classes']
weight_decay = agrs['weight_decay']
base_lr = agrs['base_lr']
num_epochs = agrs['num_epochs']
total_step = agrs['total_step']
image_shape = agrs['image_shape']
enable_ce = agrs['enable_ce']

if mlh_args.num_epochs:
    num_epochs = mlh_args.num_epochs

def load_pretrained_params(exe, program, save_model_dir, logger):
    if continue_train and os.path.exists(save_model_dir):
        fluid.io.load_persistables(executor=exe, dirname=save_model_dir, main_program=program)
        logger.info("***************已读入预先训练的模型，并准备继续训练***************")
    else:
        pass


def load_paddle_model(exe, program, paddle_model_dir, logger):
    load_vars = [x for x in program.list_vars()
                 if isinstance(x, fluid.framework.Parameter) and x.name.find('logit') == -1]

    fluid.io.load_vars(exe, dirname=paddle_model_dir, vars=load_vars)
    logger.info("***************已读入预先训练的模型，并准备继续训练***************")


# 将来这里是对模型的实例化
deeplabv3p = work.DeepLabV3p(agrs)

sp = fluid.Program()
tp = fluid.Program()

if enable_ce:
    SEED = 102
    sp.random_seed = SEED
    tp.random_seed = SEED

# 开始了
with fluid.program_guard(tp, sp):
    # 采用同步的方式读取数据
    img = fluid.layers.data(
        name='image', shape=[3] + image_shape, dtype='float32'
    )
    label = fluid.layers.data(name='label', shape=image_shape, dtype='int32')

    logit = deeplabv3p.net(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    loss, mask = work.loss(logit, label, num_classes)
    area = fluid.layers.elementwise_max(
        fluid.layers.reduce_mean(mask),
        fluid.layers.assign(np.array([0.1], dtype=np.float32))
    )
    loss_mean = fluid.layers.reduce_mean(loss) / area
    loss_mean.persistable = True

    opt = work.optimizer_momentum_setting(base_lr=base_lr, total_step=total_step, weight_decay=weight_decay)
    optimize_ops, params_grads = opt.minimize(loss_mean, startup_program=sp)

    for p, g in params_grads:
        g.persistable = True

place = fluid.CPUPlace()
if use_gpu:
    place = fluid.CUDAPlace(0)

train_path = 'data/train_list.txt'
data_dir = 'data/iccv09Data/'
file_list = []
with open(train_path, 'r') as f:
    for line in f.readlines():
        lines = line.strip()
        file_list.append(lines)

exe = fluid.Executor(place)
exe.run(sp)
now = time.strftime('%Y-%m-%d', time.localtime(time.time()))
log_name = 'train_log_' + now
logger = work.init_log_config(log_name)
logger.info("train params:%s", str(agrs))
if paddle_flag:
    load_paddle_model(exe, tp, pretrain_model_dir, logger)
else:
    load_pretrained_params(exe, tp, train_model_dir, logger)
logger.info("***************训练开始***************")
for pass_id in range(num_epochs):
    step_num = 1
    total_time = []
    total_loss = []
    for imgs, labs in work.custom_batch_reader(batch_size, work.custom_reader(file_list, data_dir, mode='train')):
        t1 = time.time()
        imgs = np.array(imgs)
        labs = np.array(labs)
        imgs = imgs.transpose([0, 3, 1, 2])
        imgs = imgs.astype(np.float32)
        labs = labs.astype(np.int32)
        loss = exe.run(tp, feed={'image': imgs, 'label': labs},
                       fetch_list=[loss_mean])
        period = time.time() - t1
        loss = np.mean(np.array(loss))
        if step_num % 10 == 0:
            logger.info("epoch: {0} step: {1} loss:{2} period:{3}".format(
                pass_id, step_num*batch_size, loss, "%2.2f sec" % period))
        step_num += 1
        total_time.append(period)
        total_loss.append(loss)
    logger.info("{0}'s epoch_total_time: {1} && mean_loss: {2}".format(
        pass_id, "%2.2f sec" % sum(total_time), sum(total_loss)/len(total_loss)))
    if pass_id % 10 == 0:
        # 每隔10个epoch 保存一次模型
        logger.info("暂时存储第{0}个epoch的训练结果".format(pass_id))
        fluid.io.save_persistables(dirname=train_model_dir, main_program=tp, executor=exe)
logger.info("***************训练完成***************")
fluid.io.save_persistables(dirname=train_model_dir, main_program=tp, executor=exe)
