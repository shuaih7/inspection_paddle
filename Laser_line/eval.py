import paddle.fluid as fluid
import work
import numpy as np
from config import agrs

num_classes = agrs['num_classes']
image_shape = agrs['image_shape']
use_gpu = agrs['use_gpu']
train_path = agrs['train_model_dir']
eval_path = agrs['eval_file_path']
data_dir = agrs['data_dir']

deeplabv3p = work.DeepLabV3p(agrs)
deeplabv3p.is_train = False


def mean_iou(pred, label):
    label = fluid.layers.elementwise_min(
        label, fluid.layers.assign(np.array(
            [num_classes], dtype=np.int32)))
    label_ignore = (label == num_classes).astype('int32')
    label_nignore = (label != num_classes).astype('int32')

    pred = pred * label_nignore + label_ignore * num_classes

    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
    return miou, wrong, correct


sp = fluid.Program()
tp = fluid.Program()
batch_size = 1

with fluid.program_guard(tp, sp):
    img = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=image_shape, dtype='int32')

    logit = deeplabv3p.net(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    miou, out_wrong, out_correct = mean_iou(pred, label)

tp = tp.clone(True)
fluid.memory_optimize(
    tp, print_log=False,
    skip_opt_set=set([pred.name, miou, out_wrong, out_correct]),
    level=1
)

place = fluid.CPUPlace()
if use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

fluid.io.load_persistables(exe, train_path, tp)
file_list = []
with open(eval_path, 'r') as f:
    for line in f.readlines():
        lines = line.strip()
        file_list.append(lines)
sum_iou = 0
all_correct = np.array([0], dtype=np.int64)
all_wrong = np.array([0], dtype=np.int64)
step = 1
mean_result = []
for imgs, labs in work.custom_batch_reader(batch_size, work.custom_reader(file_list, data_dir, mode='train')):
    imgs = np.array(imgs)
    labs = np.array(labs)
    imgs = imgs.transpose([0, 3, 1, 2])
    imgs = imgs.astype(np.float32)
    labs = labs.astype(np.int32)
    result = exe.run(tp,
                     feed={'img': imgs,
                           'label': labs},
                     fetch_list=[pred, miou, out_wrong, out_correct])
    wrong = result[2][:-1] + all_wrong
    right = result[3][:-1] + all_correct
    all_wrong = wrong.copy()
    all_correct = right.copy()
    mp = (wrong + right) != 0
    miou2 = np.mean(right[mp] * 1.0 / (right[mp] + wrong[mp]))
    mean_result.append(miou2)
    step += 1

print('eval done! total number of image is {}, mean iou: {}'.format(str(step), str(np.mean(mean_result))))
