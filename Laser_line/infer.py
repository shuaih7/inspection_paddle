import paddle.fluid as fluid
import cv2
import work
import numpy as np
import os
from config import agrs

num_classes = agrs['num_classes']
image_shape = agrs['image_shape']
infer_save_path = agrs['infer_model_dir']

deeplabv3p = work.DeepLabV3p(agrs)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=infer_save_path, executor=exe)


def read_image(img_path):
    origin = cv2.imread(img_path)
    if origin.shape[0] != image_shape[0] or origin.shape[1] != image_shape[1] or len(origin.shape) != 3:
        print("输入图片的大小不合适，我们需要的是{}，但是输入的是{}".format(str(image_shape+[3]), str(origin.shape)))
        exit()
    img = origin.astype('float32').transpose(2, 0, 1)
    return origin, [img]


train_path = agrs['eval_file_path']
data_dir = agrs['data_dir']
file_list = []
with open(train_path, 'r') as f:
    for line in f.readlines():
        lines = line.strip()
        file_list.append(lines)

test1 = file_list[15].split()
origin, img = read_image(os.path.join(data_dir,test1[0]))
img = np.array(img)
output = exe.run(inference_program,
                 feed={feed_target_names[0]: img},
                 fetch_list=fetch_targets)
output = output[0][0]
output = output.transpose(1, 2, 0)

label_out = np.zeros(image_shape)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        temp = output[i][j]
        temp_index = temp.tolist().index(max(temp))
        label_out[i][j] = temp_index
label_out = work.label2img(label_out)
cv2.imwrite('origin.jpg', origin)
cv2.imwrite('result.jpg', label_out)
print("运行结束，origin为预测图片，result为分割后的图片")
