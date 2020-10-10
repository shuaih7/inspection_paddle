import cv2, os
import paddle.fluid as fluid
import numpy as np
from data.image import image_crop_generator


model_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\saved_model"
image_path = r"C:\Users\shuai\Documents\GitHub\inspection_paddle\examples\fabric_defect_detection\data\sample.png"

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

[infer_program, feed_var_names, target_vars] = (fluid.io.load_inference_model(dirname=model_path, executor=exe))

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = img[:224,:224]
img = np.array(img, dtype=np.float32) / 255.0
img = np.expand_dims(np.expand_dims(img, 0), 0)

result = exe.run(program=infer_program, feed={feed_var_names[0] :img}, fetch_list=target_vars)
max_id = np.argmax(result[0][0])
print(max_id, result[0][0][max_id])