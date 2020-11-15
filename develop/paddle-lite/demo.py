'''
Paddle-Lite light python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
from paddlelite.lite import *


config = MobileConfig()
config.set_model_from_file("mobilenet_v1_opt.nb")

predictor = create_paddle_predictor(config)

image = Image.open('test.jpg')
resized_image = image.resize((224, 224), Image.BILINEAR)
image_data = np.array(resized_image).flatten().tolist()

#image_data = np.array(resized_image, dtype=np.float32)
#image_data = image_data.reshape((1,3,224,224))

input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data(image_data)

predictor.run()

output_tensor = predictor.get_output(0)
print(output_tensor.shape())
#print(output_tensor.float_data()[:10])

out = output_tensor.float_data()[:10]
print(type(out))