import paddle.fluid as fluid
from config import agrs
import work

num_classes = agrs['num_classes']
image_shape = [240, 320]
train_path = agrs['train_model_dir']
infer_save_path = agrs['infer_model_dir']
deeplabv3p = work.DeepLabV3p(agrs)
exe = fluid.Executor(fluid.CPUPlace())
image = fluid.layers.data(name='image', shape=[3] + image_shape, dtype='float32')
pred = deeplabv3p.net(image)

freeze_program = fluid.default_main_program()
fluid.io.load_persistables(exe, train_path, freeze_program)
freeze_program = freeze_program.clone(for_test=True)

fluid.io.save_inference_model(infer_save_path, ['image'], [pred], exe, freeze_program)
