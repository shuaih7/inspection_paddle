# -*- coding: UTF-8 -*-
"""
模型固化
"""
import paddle.fluid as fluid
# from YOLOv3 import get_yolo
from model.mobilev2 import get_yolo
import config


train_parameters = config.init_train_parameters()


def freeze_model(score_threshold):
    """
    模型固化
    :param score_threshold:
    :return:
    """
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    path = train_parameters['save_model_dir']
    model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='float32')

    boxes = []
    scores = []
    outputs = model.net(image)
    downsample_ratio = model.get_downsample_ratio()
    for i, out in enumerate(outputs):
        box, score = fluid.layers.yolo_box(
            x=out,
            img_size=image_shape,
            anchors=model.get_yolo_anchors()[i],
            class_num=model.get_class_num(),
            conf_thresh=train_parameters['valid_thresh'],
            downsample_ratio=downsample_ratio,
            name="yolo_box_" + str(i))
        boxes.append(box)
        scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
        downsample_ratio //= 2

    pred = fluid.layers.multiclass_nms(
        bboxes=fluid.layers.concat(boxes, axis=1),
        scores=fluid.layers.concat(scores, axis=2),
        score_threshold=score_threshold,
        nms_top_k=train_parameters['nms_top_k'],
        keep_top_k=train_parameters['nms_pos_k'],
        nms_threshold=train_parameters['nms_thresh'],
        background_label=-1,
        name="multiclass_nms")

    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)
    fluid.io.save_inference_model(train_parameters['freeze_dir'], ['image', 'image_shape'], pred, exe, freeze_program, model_filename='__model__', params_filename='params')
    print("freeze end")


if __name__ == '__main__':
    freeze_model(0.5)
