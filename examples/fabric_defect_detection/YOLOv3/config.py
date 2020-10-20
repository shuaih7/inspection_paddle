import os, paddle
from yolo_v3 import get_yolo
from data import get_reader
import paddle.fluid as fluid
from params import train_parameters


def build_program_with_feeder(main_prog, startup_prog, place):
    max_box_num = train_parameters['max_box_num']
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')
        
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=main_prog)
        reader = get_reader(train_parameters["img_dir"], train_parameters["ann_dir"], train_parameters["train_list"], max_box_num=train_parameters["max_box_num"])
        #reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size']) # Do not need shuffle since the items are already shuffled
        reader = paddle.batch(reader, train_parameters['train_batch_size'])
                                     
        with fluid.unique_name.guard():
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
            outputs = model.net(img)
        return feeder, reader, get_loss(model, outputs, gt_box, gt_label)


def get_optimizer(learning_rate=0.001, decay_rate=0.00005):
    optimizer = fluid.optimizer.SGDOptimizer(
        learning_rate=learning_rate,
        regularization=fluid.regularizer.L2Decay(decay_rate))

    return optimizer
    
    
def get_loss(model, outputs, gt_box, gt_label, learning_rate=0.001, decay_rate=0.00005):
    losses = []
    downsample_ratio = model.get_downsample_ratio()
    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                anchors=model.get_anchors(),
                anchor_mask=model.get_anchor_mask()[i],
                class_num=model.get_class_num(),
                ignore_thresh=train_parameters["ignore_thresh"],
                use_label_smooth=False,  # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                downsample_ratio=downsample_ratio)
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer = get_optimizer(learning_rate=learning_rate, decay_rate=decay_rate)
        optimizer.minimize(loss)
        return loss