import os, sys, paddle
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


def optimizer_sgd_setting():
    """
    SGD Optimizer
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    iters = 1 if iters < 1 else iters
    learning_strategy = train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    #logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))
    learning_rate=fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.SGDOptimizer(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        # learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer, learning_rate


def optimizer_rms_setting():
    """
    RMS Optimizer
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    iters = 1 if iters < 1 else iters
    learning_strategy = train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    #logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))
    learning_rate=fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005),)

    return optimizer, learning_rate


def get_optimizer(learning_rate=0.001, decay_rate=0.00005):
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=learning_rate)

    return optimizer
    
    
def get_loss(model, outputs, gt_box, gt_label, learning_rate=0.0002, decay_rate=0.0):
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
        #optimizer = get_optimizer(learning_rate=learning_rate, decay_rate=decay_rate)
        optimizer, lr = optimizer_rms_setting()
        optimizer.minimize(loss)
        return loss