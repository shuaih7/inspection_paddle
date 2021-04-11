# -*- coding:utf-8 -*-
"""
模型训练
"""
from __future__ import division

import os
import cv2
import xml
import time
import shutil
import numpy as np
from PIL import Image
import paddle.fluid as fluid

import config
from model.mobilev2 import get_yolo
from reader import single_custom_reader
from data import PascalVocValidParser, LabelmeValidParser


logger = config.init_log_config()
train_parameters = config.init_train_parameters()
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]

data_dir = train_parameters["data_dir"]
val_dir = train_parameters["val_dir"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
label_dict = train_parameters['num_dict']
label_dict = dict(zip(label_dict.values(), label_dict.keys()))
#test_file_path = os.path.join(data_dir, train_parameters['eval_list'])
test_file_path = train_parameters['eval_list']
val_data = []
print(label_dict)


with open(test_file_path, 'r') as f:
    lines = f.readlines()
    if train_parameters['label_format'].lower() in ['labelme', 'json']:
        valid_parser = LabelmeValidParser(train_parameters)
        label_suffix = '.json'
    elif train_parameters['label_format'].lower() in ['pascalvoc', 'voc', 'xml']:
        valid_parser = PascalVocValidParser(train_parameters)
        label_suffix = '.xml'
    else:
        raise ValueError('Unsupported label format.')
    
    for sample in range(len(lines)):
        fname = lines[sample].replace("\n","")
        image_path = os.path.join(val_dir, fname+".bmp")
        label_path = os.path.join(val_dir, fname+label_suffix)
        
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        gt_label, gt_boxes, difficult = valid_parser(label_path)
        
        if len(gt_label) == 0:
            continue  
        img = cv2.imread(image_path)
        try:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except:
            print('Warning: Could not find image path', image_path)
            continue
        input_w, input_h = img.size[0], img.size[1]
        image_shape = np.array([input_h, input_w], dtype='int32')
        img = img.resize(yolo_config["input_size"][1:], Image.BILINEAR)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
        img -= 127.5
        img *= 0.007843
        img = img[np.newaxis, :]
        val_data.append([img, image_shape, gt_label, gt_boxes, difficult, image_path])


def create_tmp_var(programe, name, dtype, shape):
    """
    create_tmp_var
    :param programe:
    :param name:
    :param dtype:
    :param shape:
    :return:
    """
    return programe.current_block().create_var(name=name, dtype=dtype, shape=shape)


def get_pred(program, fetch_list, data):
    """
    get pred result
    :param program:
    :param fetch_list:
    :param data:
    :return:
    """

    temp_image = data[0]
    temp_image_shape = data[1]
    temp_gt = data[2]
    box = exe.run(program, feed={'img': temp_image,
                                 'image_shape': temp_image_shape[np.newaxis, :]}, fetch_list=fetch_list, return_numpy=False)
    bboxes = np.array(box[0])
    if bboxes.shape[1] != 6:
        labels, scores, boxes = [], [], []
    else:
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')

    pred_list = []
    if len(labels):
        for num in range(len(boxes)):
            pred_list.append([labels[num], scores[num], boxes[num][0],
                              boxes[num][1], boxes[num][2], boxes[num][3]])
    return pred_list


def eval(program, fetch_list, eval_program, eval_fetch_list, eval_feeder):
    """
    eval model
    :param program:
    :param fetch_list:
    :param eval_program:
    :param eval_fetch_list:
    :param eval_feeder:
    :return:
    """
    
    datas = []
    pred = []
    file = []
        
    for data in val_data:

        temp_image = data[0]       # Resized image into with the input shape
        temp_image_shape = data[1] # Original image shape before the reshaping
        temp_gt_label = data[2]
        temp_gt_boxes = data[3]
        temp_difficult = data[4]
        img_path = data[-1]
        
        h = temp_image_shape[0]
        w = temp_image_shape[1]
                
        pred_list = get_pred(program, fetch_list, data)

        if len(pred_list) != 0:
            for box in pred_list:
                box[2] = box[2] / w
                box[4] = box[4] / w
                box[3] = box[3] / h
                box[5] = box[5] / h
                pred.append(box)
        
        pred_list = np.array(pred_list, dtype='float32')
        gt_label = np.array(temp_gt_label, dtype='float32')
        gt_boxes = np.array(temp_gt_boxes, dtype='float32')
        difficult = np.array(temp_difficult, dtype='float32')
        
        datas.append([pred_list, gt_boxes, gt_label, difficult])
        
    pred = np.array(pred)    
    print('pred', pred.shape)
    
    cur_map_v, accum_map_v = exe.run(eval_program, feed=eval_feeder.feed(datas), fetch_list=eval_fetch_list,
                                     return_numpy=True)
        
    return cur_map_v[0], accum_map_v[0] 


def split_by_anchors(gt_box, gt_label, image_size, down_ratio, yolo_anchors):
    """
    将 ground truth 的外接矩形框分割成一个一个小块，类似 seg-link 中的做法
    :param gt_box: 真实外接矩形框，按照 [x, y, w, h] 排布的二维 list，第一维是batch，实际的值都是除以了原始图片尺寸的比例值
    :param gt_label: 真实的类别标签二维 Lise，第一维是batch
    :param image_size: 训练图片的尺寸，[h, w]
    :param down_ratio: int 类型，下采样比例，也暗示现在的特征图被分成多大
    :param yolo_anchors: 当前批次的anchors
    :return:
    """

    gt_box = np.array(gt_box)
    gt_label = np.array(gt_label)
    image_size = np.array(image_size)
    down_ratio = np.array(down_ratio)[0]
    yolo_anchors = np.array(yolo_anchors)
    # print('gt_box shape:{0} gt_label:{1} image_size:{2} down_ratio:{3} yolo_anchors:{4}'
    #       .format(gt_box.shape, gt_label.shape, image_size, down_ratio, yolo_anchors))
    tolerant_ratio = 1.85
    ret_shift_box = np.zeros(gt_box.shape, gt_box.dtype)
    ret_shift_label = np.zeros(gt_label.shape, gt_label.dtype)
    max_bbox = 0

    for n in range(gt_box.shape[0]):
        current_index = 0
        for i in range(gt_box.shape[1]):
            bbox_h = gt_box[n, i, 3] * image_size[0]
            if bbox_h <= 0.1:
                break
            for anchor_h in yolo_anchors[::2]:
                h_d_s = bbox_h / anchor_h
                s_d_h = anchor_h / bbox_h
                if h_d_s <= tolerant_ratio and s_d_h <= tolerant_ratio:
                    ret_shift_box[n, current_index] = gt_box[n, i]
                    ret_shift_label[n, current_index] = gt_label[n, i]
                    current_index += 1
                    if i > max_bbox:
                        max_bbox = i
                    break

    return [ret_shift_box, ret_shift_label]


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
    logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))
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
    logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))
    learning_rate=fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005),)

    return optimizer, learning_rate
    

def optimizer_custom_setting(learning_rate=0.00025):
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)
    return optimizer, learning_rate
    

def build_eval_program_with_feeder(main_prog, startup_prog):
    """
    build eval program
    :param main_prog:
    :param startup_prog:
    :return:
    """
    with fluid.program_guard(main_prog, startup_prog):
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='float32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='float32', lod_level=1)
        pred = fluid.layers.data(name='pred', shape=[6], dtype='float32', lod_level=1)
        
        with fluid.unique_name.guard():
            eval_feeder = fluid.DataFeeder(feed_list=[pred, gt_box, gt_label, difficult], place=place, program=main_prog)
            map_eval = fluid.metrics.DetectionMAP(pred, gt_label, gt_box, difficult,
            train_parameters['class_dim'], overlap_threshold=0.5,
            evaluate_difficult=False, ap_version='integral')
            cur_map, accum_map = map_eval.get_map_var()
    
        return cur_map, accum_map, eval_feeder


def build_program_with_feeder(main_prog, startup_prog, place=None, istrain=True):
    """
    build_program_with_feeder
    :param main_prog:
    :param startup_prog:
    :param place:
    :param istrain:
    :return:
    """
    max_box_num = train_parameters['max_box_num']
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')
        difficult = fluid.layers.data(name='difficult', shape=[max_box_num], dtype='int32')
        with fluid.unique_name.guard():
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'],
                             yolo_config['anchor_mask'])

            outputs = model.net(img)
            if istrain:
                
                feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label, difficult], place=place, program=main_prog)
                reader = single_custom_reader(train_parameters['train_list'],
                                              train_parameters['data_dir'],
                                              yolo_config['input_size'], 'train')
                return feeder, reader, get_loss(model, outputs, gt_box, gt_label, main_prog), outputs
            else:
                boxes = []
                scores = []
                image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')
                downsample_ratio = model.get_downsample_ratio()
                # feeder = fluid.DataFeeder(feed_list=[img, image_shape, gt_label, gt_box, difficult], place=place, program=main_prog)
                # reader = create_eval_reader(train_parameters['eval_list'], train_parameters['data_dir'], yolo_config['input_size'], 'eval')
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
                    score_threshold=0.005,
                    nms_top_k=train_parameters['nms_top_k'],
                    keep_top_k=train_parameters['nms_pos_k'],
                    nms_threshold=train_parameters['nms_thresh'],
                    background_label=-1,
                    name="multiclass_nms")
                
                return pred


def get_loss(model, outputs, gt_box, gt_label, main_prog):
    """
    compute loss
    :param model:
    :param outputs:
    :param gt_box:
    :param gt_label:
    :param main_prog:
    :return:
    """
    losses = []
    downsample_ratio = model.get_downsample_ratio()
    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            if train_parameters['use_filter']:
                ues_tiny = train_parameters['use_tiny']
                yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
                train_image_size_tensor = fluid.layers.assign(np.array(yolo_config['input_size'][1:]).astype(np.int32))
                down_ratio = fluid.layers.fill_constant(shape=[1], value=downsample_ratio, dtype=np.int32)
                yolo_anchors = fluid.layers.assign(np.array(model.get_yolo_anchors()[i]).astype(np.int32))
                filter_bbox = create_tmp_var(main_prog, None, gt_box.dtype, gt_box.shape)
                filter_label = create_tmp_var(main_prog, None, gt_label.dtype, gt_label.shape)
                fluid.layers.py_func(func=split_by_anchors,
                                     x=[gt_box, gt_label, train_image_size_tensor, down_ratio, yolo_anchors],
                                     out=[filter_bbox, filter_label])
            else:
                filter_bbox = gt_box
                filter_label = gt_label
            # print(model.get_anchors())
            # print(model.get_anchor_mask()[i])
            # print(out.shape)
            # print('downsample_ratio', downsample_ratio)
            # print(model.get_class_num())
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=filter_bbox,
                gt_label=filter_label,
                anchors=model.get_anchors(),
                anchor_mask=model.get_anchor_mask()[i],
                class_num=model.get_class_num(),
                ignore_thresh=train_parameters['ignore_thresh'],
                use_label_smooth=True,  # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                downsample_ratio=downsample_ratio)
            
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        #optimizer, lr = optimizer_rms_setting()
        optimizer, lr = optimizer_custom_setting(learning_rate=0.0005)
        optimizer.minimize(loss)
        return [loss, lr]


def load_pretrained_params(exe, program):
    """
    load pretrained model
    :param exe:
    :param program:
    :return:
    """
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_model_dir']):
        logger.info('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_model_dir'],
                                   main_program=program)

    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_model_dir']):
        logger.info('load param from pretrained model')

        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=program,
                           predicate=if_exist)


def train():
    """
    train
    :return:
    """
    logger.info("start train YOLOv3, train params:%s", str(train_parameters))

    logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))
    # place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    logger.info("build network and program")
    train_program = fluid.Program()
    start_program = fluid.Program()
    eval_program = fluid.Program()
    test_program = fluid.Program()

    feeder, reader, loss, outputs = build_program_with_feeder(train_program, start_program, place)

    pred = build_program_with_feeder(test_program, start_program, istrain=False)
    
    cur_map, accum_map, eval_feeder = build_eval_program_with_feeder(eval_program, start_program)
    
    test_program = test_program.clone(for_test=True)
    eval_program = eval_program.clone(for_test=True)
    logger.info("build executor and init params")
    # exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss[0].name]
    load_pretrained_params(exe, train_program)
    fluid.contrib.model_stat.summary(train_program) # Print out the model input / output size, total params, and FLOPS

    stop_strategy = train_parameters['early_stop']
    rise_limit = stop_strategy['rise_limit']
    # sample_freq = stop_strategy['sample_frequency']
    # min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    # stop_train = False
    rise_count = 0
    total_batch_count = 0
    train_temp_loss = 0
    current_best_pass_ = 0
    current_best_map = 0

    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: {}, start read image".format(pass_id))
        batch_id = 0
        total_loss = 0.0
        for batch_id, data in enumerate(reader()):
            
            t1 = time.time()
            loss = exe.run(train_program, feed=feeder.feed(data), fetch_list=train_fetch_list)
            period = time.time() - t1
            loss = np.mean(np.array(loss))
            total_loss += loss
            batch_id += 1
            total_batch_count += 1

            if batch_id % 200 == 0:
                logger.info("pass {}, trainbatch {}, loss {}, time {}".format(pass_id, batch_id, loss,
                                                                              "%2.2f sec" % period))
        pass_mean_loss = total_loss / batch_id
        logger.info("pass {0} train result, current pass mean loss: {1}".format(pass_id, pass_mean_loss))

        if pass_id >= 90 or pass_id % 2 == 0:
            
            cur_map_, accum_map_ = eval(test_program, [pred.name], eval_program,
                                        [cur_map.name, accum_map.name], eval_feeder)
            logger.info("{} epoch current pass map is {}, accum_map is {}".format(pass_id, cur_map_, accum_map_))

            if cur_map_ > current_best_map:
                current_best_map = cur_map_
                current_best_pass_ = pass_id
                logger.info("model save {} epcho train result, current best pass MAP {}".format(pass_id,
                                                                                                current_best_map))
                fluid.io.save_persistables(dirname=train_parameters['save_model_dir'],
                                           main_program=train_program, executor=exe)
                fluid.io.save_inference_model(dirname=train_parameters['inference_model_dir'],
                                              feeded_var_names=['img'], target_vars=outputs[0], executor=exe, main_program=train_program)
                                          
            logger.info("best pass {} current best pass MAP is {}".format(current_best_pass_, current_best_map))
            
            if pass_mean_loss < min_loss:
                logger.info("Has reached the set optimum value, the training is over")
                break
    
            if rise_count > rise_limit:
                logger.info("rise_count > rise_limit, so early stop")
                break
            else:
                if pass_mean_loss > train_temp_loss:
                    rise_count += 1
                    train_temp_loss = pass_mean_loss
                else:
                    rise_count = 0
                    train_temp_loss = pass_mean_loss

    logger.info("end training")


if __name__ == '__main__':
    train()
