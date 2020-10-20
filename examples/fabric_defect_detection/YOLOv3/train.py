import os, time
import numpy as np
import paddle.fluid as fluid
from params import train_parameters
from utils import get_logger, load_pretrained_params
from config import build_program_with_feeder


logger = get_logger()


def train():
    print("start train YOLOv3 ...")
    
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
    
    train_program = fluid.Program()
    start_program = fluid.Program()
    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)
    
    exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss.name]
    load_pretrained_params(exe, train_program)
    
    # Early stop settings
    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    
    stop_train = False
    successive_count = 0
    total_batch_count = 0
    valid_thresh = train_parameters['valid_thresh']
    nms_thresh = train_parameters['nms_thresh']
    current_best_loss = 10000000000.0
    
    for pass_id in range(train_parameters["num_epochs"]):
        print("current pass: {}, start read image".format(pass_id))
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

            if batch_id % 10 == 0:
                print("pass {}, trainbatch {}, loss {} time {}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
                
        pass_mean_loss = total_loss / batch_id
        print("pass {0} train result, current pass mean loss: {1}".format(pass_id, pass_mean_loss))
        # 采用每训练完一轮停止办法，可以调整为更精细的保存策略
        if pass_mean_loss < current_best_loss:
            print("temp save {} epcho train result, current best pass loss {}".format(pass_id, pass_mean_loss))
            fluid.io.save_persistables(dirname=train_parameters['save_model_dir'], main_program=train_program, executor=exe)
            current_best_loss = pass_mean_loss

    print("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_model_dir'], main_program=train_program, executor=exe)


if __name__ == '__main__':
    train()
