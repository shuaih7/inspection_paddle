import os, paddle
import numpy as np
from paddle import fluid
from visualdl import LogWriter
from model import ResNet
from data import train_generator, valid_generator

#input层

image = fluid.layers.data(name='pixel', shape=[1,224,224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

Model = ResNet(layers=50)

predict = Model.net(input=image, class_dim=2)

# Loss Function
cost = fluid.layers.softmax_with_cross_entropy(logits=predict, label=label, soft_label=False)
avg_cost = fluid.layers.mean(cost)
batch_acc = fluid.layers.accuracy(input=fluid.layers.softmax(predict), label=label)

# Optimizer
opt = fluid.optimizer.AdamOptimizer(learning_rate=2e-4)
opt.minimize(avg_cost)

# # CPU Configurations
# place = fluid.CPUPlace()
# exe = fluid.Executor(place)

# GPU Configurations
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

# Initialize the program
test_program=fluid.default_main_program().clone(for_test=True)
exe.run(fluid.default_startup_program())

# Load the pretrained weights
pre_path = None #r"E:\Projects\Engine_Inspection\VGG16CAM\ResNet50\variable"
try: 
    fluid.io.load_params(executor=exe,dirname=pre_path)
    print("Successfully loaded the pretrained weights.")
except: print("Warning: The pretrained weights have not been loaded.")

# Load the dataset
batch_size   = 32
train_reader = paddle.batch(train_generator, batch_size=batch_size)
test_reader  = paddle.batch(valid_generator, batch_size=batch_size)
feeder       = fluid.DataFeeder(place=place, feed_list=[image,label])

# Create the logger
log_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\log"
log_writer = LogWriter(log_path, sync_cycle=5)
# 获取训练全部参数
params_name = fluid.default_startup_program().global_block().all_parameters()[0].name

"""
# This way will trigger the AttributeError: 'LogWriter' object has no attribute 'mode'
with log_writer.mode('train') as writer:
    train_cost_writer = writer.scalar('cost')
    train_acc_writer = writer.scalar('accuracy')
    histogram = writer.histogram('histogram', num_buckets=50)

with log_writer.mode('test') as writer:
    test_cost_writer = writer.scalar('cost')
    test_acc_writer = writer.scalar('accuracy')
"""

### References for future use: https://blog.csdn.net/qq_41427568/article/details/87735085
### Loading parameters: fluid.io.load_params(executor=exe,dirname=save_path) or fluid.io.load_persistables()
### Saving parameters:  fluid.io.save_params()
### Saving model for inference: fluid.io.save_inference_model(save_path,feeded_var_names=[image.name],target_vars=[model],executor=exe)
### For LogWriter information please refer to: https://blog.csdn.net/hua111hua/article/details/89422661

# Training
epoch_num = 100
report_freq = 1
base_acc = 0.0
model_save_path = r"E:\Projects\Fabric_Defect_Detection\model_proto\saved_model"
var_save_path   = r"E:\Projects\Fabric_Defect_Detection\model_proto\saved_var"

train_step, test_step = 0, 0
for epoch_id in range(epoch_num):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc, params = exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,batch_acc,params_name])
        
        # Write the training data into the LogWriter
        log_writer.add_scalar("train_cost", train_cost[0], train_step)
        log_writer.add_scalar("train_acc", train_acc[0], train_step)
        log_writer.add_histogram("histogram", params.flatten(), train_step, buckets=50)
        train_step += 1
        
        # Printing the result every 100 batch
        if batch_id % report_freq == 0:
            print("Pass:%d,Batch:%d,Cost:%0.5f,Accuracy:%0.5f"%(epoch_id,batch_id,train_cost[0],train_acc[0]))
            
    # Testing
    test_accs=[]
    test_costs=[]
    for batch_id,data in enumerate(test_reader()):
        test_cost,test_acc=exe.run(program=test_program,feed=feeder.feed(data),fetch_list=[avg_cost,batch_acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
        
        # Write the training data into the LogWriter
        log_writer.add_scalar("test_cost", test_cost[0], test_step)
        log_writer.add_scalar("test_acc", test_acc[0], test_step)
        test_step += 1
        
    # Get the avergaed valiue
    test_cost=(sum(test_costs)/len(test_costs))
    test_acc=(sum(test_accs)/len(test_accs))
    print('Test:%d,Cost:%0.5f,Accuracy:%0.5f'%(epoch_id,test_cost,test_acc))
    print()
    
    # Saving the inference model
    if test_acc > base_acc: 
        fluid.io.save(fluid.default_main_program(), os.path.join(model_save_path, str(epoch_id)))
        fluid.io.save_params(executor=exe,dirname=var_save_path)
        base_acc = test_acc # Update the baseline accuracy
        print("Model and variables are saved.")
    
