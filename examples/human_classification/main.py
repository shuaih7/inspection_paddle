import os, paddle
import numpy as np
from paddle import fluid
from model import ResNet
from data import train_generator, valid_generator

#input层

image = fluid.layers.data(name='pixel', shape=[3,224,224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

Model = ResNet(layers=50)

predict = Model.net(input=image, class_dim=2)

# Loss Function
cost = fluid.layers.softmax_with_cross_entropy(logits=predict, label=label, soft_label=False)
avg_cost = fluid.layers.mean(cost)
batch_acc = fluid.layers.accuracy(input=fluid.layers.softmax(predict), label=label)

# Optimizer
opt = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
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
pre_path = r"E:\Projects\Engine_Inspection\VGG16CAM\ResNet50\variable"
try: 
    fluid.io.load_params(executor=exe,dirname=pre_path)
    print("Successfully loaded the pretrained weights.")
except: print("Warning: The pretrained weights have not been loaded.")

# Load the dataset
batch_size = 32
train_reader = paddle.batch(train_generator, batch_size=batch_size)
test_reader  = paddle.batch(valid_generator, batch_size=batch_size)
feeder=fluid.DataFeeder(place=place,feed_list=[image,label])

### References for future use: https://blog.csdn.net/qq_41427568/article/details/87735085
### Loading parameters: fluid.io.load_params(executor=exe,dirname=save_path) or fluid.io.load_persistables()
### Saving parameters:  fluid.io.save_params()
### Saving model for inference: fluid.io.save_inference_model(save_path,feeded_var_names=[image.name],target_vars=[model],executor=exe)

# Training
epoch_num = 3
report_freq = 10
base_acc = 0.0
model_save_path = r"E:\Projects\Engine_Inspection\VGG16CAM\ResNet50\model"
var_save_path   = r"E:\Projects\Engine_Inspection\VGG16CAM\ResNet50\variable"

for epoch_id in range(epoch_num):
    for batch_id, data in enumerate(train_reader()):
        train_cost,train_acc=exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,batch_acc])
        
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
    # Get the avergaed valiue
    test_cost=(sum(test_costs)/len(test_costs))
    test_acc=(sum(test_accs)/len(test_accs))
    print('Test:%d,Cost:%0.5f,Accuracy:%0.5f'%(epoch_id,test_cost,test_acc))
    print()
    
    # Saving the inference model
    if test_acc > base_acc: 
        fluid.io.save(fluid.default_main_program(), model_save_path)
        #fluid.io.save_inference_model(model_save_path,feeded_var_names=[image.name],target_vars=[predict],executor=exe)
        fluid.io.save_params(executor=exe,dirname=os.path.join(var_save_path, str(epoch_id)))
        base_acc = test_acc # Update the baseline accuracy
        print("Model and variables are saved.")
    
