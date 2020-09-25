
import os,sys, cv2
import numpy as np
import paddle
from paddle import fluid


# region parameters
# region paths
Data_path = "./data/"
TestData_path = Data_path + 'pic/'
Model_path = 'model/'
Model_file_tf = "model/lenet5_tf.ckpt"
Model_file_keras = "model/lenet5_keras.h5"
Model_file_torch = "model/lenet5_torch.pth"
Model_file_paddle = "model/lenet5_paddle.model"
# endregion
 
# region image parameter
Img_size = 28
Img_chs = 1
Label_size = 1
Labels_classes = 10
# endregion
 
# region net parameter
Conv1_kernel_size = 5
Conv1_chs = 6
Conv2_kernel_size = 5
Conv2_chs = 16
Conv3_kernel_size = 5
Conv3_chs = 120
Flatten_size = 120
Fc1_size = 84
Fc2_size = Labels_classes
# endregion
 
# region hpyerparameter
Learning_rate = 1e-3
Batch_size = 64
Buffer_size = 256
Infer_size = 1
Epochs = 6
Train_num = 60000
Train_batch_num = Train_num // Batch_size
Val_num = 10000
Val_batch_num = Val_num // Batch_size
# endregion
place = fluid.CUDAPlace(0) if fluid.cuda_places() else fluid.CPUPlace()
# endregion


class Lenet5:
    def __init__(self,structShow=False):
        self.structShow = structShow
        self.image = fluid.layers.data(shape=[Img_chs, Img_size, Img_size], dtype='float32', name='image')
        self.label = fluid.layers.data(shape=[Label_size], dtype='int64', name='label')
        self.predict = self.net_lenet5()
 
    def net_lenet5(self):
        conv1 = fluid.layers.conv2d(self.image,6,filter_size=5,stride=1,padding=2,act='relu')
        pool1 = fluid.layers.pool2d(conv1,2,pool_stride=2,pool_type='max')
 
        conv2 = fluid.layers.conv2d(pool1,16,filter_size=5,stride=1,padding=0,act='relu')
        pool2 = fluid.layers.pool2d(conv2,2,pool_stride=2,pool_type='max')
 
        conv3 = fluid.layers.conv2d(pool2,120,filter_size=5,stride=1,padding=0,act='relu')
 
        flatten = fluid.layers.flatten(conv3,axis=1)
        fc1 = fluid.layers.fc(flatten,84,act='relu')
        fc2 = fluid.layers.fc(fc1,10,act='softmax')
 
        if self.structShow:
            print(conv1.name,conv1.shape)
            print(pool1.name,pool1.shape)
            print(conv2.name,conv2.shape)
            print(pool2.name,pool2.shape)
            print(conv3.name,conv3.shape)
            print(flatten.name,flatten.shape)
            print(fc1.name,fc1.shape)
            print(fc2.name,fc2.shape)
        return fc2
        
        
def train():
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=Buffer_size),
        batch_size=Batch_size)
    val_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.test(),buf_size=Buffer_size),
        batch_size=Batch_size)
 
    net = Lenet5(structShow=True)
    image,label,predict = net.image,net.label,net.predict
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
 
    loss = fluid.layers.cross_entropy(input=predict, label=label)
    loss_mean = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=predict, label=label)
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=Learning_rate)
    optimizer.minimize(loss_mean)
 
    val_program = fluid.default_main_program().clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
 
    best_loss = float("inf")
    best_loss_epoch = 0
    for epoch in range(Epochs):
        print('Epoch %d/%d:' % (epoch + 1, Epochs))
        train_sum_loss = 0
        train_sum_acc = 0
        val_sum_loss = 0
        val_sum_acc = 0
        for batch_num, data in enumerate(train_reader()):
            train_loss, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                            feed=feeder.feed(data),  # 给模型喂入数据
                                            fetch_list=[loss_mean, acc])  # fetch 误差、准确率
            train_sum_loss += train_loss[0]
            train_sum_acc += train_acc[0]
            process_show(batch_num + 1, Train_num/Batch_size, train_acc, train_loss, prefix='train:')
 
        for batch_num, data in enumerate(val_reader()):
            val_loss, val_acc = exe.run(program=val_program,  # 执行训练程序
                                        feed=feeder.feed(data),  # 喂入数据
                                        fetch_list=[loss_mean, acc])  # fetch 误差、准确率
            val_sum_loss += val_loss[0]
            val_sum_acc += val_acc[0]
            process_show(batch_num + 1, Val_num / Batch_size, val_acc, val_loss, prefix='train:')
 
        train_sum_loss /= (Train_num//Batch_size)
        train_sum_acc /= (Train_num//Batch_size)
        val_sum_loss /= (Val_num // Batch_size)
        val_sum_acc /= (Val_num // Batch_size)
        print('average summary:\ntrain acc %.4f, loss %.4f ; val acc %.4f, loss %.4f'
              % (train_sum_acc, train_sum_loss, val_sum_acc, val_sum_loss))
 
        if val_sum_loss < best_loss:
            print('val_loss improve from %.4f to %.4f, model save to %s ! \n' % (best_loss, val_sum_loss,Model_file_paddle))
            best_loss = val_sum_loss
            best_loss_epoch = epoch+1
            fluid.io.save_inference_model(Model_file_paddle,  # 保存推理model的路径
                                          ['image'],  # 推理（inference）需要 feed 的数据
                                          [predict],  # 保存推理（inference）结果的 Variables
                                          exe)  # executor 保存 inference model
        else:
            print('val_loss do not improve from %.4f \n' % (best_loss))
    print('best loss %.4f at epoch %d \n'%(best_loss,best_loss_epoch))
 
 
def load_image(file):
    img = Image.open(file).convert('L')                        #将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    img = img.resize((Img_size, Img_size), Image.ANTIALIAS)                 #resize image with high-quality 图像大小为28*28
    img = np.array(img).reshape(Infer_size, Img_chs, Img_size, Img_size).astype(np.float32)#返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    # print(im)
    img = img / 255.0 * 2.0 - 1.0                               #归一化到【-1~1】之间
    return img
 
def inference(infer_path=TestData_path,model_path = Model_file_paddle):
    '''
    推理代码
    :param infer_path: 推理数据
    :param model_path: 模型
    :return: 
    '''
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program,  # 推理Program
         feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
         fetch_targets] = fluid.io.load_inference_model(model_path,infer_exe)
        print('get model from',model_path)
        for image_name in os.listdir(infer_path):
            img = load_image(infer_path+image_name)
            results = infer_exe.run(program=inference_program,  # 运行推测程序
                                    feed={feed_target_names[0]: img},  # 喂入要预测的img
                                    fetch_list=fetch_targets)  # 得到推测结果,
            pre = np.argsort(results)  # argsort函数返回的是result数组值从小到大的索引值
            print("{} predict result {}" .format(image_name,pre[0][0][-1]))
 
 
def process_show(num, nums, train_acc, train_loss, prefix='', suffix=''):
    rate = num / nums
    ratenum = int(round(rate, 2) * 100)
    bar = '\r%s batch %3d/%d:train accuracy %.4f, train loss %00.4f [%s%s]%.1f%% %s; ' % (
        prefix, num, nums, train_acc, train_loss, '#' * (ratenum//2), '_' * (50 - ratenum//2), ratenum, suffix)
    sys.stdout.write(bar)
    sys.stdout.flush()
    if num >= nums:
        print()
        
if __name__ == '__main__':
    pass
    train()
    inference()
