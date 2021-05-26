from __future__ import division  # 引入精确除法，不确定这里要不要
from __future__ import absolute_import  # 引入绝对引用
from __future__ import print_function  # 输出得带括号
# 上面的这部分大概是保证python3写出来的代码在python2.7上也可以运行

import paddle
from PIL import Image
import cv2
import paddle.fluid as fluid
import numpy as np
import contextlib  # 上下文管理的库
import os


name_scope = ""  # 网络中各层的命名空间

# 递归的方式对网络的每一层进行命名
@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '/'
    yield  # 这里并不是为了把这个函数变成生成器，而是为了在这里产生中断，不让后面那个命令执行，直到最后with的生存周期结束，执行最后那句
    name_scope = bk


# 大概是对输入数据的一个审核，保证数据是一个列表而不是一个单纯的数字，如果是数字，就把他复制number遍然后形成一个list
def check(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


class DeepLabV3p(object):
    def __init__(self, args):
        self.decode_channel = args['decode_channel']
        self.encode_channel = args['encode_channel']
        self.label_number = args['num_classes']
        self.bn_momentum = args['bn_momentum']
        self.dropout_keep_prop = args['dropout_keep_prop']
        self.is_train = args['is_train']
        self.default_epsilon = args['default_epsilon']
        self.default_norm_type = args['default_norm_type']
        self.default_group_number = args['default_group_number']
        self.depthwise_use_cudnn = args['depthwise_use_cudnn']

        self.bn_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)
        self.depthwise_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)

        self.op_results = {}

    # 由于不太了解op_results，所以这个函数目前用途不明
    def append_op_result(self, result, name):
        op_index = len(self.op_results)
        name = name_scope + name + str(op_index)
        self.op_results[name] = result
        return result

    # 卷积操作
    def conv(self, *args, **kargs):
        # 这里大概是先根据不同层设置一下参数初始化时候的标准差？？
        if "xception" in name_scope:
            init_std = 0.09
        elif "logit" in name_scope:
            init_std = 0.01
        elif name_scope.endswith('depthwise/'):
            init_std = 0.33
        else:
            init_std = 0.06

        # 根据是不是深度可分离卷积来决定是否要使用正则化
        if name_scope.endswith('depthwise/'):
            regularizer = self.depthwise_regularizer
        else:
            regularizer = None

        # 根据本层的名字来对参数的属性进行设置，包括名字、正则化、初始化方式这三个
        kargs['param_attr'] = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=regularizer,
            initializer=fluid.initializer.TruncatedNormal(
            loc=0.0, scale=init_std))

        # 判断是否需要加偏置
        if 'bias_attr' in kargs and kargs['bias_attr']:
            kargs['bias_attr'] = fluid.ParamAttr(
                name=name_scope + 'biases',
                regularizer=regularizer,
                initializer=fluid.initializer.ConstantInitializer(value=0.0))
        else:
            kargs['bias_attr'] = False
        kargs['name'] = name_scope + 'conv'
        return self.append_op_result(fluid.layers.conv2d(*args, **kargs), 'conv')

    # 一种船新的归一化方式，克服了BN对bitch_size的依赖
    def group_norm(self, input, G, eps=1e-5, param_attr=None, bias_attr=None):
        N, C, H, W = input.shape
        if C % G != 0:
            # print "group can not divide channle:", C, G
            for d in range(10):
                for t in [d, -d]:
                    if G + t <= 0: continue
                    if C % (G + t) == 0:
                        G = G + t
                        break
                if C % G == 0:
                    # print "use group size:", G
                    break
        assert C % G == 0
        x = fluid.layers.group_norm(
            input,
            groups=G,
            param_attr=param_attr,
            bias_attr=bias_attr,
            name=name_scope + 'group_norm')
        return x

    # 归一化
    def bn(self, *args, **kargs):
        # 在这里选择采用BN 还是 船新的GN归一化方式
        if self.default_norm_type == 'bn':
            with scope('BatchNorm'):
                return self.append_op_result(
                    fluid.layers.batch_norm(
                        *args,
                        epsilon=self.default_epsilon,
                        momentum=self.bn_momentum,
                        param_attr=fluid.ParamAttr(
                            name=name_scope + 'gamma', regularizer=self.bn_regularizer),
                        bias_attr=fluid.ParamAttr(
                            name=name_scope + 'beta', regularizer=self.bn_regularizer),
                        moving_mean_name=name_scope + 'moving_mean',
                        moving_variance_name=name_scope + 'moving_variance',
                        **kargs),
                    'bn')
        elif self.default_norm_type == 'gn':
            with scope('GroupNorm'):
                return self.append_op_result(
                    self.group_norm(
                            args[0],
                            self.default_group_number,
                            eps=self.default_epsilon,
                            param_attr=fluid.ParamAttr(
                                name=name_scope + 'gamma', regularizer=self.bn_regularizer),
                            bias_attr=fluid.ParamAttr(
                                name=name_scope + 'beta', regularizer=self.bn_regularizer)), 'gn')
        else:
            raise "Unsupport norm type:" + self.default_norm_type

    # 激活函数的添加
    def bn_relu(self, data):
        return self.append_op_result(fluid.layers.relu(self.bn(data)), 'relu')

    def relu(self, data):
        return self.append_op_result(
            fluid.layers.relu(
                data, name=name_scope + 'relu'), 'relu')

    # 空洞卷积,具体实现时被分成了可分离卷积，分诶depthwise和pointwise两个部分
    def seperate_conv(self, input, channel, stride, filter, dilation=1, act=None):
        with scope('depthwise'):
            input = self.conv(
                input,
                input.shape[1],
                filter,
                stride,
                groups=input.shape[1],
                padding=(filter // 2) * dilation,
                dilation=dilation,
                use_cudnn=self.depthwise_use_cudnn)
            input = self.bn(input)
            if act:
                input = act(input)
        with scope('pointwise'):
            input = self.conv(input, channel, 1, 1, groups=1, padding=0)
            input = self.bn(input)
            if act:
                input = act(input)
        return input

    # xception块
    def xception_block(self, input,
                       channels,
                       strides=1,
                       filters=3,
                       dilation=1,
                       skip_conv=True,
                       has_skip=True,
                       activation_fn_in_separable_conv=False):
        # 重复次数？？,用来控制这个xception_block做几次空洞卷积
        repeat_number = 3
        channels = check(channels, repeat_number)
        filters = check(filters, repeat_number)
        strides = check(strides, repeat_number)
        data = input
        results = []
        for i in range(repeat_number):
            with scope('separable_conv' + str(i + 1)):
                if not activation_fn_in_separable_conv:
                    data = self.relu(data)
                    data = self.seperate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation)
                else:
                    data = self.seperate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation,
                        act=self.relu)
                results.append(data)
        # 这里判断是否有远跳链接
        if not has_skip:
            return self.append_op_result(data, 'xception_block'), results
        if skip_conv:
            with scope('shortcut'):
                skip = self.bn(
                    self.conv(
                        input, channels[-1], 1, strides[-1], groups=1, padding=0))
        else:
            skip = input
        return self.append_op_result(data + skip, 'xception_block'), results

    def entry_flow(self, data):
        with scope("entry_flow"):
            with scope("conv1"):
                data = self.conv(data, 32, 3, stride=2, padding=1)
                data = self.bn_relu(data)
            with scope("conv2"):
                data = self.conv(data, 64, 3, stride=1, padding=1)
                data = self.bn_relu(data)
            with scope("block1"):
                data, _ = self.xception_block(data, 128, [1, 1, 2])
            with scope("block2"):
                data, results = self.xception_block(data, 256, [1, 1, 2])
            with scope("block3"):
                data, _ = self.xception_block(data, 728, [1, 1, 2])
            return data, results[1]

    def middle_flow(self, data):
        with scope("middle_flow"):
            for i in range(16):
                with scope("block" + str(i + 1)):
                    data, _ = self.xception_block(data, 728, [1, 1, 1], skip_conv=False)
        return data

    def exit_flow(self, data):
        with scope("exit_flow"):
            with scope('block1'):
                data, _ = self.xception_block(data, [728, 1024, 1024], [1, 1, 1])
            with scope('block2'):
                data, _ = self.xception_block(
                    data, [1536, 1536, 2048], [1, 1, 1],
                    dilation=2,
                    has_skip=False,
                    activation_fn_in_separable_conv=True)
            return data

    # dropout层
    def dropout(self, x, keep_rate):
        if self.is_train:
            return fluid.layers.dropout(x, 1 - keep_rate) / keep_rate
        else:
            return x

    # 编码器的实现
    def encoder(self, input):
        with scope('encoder'):
            channel = 256  # 目前没看出来这个channel是控制哪里的，为啥要在这里写成固定的常量,补充：好的知道这个变量是干嘛的了
            with scope("image_pool"):
                # 这里大概就是deeplab v3+ 所说的图像级的特征了，换句话说就是做了一个global pooling，然后过两个卷积，然后就差不多了。
                image_avg = fluid.layers.reduce_mean(input, [2, 3], keep_dim=True)  # 这个函数是用来求给定纬度平均值的
                                                                                    # 寻找图像的均值
                self.append_op_result(image_avg, 'reduce_mean')

                image_avg = self.bn_relu(
                    self.conv(
                        image_avg, channel, 1, 1, groups=1, padding=0  # 这个groups是干啥用呢
                    )
                )

                image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])
            # 上面这个image_pool部分是为了干啥呢？可能得看看论文
            with scope("aspp0"):
                aspp0 = self.bn_relu(self.conv(input, channel, 1, 1, groups=1, padding=0))
            with scope("aspp1"):
                aspp1 = self.seperate_conv(input, channel, 1, 3, dilation=6, act=self.relu)  # 很僵，这个代码filter_size 和 stride 的顺序和paddle的是反着的
            with scope("aspp2"):
                aspp2 = self.seperate_conv(input, channel, 1, 3, dilation=12, act=self.relu)
            with scope("aspp3"):
                aspp3 = self.seperate_conv(input, channel, 1, 3, dilation=18, act=self.relu)
            with scope("concat"):
                data = self.append_op_result(
                    fluid.layers.concat(
                        [image_avg, aspp0, aspp1, aspp2, aspp3], axis=1), 'concat'
                    )
                data = self.bn_relu(self.conv(data, channel, 1, 1, groups=1, padding=0))
                data = self.dropout(data, self.dropout_keep_prop)
            return data

    # 下来该解码器
    def decoder(self, encode_data, decode_shortcut):
        with scope('decoder'):
            with scope('concat'):
                decode_shortcut = self.bn_relu(
                    self.conv(
                        decode_shortcut, self.decode_channel, 1, 1, groups=1, padding=0
                    )
                )
                encode_data = fluid.layers.resize_bilinear(
                    encode_data, decode_shortcut.shape[2:]
                )
                encode_data = fluid.layers.concat(
                    [encode_data, decode_shortcut], axis=1)
                self.append_op_result(encode_data, 'concat')
            with scope("separable_conv1"):
                encode_data = self.seperate_conv(
                    encode_data, self.encode_channel, 1, 3, dilation=1, act=self.relu)
            with scope("separable_conv2"):
                encode_data = self.seperate_conv(
                    encode_data, self.encode_channel, 1, 3, dilation=1, act=self.relu)
            return encode_data

    def net(self, img):
        self.append_op_result(img, 'img')
        with scope("xception_65"):
            self.default_epsilon = 1e-3

            # Entry flow
            data, decode_shortcut = self.entry_flow(img)

            # Middle flow
            data = self.middle_flow(data)

            # Exit flow
            data = self.exit_flow(data)
        self.default_epsilon = 1e-5
        encode_data = self.encoder(data)
        encode_data = self.decoder(encode_data, decode_shortcut)
        with scope('logit'):
            logit = self.conv(
                encode_data, self.label_number, 1, stride=1, padding=0, bias_attr=True)
            logit = fluid.layers.resize_bilinear(logit, img.shape[2:])
        return logit
