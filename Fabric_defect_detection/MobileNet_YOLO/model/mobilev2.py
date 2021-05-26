# -*- coding: UTF-8 -*-
"""
主干网络基于MobileNetV2
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class MobileNetV2YOLOv3(object):
    """
    定义模型
    """
    def __init__(self, class_num, anchors, anchor_mask, scale=1.5):
        """
        模型参数初始化
        :param class_num:
        :param anchors:
        :param anchor_mask:
        :param scale:
        """
        self.outputs = []
        self.downsample_ratio = 16
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num
        self.scale = scale
        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def inverted_residual_unit(self,
                               input,
                               num_in_filter,
                               num_filters,
                               ifshortcut,
                               stride,
                               filter_size,
                               padding,
                               expansion_factor,
                               name=None):
        num_expfilter = int(round(num_in_filter * expansion_factor))

        channel_expand = self.conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name=name + '_expand')

        bottleneck_conv = self.conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = self.conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')
        if ifshortcut:
            out = self.shortcut(input=input, data_residual=linear_out)
            return out
        else:
            return linear_out

    def invresi_blocks(self, input, in_c, t, c, n, s, name=None):
        # 定义残差块
        first_block = self.inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block = self.inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=3,
                padding=1,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block

    def yolo_detection_block(self, input, num_filters, k=2,name=None):
        """
        yolo_detection_block
        :param input:
        :param num_filters:
        :param k:
        :return:
        """
        yolo_name = name
        assert num_filters % 2 == 0, "num_filters {} cannot be divided by 2".format(num_filters)
        conv = input
        for j in range(k):
            c = self.conv_bn_layer(conv, num_filters, filter_size=1, stride=1, padding=0,name = yolo_name +str(j)+ 'c')
            conv = self.conv_bn_layer(conv, num_filters * 2, filter_size=3, stride=1, padding=1,name = yolo_name+str(j) + 'conv')
        route = self.conv_bn_layer(conv, num_filters, filter_size=1, stride=1, padding=0,name = yolo_name +str(j)+ 'route')
        tip = self.conv_bn_layer(route, num_filters * 2, filter_size=3, stride=1, padding=1,name = yolo_name +str(j)+ 'tip')
        return route, tip

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      if_act=True,
                      name=None,
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            return fluid.layers.relu(bn)
        else:
            return bn

    def shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)

    def net(self, input):
        """
        构造模型
        :param input:
        :return:
        """
        blocks = []
        scale = 1.0
        class_num = 20
        """
        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        """
        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 1),
        ]
        
        # MobileNet的第一个卷积层 
        input = self.conv_bn_layer(
            input,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            name='conv1_1')
        # bottleneck sequences
        # MobileNet 一共有7个bottleneck(即残差块)
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            input = self.invresi_blocks(
                input=input,
                in_c=in_c,
                t=t,
                c=int(c * scale),
                n=n,
                s=s,
                name='conv' + str(i))
            in_c = int(c * scale)
            blocks.append(input)

        # Attach the yolo head
        block = blocks[-1] # Get the last output as the Fast-YOLO input
        #route, tip = self.yolo_detection_block(block, num_filters=256, k=2,name='dect_'+str(i))
        route, tip = self.yolo_detection_block(block, num_filters=256, k=1,name='dect_'+str(i))
        block_out = fluid.layers.conv2d(
            input=tip,
            num_filters=len(self.anchor_mask[0]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name="block-out-" + str(i),
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)
            #bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
        self.outputs.append(block_out)
        
        return self.outputs


def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    return MobileNetV2YOLOv3(class_num, anchors, anchor_mask)
