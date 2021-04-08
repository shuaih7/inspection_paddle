import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class YOLOHead(object):
    def __init__(self):
        pass
        
    def __call__(self, blocks, kind='full'):
        if kind == 'single':
            return self.get_single_outputs(blocks)
        else: 
            return self.get_outputs(blocks)
            
    def get_single_outputs(self, blocks):
        outputs = list()
        # Attach the yolo head
        block = blocks[0] # Get the last output as the Fast-YOLO input
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
        outputs.append(block_out)
        
        return outputs
        
    def get_outputs(self, blocks):
        outputs = list()
        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
                # print(block.shape)
            
            route, tip = self.yolo_detection_block(block, num_filters=256 // (2 ** i), k=2)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return outputs
        
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