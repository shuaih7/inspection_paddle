import paddle.fluid as fluid
import numpy as np


def loss(logit, label, num_classes):
    label_nignore = fluid.layers.less_than(
        label.astype('float32'),
        fluid.layers.assign(np.array([num_classes], 'float32')),
        force_cpu=False).astype('float32')
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    logit = fluid.layers.softmax(logit, use_cudnn=False)
    loss = fluid.layers.cross_entropy(logit, label, ignore_index=8)
    label_nignore.stop_gradient = True
    label.stop_gradient = True
    return loss, label_nignore


def optimizer_momentum_setting(*args, **kwargs):
    learning_rate = fluid.layers.polynomial_decay(kwargs["base_lr"], kwargs["total_step"],
                                                  end_learning_rate=0, power=0.9)
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1,
                                                  regularization=fluid.regularizer.L2DecayRegularizer(
                                                      regularization_coeff=kwargs["weight_decay"]
                                                  ))
    return optimizer
