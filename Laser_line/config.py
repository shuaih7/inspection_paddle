# 基础参数, 还没有分类整理好
agrs = {
    "data_dir": 'data/iccv09Data/',       # 数据所在文件夹
    "batch_size": 10,                     # 设置训练时的batch_size
    "use_gpu": True,                      # 是否使用gpu进行训练
    "train_model_dir": "train model",     # 训练阶段暂时保存模型的路径，可重新加载再训练
    "infer_model_dir": "infer model",     # 最终模型保存的路径
    "pretrained_model_dir": "pretrained model/deeplabv3plus_gn",  # 预训练模型存在的地方
    "eval_file_path": 'data/eval_list.txt',  # 验证集的路径
    "continue_train": False,               # 是否接着上次的继续训练
    "paddle_flag": False,                  # 是否使用paddle上预训练好的模型进行微调，如果要接着上次断点训练，这里需要改为False
    "num_classes": 8,                     # 标签的类数
    "weight_decay": 0.00004,
    "base_lr": 0.001,                     # 初始的学习率
    "num_epochs": 500,                    # 总的epochs数
    "total_step": 20000,                  # 总的步数，计算方式：num_epochs * (样本总数 / batch_size)
    "image_shape": [240, 320],             # 图像的大小
    "enable_ce": False,
    "bn_momentum": 0.9997,
    "dropout_keep_prop": 0.9,
    "default_norm_type": 'gn',             # 默认的归一化方式
    "decode_channel": 48,
    "encode_channel": 256,
    "default_epsilon": 1e-3,
    "default_group_number": 32,
    "depthwise_use_cudnn": False,
    "is_train": True,
    "data_augmentation_config": {
        "use_augmentation": False,
        "min_resize": 0.5,
        "max_resize": 4,
        "crop_size": 240
    }
}
# 是否数据增强
