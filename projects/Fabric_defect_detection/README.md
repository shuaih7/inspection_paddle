<h1 align="center">
  Fabric Defect Detection Algorithm
</h1>

<h4 align="center">
  AI Object Detection Development for Skipping Stitch and Striation
</h4>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue"></a>
  <a href="https://github.com/shuaih7/inspection_paddle/tree/master/Fabric_defect_detection"><img src="https://img.shields.io/badge/version-0.2.0-brightgreen"></a>
</div>

## Description
In the textile industry, long defects like the skipping stitch severely postpone the manufactory pipeline and cost lots of cloth waste. This project is aimed to build up an object detection algorithm which could perform a real-time fabric defect detection on a portable device (for example, the Jetson Nano). The algorithm would deliver a >99% recall rate with the false alarm rate less than 1%. Make sure to optimize the algorithm to be portable and compatible enough to fit the terminal's computing power.

## Visuals
Below are the sample skipping stitch detection results for white and gray linen:
<div align=center><img src="https://github.com/shuaih7/inspection_paddle/blob/master/projects/Fabric_defect_detection/src/results.jpg"></div>

## Requirements
- Ubuntu / macOS / Windows
- Python3.5 / Python3.6 / Python3.7

## Installation
Install paddlepaddle-gpu v1.8.4 using pip:

    pip install paddlepaddle-gpu==1.8.4
    
If the GPU is not available or not supported by paddle, then install the CPU version instand:

    pip install paddlepaddle==1.8.4

Install other dependencies before running the algorithm development tools. To install dependencies, run the following command in the virtual environment:

    pip install -r Requirements.txt
    
If encountered the url error multiple times, try using the Tsinghua Pypi Source:

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some_pkgs
    
## Usage

    data:  scripts for the dataset generation, pre and post-processing, and data augmentation modules
    log:  folder containing the log file
    config.py:  The model configuration matrix
    Fast_YOLO.py:  YOLO model structure
    reader.py: Create the data input pipeline
    infer.py:  Do the inference, where the model file should be freezed
    freeze.py:  Freeze the trained paddle model
    utils.py:  Utility functions
    train.py: Script to start training

## Development
* Version 0.2.0 is developed based on YOLOv3 with the MobileNetV2 backbone
* Model receives a 95.42% mAP on the dataset obtained on 10.13.2020

## Jetson TX2 / Nano Deployment
- Download Jetpack v4.2.0 and upgrade the Jetson device
- Transfer the backbone and YOLO head into onnx file

    paddle2onnx --model_dir "model_save_dir" --save_file onnx_file_name.onnx
    
 - Transfer the onnx model into tensorRT model and add the pre and post-processing methods, referring to this [repository](https://github.com/xuwanqi/yolov3-tensorrt)
 - The computing speed on both TX2 and Nano are around 20fps
 - Alternatively, you can compiling install paddlepaddle on Jetson, referring to this [blog](https://blog.csdn.net/weixin_45449540/article/details/107704028)

## Acknowledgement
This project is supported by Shanghai Three Gun Group Co., Ltd. All of the image dataset are captured from its factory in Dafeng, Jiangsu Province. Persons contributing to this project are:

    Jingping Zhao: zhaojingping@handaotech.com
    Jay Han:       hanjie@handaotech.com
    Shuai Hao:     haoshuai@handaotech.com

## License

    Copyright (c) 2020 Fabric_defect_detection Authors. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.



