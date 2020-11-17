## Searching for Jetson-TX2 Deployment

### 1. Compile PaddlePaddle with TensorRT together on the terminal

- Failed because this takes too long and the "sudo make install" command will cause the memory overflow.

### 2. Transfer the Paddle model into onnx and load with TensorRT

- Command line: 
  1). cd to the folder where you want to save the onnx model
  2). paddle2onnx --model_dir "E:\Projects\Fabric_Defect_Detection\model_proto\ResNet18\saved_model" --save_file onnx_file_name
