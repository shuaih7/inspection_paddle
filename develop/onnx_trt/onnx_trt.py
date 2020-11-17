#--*-- coding:utf-8 --*--
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import time 
from PIL import Image
import cv2, os, sys
import numpy as np

filename = '/home/zwzhou/Code/img.png'
max_batch_size = 1
#onnx_model_path = "./lenet_opt10.onnx"
onnx_model_path = "./mnist_model_pytorch.onnx"
#onnx_model_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/ResNet18/resnet18_opt10.onnx"

TRT_LOGGER = trt.Logger()
    
def get_img_np_nchw(filename): # -> read grayscale image
    #image_cv = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image_cv = np.zeros((28,28), dtype=np.float32)
    img_np = np.array(image_cv, dtype=np.float)/255.
    img_np_nchw = np.expand_dims(np.expand_dims(np.squeeze(img_np), 0), 0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        # print(binding) # 绑定的输入输出
        # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
        size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
        # volume 计算可迭代变量的空间，指元素个数
        # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # get_binding_dtype  获得binding的数据类型
        # nptype等价于numpy中的dtype，即数据类型
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)    # cuda分配空间
        # print(int(device_mem)) # binding在计算图中的缓冲地址
        bindings.append(int(device_mem))
        #append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
    
    
def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            #last_layer = network.get_layer(network.num_layers-1)
            #network.mark_output(last_layer.get_output(0))

            ### This part is for the network checking 
            inp = network.get_input(0)
            print(inp.shape)
            #sys.exit()

            for i in range(network.num_layers):
                cur_layer = network.get_layer(i)
                output = cur_layer.get_output(0)
                print(output.shape)
            #output = last_layer.get_output(0)
            #print(output.shape)
            #sys.exit()

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    """
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
    """
    return build_engine()
    

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # htod： host to device 将数据由cpu复制到gpu device
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # 当创建network时显式指定了batchsize， 则使用execute_async_v2, 否则使用execute_async
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

img_np_nchw = get_img_np_nchw(filename).astype(np.float32)
#These two modes are depend on hardwares
trt_engine_path = "./model_fp16.trt"
# Build an cudaEngine
engine = get_engine(onnx_model_path, trt_engine_path)
# 创建CudaEngine之后,需要将该引擎应用到不同的卡上配置执行环境
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# Do inference
shape_of_output = (max_batch_size, 1000)
# Load data to the buffer
inputs[0].host = img_np_nchw.reshape(-1)

# inputs[1].host = ... for multiple input
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

print('TensorRT ok')

