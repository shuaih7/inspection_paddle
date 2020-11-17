#--*-- coding:utf-8 --*--
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
import cv2, os, sys, time
import numpy as np
from paddle import fluid
import config

filename = 'sample.png'
max_batch_size = 1
onnx_model_path = "./fast_yolo.onnx"
train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
TRT_LOGGER = trt.Logger()


def draw_bbox_image(img, boxes, scores, gt=False):
    '''
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    '''
    color = ['red', 'blue']
    if gt:
        c = color[1]
    else:
        c = color[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
        draw.text((xmin, ymin), str(score), (255, 255, 0))
    return img
    

def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img


def read_image(img):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = img
    img = resize_img(origin, yolo_config["input_size"])
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img

    
def get_img_np_nchw(filename): # -> read grayscale image
    image_cv = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32)
    img_np = np.array(image_cv-127.0, dtype=np.float)/255.
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
    
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
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
    trt_outputs = [out.host for out in outputs]
    
    # For Fast-YOLO, there is only one output in the trt_outputs list
    #output = np.array(trt_outputs[0], dtype=np.float32)
    boxes = []
    scores = []
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')
    downsample_ratio = 1
    # feeder = fluid.DataFeeder(feed_list=[img, image_shape, gt_label, gt_box, difficult], place=place, program=main_prog)
    # reader = create_eval_reader(train_parameters['eval_list'], train_parameters['data_dir'], yolo_config['input_size'], 'eval')
    for i, out in enumerate(trt_outputs):
        out = np.array(out, dtype=np.float32)
        
        box, score = fluid.layers.yolo_box(
            x=out,
            img_size=image_shape,
            anchors=model.get_yolo_anchors()[i],
            class_num=model.get_class_num(),
            conf_thresh=train_parameters['valid_thresh'],
            downsample_ratio=downsample_ratio,
            name="yolo_box_" + str(i))
        boxes.append(box)
        scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
        downsample_ratio //= 2

    pred = fluid.layers.multiclass_nms(
        bboxes=fluid.layers.concat(boxes, axis=1),
        scores=fluid.layers.concat(scores, axis=2),
        score_threshold=score_threshold,
        nms_top_k=train_parameters['nms_top_k'],
        keep_top_k=train_parameters['nms_pos_k'],
        nms_threshold=train_parameters['nms_thresh'],
        background_label=-1,
        name="multiclass_nms")

    return pred
    
    
def infer(image):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, tensor_img, resized_img = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = (time.time() - t1)*1000
    print("predict cost time:{0}".format("%2.2f ms" % period))
    bboxes = np.array(batch_outputs[0])
    #print(bboxes)
    if bboxes.shape[1] != 6:
        # print("No object found")
        return False, [], [], [], [], period
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    return True, boxes, labels, scores, bboxes, period
    
    
    
    

image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
origin, tensor_img, resized_img = read_image(image)
#These two modes are depend on hardwares
trt_engine_path = "./fast_yolo.trt"
# Build an cudaEngine
engine = get_engine(onnx_model_path, trt_engine_path)
# 创建CudaEngine之后,需要将该引擎应用到不同的卡上配置执行环境
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# Do inference
shape_of_output = (max_batch_size, 1000)
# Load data to the buffer
inputs[0].host = tensor_img #.reshape(-1)

# inputs[1].host = ... for multiple input
t1 = time.time()
pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
# feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

print(type(pred))
print()
