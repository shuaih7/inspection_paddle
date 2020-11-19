#!/usr/bin/env python2

from __future__ import print_function
import time, glob
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

#from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.
    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

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

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = '/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/fast_yolo.onnx'
    engine_file_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/fast_yolo.trt"
    # Download a dog image and save it to the following file path:
    input_image_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/sample.png"

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (352, 352)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 18, 11, 11)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        # start = time.time()
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # print("time: %.2f s" %(time.time()-start))
        # print(trt_outputs)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    postprocessor_args = {"yolo_masks": [(0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(188,15), (351,16), (351,30)],  # A list of 9 two-dimensional tuples for the YOLO anchors],
                          "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.2,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    # obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    # output_image_path = 'dog_bboxes.png'
    # obj_detected_img.save(output_image_path, 'PNG')
    # print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

    return boxes, classes, scores
    

def batch_infer(image_path, onnx_file_path, engine_file_path):
    img_list = gb.glob(image_path + r"/*.png")

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (352, 352)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Create a post-processor object by specifying the required input resolution for YOLOv3
    postprocessor_args = {"yolo_masks": [(0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(188,15), (351,16), (351,30)],  # A list of 9 two-dimensional tuples for the YOLO anchors],
                          "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.2,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = input_resolution_yolov3_HW

    # Output shapes expected by the post-processor
    output_shapes = [(1, 18, 11, 11)]
    # Do inference with TensorRT
    total_time, trt_outputs = 0, []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for i, img_file in enumerate(img_list):
            start = time.time()
            image_raw, image = preprocessor.process(input_image_path)
            inputs[0].host = image
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
            
            total_time += time.time() - start
            
        ave_time = total_time / (i+1)
        fps = 1.0/ave_time
        
    print("The averaged fps is", fps)
    
    
def batch_show(image_path, image_save_path, onnx_file_path, engine_file_path):
    img_list = gb.glob(image_path + r"/*.png")

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (352, 352)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Create a post-processor object by specifying the required input resolution for YOLOv3
    postprocessor_args = {"yolo_masks": [(0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(188,15), (351,16), (351,30)],  # A list of 9 two-dimensional tuples for the YOLO anchors],
                          "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.2,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = input_resolution_yolov3_HW

    # Output shapes expected by the post-processor
    output_shapes = [(1, 18, 11, 11)]
    # Do inference with TensorRT
    total_time, trt_outputs = 0, []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for i, img_file in enumerate(img_list):
            image_raw, image = preprocessor.process(input_image_path)
            inputs[0].host = image
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
            image_show = draw_bboxes(Image.fromarray(image_raw), boxes, scores, classes, ['defect'], bbox_color='green')
            
            # Save the marked image
            filename, suffix = os.path.split(img_file)
            _, fname = os.path.splitext(filename)
            save_name = os.path.join(image_save_path, fname+suffix)
            image_show.save(save_name)
            print("Image", save_name, "saved.")
    

if __name__ == '__main__':
    img_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/valid"
    img_save_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/valid_output"

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = '/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/fast_yolo.onnx'
    engine_file_path = "/home/nvidia/Documents/Projects/Fabric_defect_detection/YOLO/fast_yolo.trt"

    #batch_infer(img_path, onnx_file_path, engine_file_path)
    batch_show(img_path, img_save_path, onnx_file_path, engine_file_path)
