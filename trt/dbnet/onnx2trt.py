import os
import tensorrt as trt
import onnx
import torchvision.transforms as transforms
import cv2, time
import torch
import pycuda.driver as cuda
import numpy as np
import shutil
from torchocr.postprocess import build_post_process
from matplotlib import pyplot as plt

TRT_LOGGER = trt.Logger()
def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""

        # network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(network_creation_flag) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 256MiB
            builder.max_batch_size = 1

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))

            onnx.checker.check_model(onnx_file_path)

            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            print('network layers ',network.num_layers)

            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))

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


def prediction(test_engine, image, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []


    # hwc -> 1chw
    model_input = transform(image)#.unsqueeze(0)
    model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)
    model_inputs = model_inputs.numpy()
    # compute output heatmap

    h_output = np.empty(409600, dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(model_inputs.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()


    with test_engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, model_inputs, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
    # Return the host output.
    h_output = h_output.reshape(640, 640)

    return h_output


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path

def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir

def do_infer_test(test_engine, imgpath):
    pose_dir = prepare_output_dirs('./output')

    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    count = 0

    while 1:
        img_ori = cv2.imread(imgpath.format(str(count)))
        img = cv2.resize(img_ori, (640, 640), interpolation=cv2.INTER_CUBIC)
        # img_input = img[..., ::-1].copy()  # BGR to RGB


        now = time.time()
        preds = prediction(test_engine, img, pose_transform)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        cc = {'type': 'DBPostProcess'}
        post_process = build_post_process(cc)
        box_list, score_list, out_mask = post_process(preds, [img.shape[:2]], True)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_bbox(img, box_list)
        plt.imshow(img)
        plt.imshow(out_mask)
        plt.show()

        img_filem = os.path.join(pose_dir, 'pose_{:08d}_m.jpg'.format(count))
        img_files = os.path.join(pose_dir, 'pose_{:08d}_s.jpg'.format(count))
        count = count + 1
        cv2.imwrite(img_filem, out_mask)
        cv2.imwrite(img_files, img)


if __name__ == '__main__':
    model_path = './dbnet.onnx'
    trt_engine = './dbnet.engine'

    imgspath = './testimages/gt_{}.jpg'
    test_engine = get_engine(model_path,trt_engine)

    do_infer_test(test_engine, imgspath)