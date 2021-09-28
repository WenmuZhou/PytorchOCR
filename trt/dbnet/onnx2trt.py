import os
import tensorrt as trt
import onnx
import torchvision.transforms as transforms
import cv2, time
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import shutil
from torchocr.postprocess import build_post_process
from matplotlib import pyplot as plt

TRT_LOGGER = trt.Logger()
def get_engine(onnx_path: str, trt_path: str, min_shape: list,
             opt_shape: list, max_shape: list, fp16=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # Building engine
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                builder.strict_type_constraints = True
                if not builder.platform_has_fast_fp16:
                    print("FP16 not supported on this platform.")
            config.max_workspace_size = 2 ** 30  # 1GiB

            with open(onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            profiles = builder.create_optimization_profile()
            for idx, input in enumerate(inputs):
                profiles.set_shape(input.name, min=min_shape[idx], opt=opt_shape[idx], max=max_shape[idx])
            config.add_optimization_profile(profiles)

            with builder.build_engine(network, config) as engine, open(trt_path, "wb") as f:
                f.write(engine.serialize())

            return engine

    if os.path.exists(trt_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(trt_path))
        with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, input_shape=None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        dims = engine.get_binding_shape(binding)
        if dims[-1] == -1:
            assert(input_shape is not None)
            dims[0] = input_shape[0]
            dims[2] = input_shape[2]
            dims[3] = input_shape[3]

        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


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

def preprocess_image(imagepath, model_inputs):
    origin_img = cv2.imread(imagepath)  # BGR
    origin_height=origin_img.shape[0]
    origin_width=origin_img.shape[1]
    new_height=model_inputs[2]
    new_width=model_inputs[3]
    pad_img=cv2.resize(origin_img,(new_height,new_width))
    pad_img = pad_img[:, :, ::-1].transpose(2, 0, 1)
    pad_img = pad_img.astype(np.float32)
    pad_img /= 255.0
    pad_img = np.ascontiguousarray(pad_img)
    pad_img = np.expand_dims(pad_img, axis=0)
    return pad_img,(new_height,new_width),(origin_height,origin_width)

def do_infer_test(test_engine, imgpath, batchsize=1):
    assert (test_engine is not None)

    pose_dir = prepare_output_dirs('./output')
    count = 0
    model_inputs = [1, 3, 736, 736]
    model_outputs = [1, 1, 736, 736]
    inputs, outputs, bindings, stream = allocate_buffers(test_engine, model_inputs)


    context = test_engine.create_execution_context()
    context.active_optimization_profile = 0  # 增加部分
    origin_inputshape = context.get_binding_shape(0)
    # 增加部分
    if (origin_inputshape[-1] == -1):
        origin_inputshape[-2], origin_inputshape[-1] = (model_inputs[2], model_inputs[3])
        origin_inputshape[0] = 1
        context.set_binding_shape(0, (origin_inputshape))

    if os.path.isdir(imgpath):
        imagepaths = []
        for imagname in os.listdir(imgpath):
            temppath = os.path.join(imgpath, imagname)
            imagepaths.append(temppath)
    else:
        imagepaths = [imgpath]
    for tempimagepath in imagepaths:

        input_image, input_shape, input_orishape = preprocess_image(tempimagepath, model_inputs)

        # img_input = img[..., ::-1].copy()  # BGR to RGB
        inputs[0].host = np.array(input_image, dtype=np.float32,
                                  order='C')

        now = time.time()
        trt_outputs = do_inference_v2(context, bindings=bindings,
                                      inputs=inputs, outputs=outputs,
                                      stream=stream)

        preds = trt_outputs[0].reshape(model_outputs)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        cc = {'type': 'DBPostProcess'}
        post_process = build_post_process(cc)
        box_list, score_list, out_mask = post_process(preds, [input_shape], True)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []

        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img = draw_bbox(img, box_list)
        plt.imshow(img)
        plt.imshow(out_mask)
        plt.show()

        img_filem = os.path.join(pose_dir, 'pose_{:08d}_m.jpg'.format(count))
        img_files = os.path.join(pose_dir, 'pose_{:08d}_s.jpg'.format(count))
        count = count + 1
        cv2.imwrite(img_filem, out_mask)
        cv2.imwrite(img_files, img)

    del stream


if __name__ == '__main__':
    model_path = './dbnet.onnx'
    trt_engine = './dbnet.engine'

    batchsize = 1
    imgspath = '/home/wwe/ocr/opendata/art/test_part1_images'
    test_engine = get_engine(model_path,trt_engine, min_shape=[[1, 3, 640, 640]], opt_shape=[[1, 3, 1280, 736]],
             max_shape=[[16, 3, 1280, 1280]], fp16=False)

    do_infer_test(test_engine, imgspath, batchsize)