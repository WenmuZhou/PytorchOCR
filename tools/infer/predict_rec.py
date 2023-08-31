import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import math
import time

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.data import create_operators, transform
from torchocr.utils.logging import get_logger
from torchocr.utils.utility import get_image_file_list, check_and_read
from tools.infer.onnx_engine import ONNXEngine
from tools.infer.utility import check_gpu, parse_args

logger = get_logger()


class TextRecognizer(ONNXEngine):
    def __init__(self, args):
        if args.rec_model_dir is None or not os.path.exists(args.rec_model_dir):
            raise Exception(f'args.rec_model_dir is set to {args.rec_model_dir}, but it is not exists')

        onnx_path = os.path.join(args.rec_model_dir, 'model.onnx')
        config_path = os.path.join(args.rec_model_dir, 'config.yaml')
        super(TextRecognizer, self).__init__(onnx_path, args.use_gpu)

        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm

        cfg = Config(config_path).cfg
        self.ops = create_operators(cfg['Transforms'][1:])
        self.postprocess_op = build_post_process(cfg['PostProcess'])

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == 'nrtr':
                    norm_img = transform({'image':img_list[indices[ino]]}, self.ops)[0]
                else:
                    norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            preds = self.run(norm_img_batch)

            if len(preds) == 1:
                preds = preds[0]

            rec_result = self.postprocess_op({'res': preds})
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res, time.time() - st


def main(args):
    args.use_gpu = check_gpu(args.use_gpu)

    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info(f"error in loading image:{image_file}")
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    rec_res, _ = text_recognizer(img_list)
    for ino in range(len(img_list)):
        logger.info(f"result of {valid_image_file_list[ino]}:{rec_res[ino]}")


if __name__ == "__main__":
    main(parse_args())
