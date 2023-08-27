import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import copy
import time

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.utils.logging import get_logger
from torchocr.utils.utility import get_image_file_list, check_and_read
from torchocr.data import create_operators, transform
from tools.infer.onnx_engine import ONNXEngine
from tools.infer.utility import check_gpu, parse_args

logger = get_logger()


class TextClassifier(ONNXEngine):
    def __init__(self, args):
        if args.cls_model_dir is None or not os.path.exists(args.cls_model_dir):
            raise Exception(f'args.cls_model_dir is set to {args.cls_model_dir}, but it is not exists')

        onnx_path = os.path.join(args.cls_model_dir, 'model.onnx')
        config_path = os.path.join(args.cls_model_dir, 'config.yaml')
        super(TextClassifier, self).__init__(onnx_path, args.use_gpu)

        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh

        cfg = Config(config_path).cfg
        self.ops = create_operators(cfg['Transforms'][1:])
        self.postprocess_op = build_post_process(cfg['PostProcess'])

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            tic = time.time()
            for ino in range(beg_img_no, end_img_no):
                data = {'image': img_list[ino]}
                norm_img = transform(data, self.ops)[0]
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            preds = self.run(norm_img_batch)[0]
            cls_result = self.postprocess_op(preds)
            elapse += time.time() - tic

            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[beg_img_no + rno] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[beg_img_no + rno] = cv2.rotate(img_list[beg_img_no + rno], 1)
        return img_list, cls_res, elapse


def main(args):
    args.use_gpu = check_gpu(args.use_gpu)

    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    img_list, cls_res, predict_time = text_classifier(img_list)
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))


if __name__ == "__main__":
    main(parse_args())
