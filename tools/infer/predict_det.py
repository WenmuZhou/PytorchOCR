import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import json
import numpy as np
import time

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.data.imaug import create_operators, transform
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from torchocr.utils.utility import get_image_file_list, check_and_read
from tools.infer.onnx_engine import ONNXEngine
from tools.infer.utility import check_gpu, parse_args

logger = get_logger()


class TextDetector(ONNXEngine):
    def __init__(self, args):
        if args.det_model_dir is None or not os.path.exists(args.det_model_dir):
            raise Exception(f'args.det_model_dir is set to {args.det_model_dir}, but it is not exists')

        onnx_path = os.path.join(args.det_model_dir, 'model.onnx')
        config_path = os.path.join(args.det_model_dir, 'config.yaml')
        super(TextDetector, self).__init__(onnx_path, args.use_gpu)
        self.args = args
        self.det_algorithm = args.det_algorithm
        cfg = Config(config_path).cfg

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(cfg['PostProcess'])

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        outputs = self.run(img)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs[0]
            preds['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs[0]
            preds['f_score'] = outputs[1]
            preds['f_tco'] = outputs[2]
            preds['f_tvo'] = outputs[3]
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['res'] = outputs[0]
        elif self.det_algorithm == 'FCE':
            for i, output in enumerate(outputs):
                preds['level_{}'.format(i)] = output
        elif self.det_algorithm == "CT":
            preds['maps'] = outputs[0]
            preds['score'] = outputs[1]
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, [-1, shape_list])
        dt_boxes = post_result[0]['points']

        if self.args.det_box_type == 'poly':
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et = time.time()
        return dt_boxes, et - st


def main(args):
    args.use_gpu = check_gpu(args.use_gpu)

    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)

    total_time = 0
    save_res_path = args.output
    os.makedirs(save_res_path, exist_ok=True)

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    with open(os.path.join(save_res_path, 'inference_det.txt'), "w") as fout:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info(f"error in loading image:{image_file}")
                continue

            tic = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - tic
            total_time += elapse

            dt_boxes_json = []
            # write result
            for box in dt_boxes:
                tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                dt_boxes_json.append(tmp_json)
            out_str = f'{image_file}\t{json.dumps(dt_boxes_json)}'
            fout.write(out_str + '\n')

            logger.info(out_str)
            logger.info(f"The predict time of {image_file}: {elapse}")

            save_path = os.path.join(save_res_path, f'inference_det_{os.path.basename(image_file)}')
            img = draw_det(dt_boxes, img)
            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main(parse_args())
