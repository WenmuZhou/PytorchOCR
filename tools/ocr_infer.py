from det_infer import DetInfer
from rec_infer import RecInfer
import argparse
from line_profiler import LineProfiler
from memory_profiler import profile
from torchocr.utils.vis import draw_ocr_box_txt
import numpy as np

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class OCRInfer(object):
    def __init__(self, det_path, rec_path, rec_batch_size=16, time_profile=False, mem_profile=False ,**kwargs):
        super().__init__()
        self.det_model = DetInfer(det_path)
        self.rec_model = RecInfer(rec_path, rec_batch_size)
        assert not(time_profile and mem_profile),"can not profile memory and time at the same time"
        self.line_profiler = None
        if time_profile:
            self.line_profiler = LineProfiler()
            self.predict = self.predict_time_profile
        if mem_profile:
            self.predict = self.predict_mem_profile

    def do_predict(self, img):
        box_list, score_list = self.det_model.predict(img)
        if len(box_list) == 0:
            return [], [], img
        draw_box_list = [tuple(map(tuple, box)) for box in box_list]
        imgs =[get_rotate_crop_image(img, box) for box in box_list]
        texts = self.rec_model.predict(imgs)
        texts = [txt[0][0] for txt in texts]
        debug_img = draw_ocr_box_txt(img, draw_box_list, texts)
        return box_list, score_list, debug_img

    def predict(self, img):
        return self.do_predict(img)

    def predict_mem_profile(self, img):
        wapper = profile(self.do_predict)
        return wapper(img)

    def predict_time_profile(self, img):
        # run multi time
        for i in range(8):
            print("*********** {} profile time *************".format(i))
            lp = LineProfiler()
            lp_wrapper = lp(self.do_predict)
            ret = lp_wrapper(img)
            lp.print_stats()
        return ret


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='OCR infer')
    parser.add_argument('--det_path', required=True, type=str, help='det model path')
    parser.add_argument('--rec_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--rec_batch_size', type=int, help='rec batch_size', default=16)
    parser.add_argument('-time_profile', action='store_true', help='enable time profile mode')
    parser.add_argument('-mem_profile', action='store_true', help='enable memory profile mode')
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    import cv2
    args = init_args()
    img = cv2.imread(args['img_path'])
    model = OCRInfer(**args)
    txts, boxes, debug_img = model.predict(img)
    h,w,_, = debug_img.shape
    raido = 1
    if w > 1200:
        raido = 600.0/w
    debug_img = cv2.resize(debug_img, (int(w*raido), int(h*raido)))
    if not(args['mem_profile'] or args['time_profile']):
        cv2.imshow("debug", debug_img)
        cv2.waitKey()

