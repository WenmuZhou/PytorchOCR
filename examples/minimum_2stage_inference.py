import math
import operator
import os
from functools import reduce

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms
import cv2
import numpy as np
from torchocr.networks.architectures.RecModels import *
from torchocr.networks.architectures.DetModels import *
from utils import StrLabelConverter

default_font_for_annotate = ImageFont.truetype('./田氏颜体大字库2.0.ttf', 20)

def get_data(_to_eval_directory, _to_eval_file, _transform):
    """
    将一个文件夹中所有的图片或特定图片转换为特定网络用tensor
    :param _to_eval_directory:  图像所在文件夹
    :param _to_eval_file:   需要进行评估的文件
    :param _transform:  eval需要用到的transform
    :return:    每张图片的tensor
    """
    available_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    # 找到有效图像
    all_to_eval = []
    if _to_eval_file is not None:
        assert os.path.splitext(_to_eval_file)[1].lower() in available_extensions, f'{_to_eval_file} 格式不支持'
        target_file_path = os.path.join(_to_eval_directory, _to_eval_file)
        assert os.path.exists(target_file_path), f'{target_file_path} 文件不存在'
        all_to_eval.append(target_file_path)
    else:
        assert os.path.exists(_to_eval_directory) and os.path.isdir(_to_eval_directory), f'{_to_eval_directory} 文件夹无效'
        for m_file in os.listdir(_to_eval_directory):
            if os.path.splitext(m_file)[1].lower() in available_extensions:
                all_to_eval.append(os.path.join(_to_eval_directory, m_file))
    for m_file in all_to_eval:
        m_pil_img = Image.open(m_file)
        yield m_file, m_pil_img, _transform().unsqueeze(0)


def plot_detect_result_on_img(_img, _polygons):
    """
    在图中将检测和识别的结果画出来
    :param _img:    对应图片
    :param _polygons:  文本所在的多边形
    :return:    将检测结果绘画到图中
    """
    to_return_img = _img.copy()
    to_draw = ImageDraw.Draw(to_return_img)
    for m_polygon_index, m_polygon in enumerate(_polygons):
        to_draw.polygon(m_polygon, outline="red")
        to_draw.text(m_polygon[0], f'{m_polygon_index}', font=default_font_for_annotate, fill='blue')
    return to_return_img


def mask2polygon(_mask):
    """
    将mask区域转换为多边形区域（凸包）
    :param _mask:   当前检测之后的处理后的结果（h*w，每个元素位置为0和1）
    :return:    当前mask所有的能够提取的多边形
    """
    assert len(_mask.shape) == 2
    to_return_polygons = []
    contours, _ = cv2.findContours((_mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        to_return_polygons.append(points)
    return to_return_polygons


def clockwise_points(_point_coords):
    """
    以左上角为起点的顺时针排序
    原理就是将笛卡尔坐标转换为极坐标，然后对极坐标的φ进行排序
    :param _point_coords:    待排序的点[(x,y),]
    :return:    排序完成的点
    """
    center_point = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), _point_coords), [len(_point_coords)] * 2))
    return sorted(_point_coords, key=lambda coord: (135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center_point))[::-1]))) % 360)


def assign_points_to_four_edges(_point_coords):
    """
    将所有的点抽象到四条虚拟边上面（所有都是基于凸包的基础上）
    简单思想就是有了第一个点，然后找到这个点最远点，则得到第三个点
    然后第一个点和第三个点之间的两个鞍点分别为第二个点和第四个点
    :param _point_coords:   所有关键点
    :return:    将四条边所关联的点组成四个list并返回
    """
    to_return_edges = []
    max_distance_index = 0
    max_distance = -1
    points_np = np.array(_point_coords)

    for i in range(1, len(_point_coords)):
        distance = np.linalg.norm(points_np[0] - points_np[i])
        if distance > max_distance:
            max_distance_index = i
            max_distance = distance
    angles = []
    for i in range(1, max_distance_index):
        # 计算每个两边之间的夹角
        m_angle_1 = math.atan2(points_np[i][1] - points_np[0][1],
                               points_np[i][0] - points_np[0][0])
        m_angle_2 = math.atan2(points_np[i][1] - points_np[max_distance_index][1],
                               points_np[i][0] - points_np[max_distance_index][0])
        angles.append(abs(m_angle_1 - m_angle_2) % 180)
    second_point_index = np.argmin(angles)
    angles.clear()
    for i in range(max_distance_index, len(_point_coords) - 1):
        m_angle_1 = math.atan2(points_np[i][1] - points_np[0][1],
                               points_np[i][0] - points_np[0][0])
        m_angle_2 = math.atan2(points_np[i][1] - points_np[max_distance_index][1],
                               points_np[i][0] - points_np[max_distance_index][0])
        angles.append(abs(m_angle_1 - m_angle_2) % 180)
    fourth_point_index = np.argmin(angles)
    to_return_edges.append(_point_coords[:second_point_index])
    to_return_edges.append(_point_coords[second_point_index:max_distance_index])
    to_return_edges.append(_point_coords[max_distance_index:fourth_point_index])
    to_return_edges.append(_point_coords[fourth_point_index:])
    return to_return_edges


def polygon_rectify_to_rectangle(_img, _polygon):
    """
    将扇形或环形的多边形区域转换为矩形区域，方便识别模型
    :param _img:    当前图像
    :param _polygon:    多边形区域
    :return:    矩形区域
    """
    pass


def polygon_to_rectangle_basic(_img, _polygon):
    """
    将多边形的最小面积四边形进行透视变换得到矩形
    适用于绝大部分场景
    :param  _img:   当前图像
    :param _polygon:    多边形区域
    :return:    对应的矩形区域(ndarray)
    """
    sorted_points = clockwise_points(_polygon)
    fours_edges = assign_points_to_four_edges(sorted_points)
    arc_lengths = [cv2.arcLength(m_edge, False) for m_edge in fours_edges]
    first_edge = (arc_lengths[0] + arc_lengths[2]) // 2
    second_edge = (arc_lengths[1] + arc_lengths[3]) // 2
    img_np = np.array(_img)
    M = cv2.getPerspectiveTransform([m_edge[0] for m_edge in fours_edges],
                                    [[0, 0], [first_edge, 0], [first_edge, second_edge], [0, second_edge]])
    warped_roi = cv2.warpPerspective(img_np, M, (first_edge, second_edge))
    if first_edge < second_edge:
        return cv2.rotate(warped_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return warped_roi


def detect_result_post_process(_detect_result, _detector_model_type):
    """
    对检测的结果根据检测算法类型进行后处理
    :param _detect_result:  检测结果
    :param _detector_model_type:    检测模型的类型
    :return:    后处理后的结果
    """
    to_return_polygons = []
    if detector_model_type in {'pse', 'pan'}:
        # 对label map进行多边形抽取以及聚合
        pass
    elif detector_model_type in {'db'}:
        # 对shrink map进行多边形抽取以及聚合
        pass
    elif detector_model_type in {'centernet', 'fcos', 'east', 'advancedeast'}:
        # 对于回归出来的直接是矩形框的，主要是做nms以及聚合
        pass
    elif detector_model_type in {'ctpn', 'yolo_ctpn'}:
        # 对ctpn类型的回归框进行连接得到多边形区域，并进行多边形聚合
        pass
    return to_return_polygons


def extract_tensor_for_recognize(_img, _polygons, _transform):
    """
    将检测得到的区域提取出来，处理后转换为tensor用于识别模型
    :param _img:    原始图像
    :param _polygons:   所有检测到的文本的多边形区域
    :param _transform:  需要对图像区域进行变换的部分
    :return:    由于文本区域长度不同，所以每次只yield一条数据
    """
    for m_polygon in _polygons:
        # 未来需要对多边形进行判断，不同类型的多边形使用不同的方案进行tensor的转换
        yield _transform(polygon_to_rectangle_basic(_img, m_polygon)).unsqueeze(0)


def recognition_result_post_process(_recognition_result, _recognizer_model_type, _str_label_converter):
    """
    对识别的结果进行后处理，包括ctc decoder以及correct以及词典的矫正
    :param _recognition_result: 识别的结果
    :param _recognizer_model_type:  识别网路的类型
    :param _str_label_converter: 所有识别出来对应的字符标签
    :return:    进行后处理后的识别结果
    """
    if 'crnn' in _recognizer_model_type:
        return _str_label_converter.decode(_recognition_result)
    else:
        pass
    pass


def related_polygon_assembly(_polygons, _recognition_result):
    """
    将关联的多边形根据识别文本结果进行融合
    :param _polygons:   每个多边形
    :param _recognition_result: 每个多边形检测的结果
    :return:    融合后的多边形与对应的文本识别结果
    """
    return _polygons, _recognition_result


if __name__ == '__main__':
    # 配置参数
    device_name = 'cpu'
    eval_dataset_directory = './data'
    eval_file = None
    target_size = (1024, 1024)
    eval_stds = [0.229, 0.224, 0.225]
    eval_means = [0.485, 0.456, 0.406]


    def _resize_img_for_detect(_img, _longer_edge_length, _shorter_edge_base_length):
        """
        将图像按照最长边进行等比缩小且最短边能够被特定基数整除
        :param _img:    需要进行resize的图像
        :param _longer_edge_length:     最长边长度
        :param _shorter_edge_base_length:   短边的基数
        :return:    resize后的图像
        """
        h, w = _img.size
        if h > w:
            new_h = _longer_edge_length
            new_w = (new_h / h * w) // _shorter_edge_base_length * _shorter_edge_base_length
        else:
            new_w = _longer_edge_length
            new_h = (new_w / w * h) // _shorter_edge_base_length * _shorter_edge_base_length
        return _img.resize((new_w, new_h))


    def _resize_img_for_recognize(_img, _height=32):
        h, w = _img.size
        ratio = h / _height
        return _img.resize((w // ratio, _height))


    eval_detect_transformer = transforms.Compose([
        lambda x: _resize_img_for_detect(x, 2240, 32),
        transforms.Normalize(std=eval_stds, mean=eval_means),
        transforms.ToTensor(),
    ])
    eval_recognize_transformer = transforms.Compose([
        transforms.ToPILImage(),
        lambda x: _resize_img_for_recognize(x, 32),
        transforms.Normalize(std=[1, 1, 1], mean=[0.5, 0.5, 0.5]),
        transforms.ToTensor(),
    ])
    detector_model_type = 'db'
    recognizer_model_type = 'crnn_res'
    detector_pretrained_model_file = ''
    recognizer_pretrained_model_file = ''
    annotate_on_image = True
    need_rectify_on_single_character = True
    labels = ''.join([f'{i}' for i in range(10)] + [chr(97 + i) for i in range(26)])
    # 模型推断
    label_converter = StrLabelConverter(labels)
    device = torch.device(device_name)
    detector = get_detector(detector_model_type, detector_model_type).to(device)
    recognizer = get_recognizer(recognizer_model_type, recognizer_pretrained_model_file).to(device)
    detector.eval()
    recognizer.eval()
    with torch.no_grad():
        for m_path, m_pil_img, m_eval_tensor in tqdm(
                get_data(eval_dataset_directory, eval_file, eval_detect_transformer)
        ):
            m_eval_tensor = m_eval_tensor.to(device)
            # 获得检测需要的相关信息
            m_detect_result = detector(m_eval_tensor)
            # 根据网络类型，处理检测的相关信息，最后转换为一堆多边形
            m_polygons = detect_result_post_process(m_detect_result, detector_model_type)
            m_recognized_results = []
            # 提取所有的文本区域
            for m_region_tensor in extract_tensor_for_recognize(m_pil_img, m_polygons, eval_recognize_transformer):
                refined_recognized_result = \
                    recognition_result_post_process(recognizer(m_region_tensor), recognizer_model_type, label_converter)
                m_recognized_results.append(refined_recognized_result)
            m_final_polygons, m_final_text = related_polygon_assembly(m_polygons, m_recognized_results)
            if annotate_on_image:
                annotated_img = plot_detect_result_on_img(m_pil_img, m_final_polygons)
                m_base_name, m_ext = os.path.splitext(m_path)
                annotated_img.save(f'{m_base_name}_result{m_ext}')
                with open(f'{m_base_name}_result.txt', mode='w', encoding='utf-8') as to_write:
                    to_write.write('index,text\n')
                    for m_index, m_text in enumerate(m_final_text):
                        to_write.write(f'{m_index},{m_text}\n')
