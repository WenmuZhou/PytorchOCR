# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 15:25
# @Author  : zhoujun
import os

import numpy as np
import onnxruntime


class ONNXEngine:
    def __init__(self, onnx_path, use_gpu):
        """
        :param onnx_path:
        """
        if not os.path.exists(onnx_path):
            raise Exception(f'{onnx_path} is not exists')

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ],
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=providers
        )
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image_numpy):
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result
