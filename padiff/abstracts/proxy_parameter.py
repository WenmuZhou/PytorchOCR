# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import torch


class ProxyParam:
    def __init__(self, param, param_type):
        self.param = param
        self.param_type = param_type

    @staticmethod
    def create_from(param):
        if isinstance(param, ProxyParam):
            return param
        elif isinstance(param, paddle.framework.io.EagerParamBase):
            return PaddleParam(param)
        elif isinstance(param, torch.nn.parameter.Parameter):
            return TorchParam(param)
        else:
            raise RuntimeError(f"Can not create ProxyParam from {type(param)}")

    def numpy(self):
        raise NotImplementedError()

    def set_data(self, np_value):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def grad(self):
        raise NotImplementedError()

    def main_grad(self):
        raise NotImplementedError()


class PaddleParam(ProxyParam):
    def __init__(self, param):
        super(PaddleParam, self).__init__(param, "paddle")

    def numpy(self):
        return self.param.numpy()

    def set_data(self, np_value):
        paddle.assign(paddle.to_tensor(np_value), self.param)

    def shape(self):
        return list(self.param.shape)

    def grad(self):
        if self.param.grad is not None:
            return self.param.grad.numpy()
        else:
            return None

    def main_grad(self):
        if hasattr(self.param, "main_grad") and self.param.main_grad is not None:
            assert self.param.grad is None
            return self.param.main_grad.numpy()
        else:
            return None


class TorchParam(ProxyParam):
    def __init__(self, param):
        super(TorchParam, self).__init__(param, "torch")

    def numpy(self):
        return self.param.data.detach().cpu().numpy()

    def set_data(self, np_value):
        self.param.data = torch.as_tensor(np_value).type(self.param.dtype).to(self.param.device)

    def shape(self):
        return list(self.param.shape)

    def grad(self):
        if self.param.grad is not None:
            return self.param.grad.data.detach().cpu().numpy()
        else:
            return None

    def main_grad(self):
        return None
