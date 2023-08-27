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
import re

from .proxy_parameter import ProxyParam
from .proxy_utils import deco_iter
from .marker import Marker

from ..report import Report, report_guard, register_hooker
from ..utils import reset_dir, log
from ..dump_tools import dump_report, dump_params, dump_weights, dump_grads, get_dump_root_path


class ProxyModel:
    def __init__(self, model, name, framework, dump_freq=1):
        self.model = model
        self.framework = framework  # "paddle"/"torch"
        self.name = name

        self.marker = Marker(self)
        self.report = Report(self.marker)
        self.step = 0

        self.dump_path = get_dump_root_path() + "/" + self.name

        self.dump_freq = dump_freq
        if self.dump_freq > 1:
            log(
                "WARNING: after setting dump_freq > 1 please call dump_params, dump_weighs, dump_grads apis "
                "on the steps that can be divided by dump_freq. The calling of try_dump is intact."
            )

    @staticmethod
    def create_from(model, name=None, dump_freq=1):
        if name is None:
            name = model.__class__.__name__
        if isinstance(model, ProxyModel):
            return model
        elif isinstance(model, paddle.nn.Layer):
            retval = PaddleModel(model, name, dump_freq)
            return retval
        elif isinstance(model, torch.nn.Module):
            retval = TorchModel(model, name, dump_freq)
            return retval
        else:
            raise RuntimeError(f"Can not create ProxyModel from {type(model)}")

    @property
    def class_name(self):
        return self.model.__class__.__name__

    @property
    def fullname(self):
        return f"{self.framework}::{self.class_name}"

    def __str__(self, *args, **kwargs):
        return f"Model({self.fullname})"

    def __repr__(self, *args, **kwargs):
        return f"Model({self.fullname})"

    def model_repr_info(self):
        model = self.model
        extra_lines = []
        extra_repr = model.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        if len(extra_lines) == 1:
            repr_info = extra_lines[0]
        else:
            repr_info = ""

        retstr = model.__class__.__name__ + "(" + repr_info + ")"
        return retstr

    @property
    def route(self):
        return self.model.route

    """
        training
    """

    def __call__(self, *args, **kwargs):
        if self.step % self.dump_freq == 0:
            with register_hooker(self), report_guard(self.report):
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)

    def backward(self, loss):
        if self.step % self.dump_freq == 0:
            with register_hooker(self), report_guard(self.report):
                return loss.backward()
        else:
            return loss.backward()

    """
        black_list and white_list
        mode : "self" | "sublayers" | "all"
    """

    def update_black_list(self, layers, mode="all"):
        self.marker.update_black_list(layers, mode)

    def update_white_list(self, layers, mode="self"):
        self.marker.update_white_list(layers, mode)

    def update_unassigned_weights_list(self, layers, mode="self"):
        self.marker.update_unassigned_weights_list(layers, mode)

    def update_black_list_with_class(self, layer_class, mode="self"):
        all_sub_layers = []
        for sub_layer in self.submodels():
            if isinstance(sub_layer.model, layer_class):
                all_sub_layers.append(sub_layer.model)
        self.update_black_list(all_sub_layers, mode)

    def update_white_list_with_class(self, layer_class, mode="self"):
        all_sub_layers = []
        for sub_layer in self.submodels():
            if isinstance(sub_layer.model, layer_class):
                all_sub_layers.append(sub_layer.model)
        self.update_white_list(all_sub_layers, mode)

    def set_layer_map(self, layers):
        self.marker.set_layer_map(layers)

    def auto_layer_map(self, place):
        self.marker.auto_layer_map(place)

    """
        about dump
    """

    def clear_report(self):
        self.report = Report(self.marker)

    def try_dump(self, dump_path=None):
        if self.step % self.dump_freq == 0:
            if dump_path is None:
                dump_path = f"{self.dump_path}/step_{self.step}/rank_{paddle.distributed.get_rank()}"
            reset_dir(dump_path)
            self.dump_params(dump_path)
            self.dump_report(dump_path)
        self.clear_report()
        self.step += 1

    def dump_report(self, dump_path=None):
        if dump_path is None:
            dump_path = f"{self.dump_path}/rank_{paddle.distributed.get_rank()}"
        dump_report(self, dump_path)

    def dump_params(self, dump_path=None):
        if dump_path is None:
            dump_path = f"{self.dump_path}/rank_{paddle.distributed.get_rank()}"
        dump_params(self, dump_path)

    def dump_weights(self, dump_path=None):
        if dump_path is None:
            dump_path = f"{self.dump_path}/rank_{paddle.distributed.get_rank()}"
        dump_weights(self, dump_path)

    def dump_grads(self, dump_path=None):
        if dump_path is None:
            dump_path = f"{self.dump_path}/rank_{paddle.distributed.get_rank()}"
        dump_grads(self, dump_path)

    """
        support native interfaces
    """

    def parameters(self, recursively):
        raise NotImplementedError()

    def named_parameters(self, recursively):
        raise NotImplementedError()

    # child sublayers, do not include self
    def children(self):
        raise NotImplementedError()

    def named_children(self):
        raise NotImplementedError()

    # get sublayers recursively include self
    def submodels(self):
        raise NotImplementedError()

    def named_submodels(self):
        raise NotImplementedError()

    def get_device(self):
        raise NotImplementedError()

    def to_cpu(self):
        raise NotImplementedError()

    def to(self, device):
        raise NotImplementedError()

    def register_forward_pre_hook(self, hook):
        raise NotImplementedError()

    def register_forward_post_hook(self, hook):
        raise NotImplementedError()


class PaddleModel(ProxyModel):
    def __init__(self, model, name, dump_freq=1):
        super(PaddleModel, self).__init__(model, name, "paddle", dump_freq)

    def parameters(self, recursively):
        origin_iter = self.model.parameters(include_sublayers=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def named_parameters(self, recursively):
        origin_iter = self.model.named_parameters(include_sublayers=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def children(self):
        origin_iter = self.model.children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_children(self):
        origin_iter = self.model.named_children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def submodels(self):
        origin_iter = self.model.sublayers(True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_submodels(self):
        origin_iter = self.model.named_sublayers(include_self=True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def get_device(self):
        place_str = str(self.model.parameters()[0].place)
        return re.match(r"Place\((.*)\)", place_str).group(1)

    def to_cpu(self):
        self.model.to("cpu")
        paddle.device.cuda.empty_cache()

    def to(self, device):
        self.model.to(device)

    def register_forward_pre_hook(self, hook):
        return self.model.register_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        return self.model.register_forward_post_hook(hook)

    def _prepare_training(self, data, optimizer, lr_scheduler):
        # Only for Paddle's pipeline parallel used.
        return self.model._prepare_training(data, optimizer, lr_scheduler)

    def forward_backward_pipeline(self, data, scaler):
        # Only for Paddle's pipeline parallel used.
        if self.step % self.dump_freq == 0:
            with register_hooker(self), report_guard(self.report):
                return self.model.forward_backward_pipeline(data, scaler)
        else:
            return self.model.forward_backward_pipeline(data, scaler)

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        # Only for Paddle's pipeline parallel used.
        if self.step % self.dump_freq == 0:
            with register_hooker(self), report_guard(self.report):
                return self.model.train_batch(data, optimizer, lr_scheduler, scaler)
        else:
            return self.model.train_batch(data, optimizer, lr_scheduler, scaler)


class TorchModel(ProxyModel):
    def __init__(self, model, name, dump_freq=1):
        super(TorchModel, self).__init__(model, name, "torch", dump_freq)

    def parameters(self, recursively):
        origin_iter = self.model.parameters(recurse=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def named_parameters(self, recursively):
        origin_iter = self.model.named_parameters(recurse=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def children(self):
        origin_iter = self.model.children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_children(self):
        origin_iter = self.model.named_children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def submodels(self):
        origin_iter = self.model.modules()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_submodels(self):
        origin_iter = self.model.named_modules(remove_duplicate=True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def get_device(self):
        return next(self.model.parameters()).device

    def to_cpu(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def to(self, device):
        self.model.to(device)

    def register_forward_pre_hook(self, hook):
        return self.model.register_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        return self.model.register_forward_hook(hook)
