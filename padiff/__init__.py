# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


__version__ = "0.3.0"


# for api -> Layer

import sys, os
import inspect
from functools import partial

from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import SourceFileLoader, ExtensionFileLoader, PathFinder

from .report.hooks import info_hook
from .datas import global_json_laoder as jsons


def module_filter(name):
    if name in jsons.paddle_apis.keys() or name in jsons.torch_apis.keys():
        return True, name.partition(".")[0]
    return False, None


class PaDiffFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in sys.modules.keys():
            return None
        found, module_type = module_filter(fullname)
        if found:
            spec = self.sys_find_spec(fullname, path, target)
            # ExtensionFileLoader loader .so and other lib file
            # if loader is None, python may use a _NamespaceLoader or other way to init
            if spec is not None and not isinstance(spec.loader, (ExtensionFileLoader)) and spec.loader is not None:
                spec._module_type = module_type
                spec.loader = PaDiffLoader(spec.loader)
            return spec
        return None

    # find module by using sys defined finders
    def sys_find_spec(self, fullname, path, target=None):
        for finder in sys.meta_path:
            if isinstance(finder, PaDiffFinder):
                continue
            try:
                find_spec = finder.find_spec
            except AttributeError:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                spec._finder = finder
                spec._fullname = fullname
                return spec

        return None


def wrap_func(fullname, func):
    def wrapped(*args, **kwargs):

        if fullname.startswith("paddle"):

            class PaddleApi(paddle.nn.Layer):
                def __init__(self, func):
                    super(PaddleApi, self).__init__()
                    self._func = func
                    self.__name__ = fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self._func(*args, **kwargs)

                def __str__(self):
                    return self.__name__

            layer = PaddleApi(func)
            # need idx to support single step, set idx -1 here to skip api in single step mode
            handle = layer.register_forward_post_hook(partial(info_hook, net_id=-1))

        elif fullname.startswith("torch"):

            class TorchApi(torch.nn.Module):
                def __init__(self, func):
                    super(TorchApi, self).__init__()
                    self.func = func
                    self.__name__ = fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

                def __str__(self):
                    return self.__name__

            layer = TorchApi(func)
            handle = layer.register_forward_hook(partial(info_hook, net_id=-1))

        else:
            raise RuntimeError("Required module_type is in [paddle, torch], but received {}".format(full_name))

        out = layer(*args, **kwargs)

        handle.remove()

        return out

    return wrapped


def wrap_method(method_fullname, method):
    def wrapped(tensor_obj, *args, **kwargs):

        if method_fullname.startswith("paddle"):

            class PaddleMethod(paddle.nn.Layer):
                def __init__(self, method):
                    super(PaddleMethod, self).__init__()
                    self._method = method
                    self.__name__ = method_fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self._method(tensor_obj, *args, **kwargs)

                def __str__(self):
                    return self.__name__

            layer = PaddleMethod(method)
            handle = layer.register_forward_post_hook(partial(info_hook, net_id=-1))

        elif method_fullname.startswith("torch"):

            class TorchMethod(torch.nn.Module):
                def __init__(self, method):
                    super(TorchMethod, self).__init__()
                    self._method = method
                    self.__name__ = method_fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self._method(tensor_obj, *args, **kwargs)

                def __str__(self):
                    return self.__name__

            layer = TorchMethod(method)
            handle = layer.register_forward_hook(partial(info_hook, net_id=-1))

        else:
            raise RuntimeError("Required module_type is in [paddle, torch], but received {}".format(method_fullname))

        out = layer(*args, **kwargs)

        handle.remove()

        return out

    return wrapped


def wrap_api_method(module):
    if module.__name__.startswith("paddle"):
        apis = jsons.paddle_apis[module.__name__]
    elif module.__name__.startswith("torch"):
        apis = jsons.torch_apis[module.__name__]
    else:
        apis = []

    for api in apis:
        if api in module.__dict__.keys():
            obj = module.__dict__[api]
            if (inspect.isfunction(obj) or inspect.isbuiltin(obj)) and not hasattr(obj, "padiff_wrapped"):
                module.__dict__[api] = wrap_func(module.__name__ + "." + api, obj)
                setattr(module.__dict__[api], "padiff_wrapped", True)

    def replace_method(local_tensor, method_fullname):
        method_name = method_fullname.rpartition(".")[2]
        if hasattr(local_tensor, method_name):
            origin_method = getattr(local_tensor, method_name)
            # callable member of torch.Tensor is methoddescriptor
            # but callable member of paddle.Tensor is function
            if not hasattr(origin_method, "padiff_wrapped") and (
                inspect.ismethoddescriptor(origin_method) or inspect.isfunction(origin_method)
            ):
                method_impl = wrap_method(method_fullname, origin_method)
                setattr(method_impl, "padiff_wrapped", True)
                setattr(local_tensor, method_name, method_impl)

    if os.getenv("PADIFF_TENSOR_METHOD") != "OFF":
        if module.__name__ == "paddle":
            local_tensor = module.Tensor
            for method_fullname in jsons.paddle_tensor_methods:
                replace_method(local_tensor, method_fullname)

        if module.__name__ == "torch":
            local_tensor = module.Tensor
            for method_fullname in jsons.torch_tensor_methods:
                replace_method(local_tensor, method_fullname)


class PaDiffLoader(Loader):
    def __init__(self, _loader):
        self._loader = _loader

    def exec_module(self, module):
        self._loader.exec_module(module)
        wrap_api_method(module)

    def create_module(self, spec):
        return None


if os.getenv("PADIFF_API_CHECK") == "ON":
    for name in jsons.TORCH_PATH:
        if name in sys.modules.keys():
            module = sys.modules[name]
            wrap_api_method(module)

    for name in jsons.PADDLE_PATH:
        if name in sys.modules.keys():
            module = sys.modules[name]
            wrap_api_method(module)

    sys.meta_path = [PaDiffFinder()] + sys.meta_path


import paddle
import torch

paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)

from .interfaces import *

__all__ = [
    "create_model",
    "check_report",
    "check_params",
    "check_weights",
    "check_grads",
    "assign_weight",
    "auto_diff",
    "check_dataloader",
    "set_dump_root_path",
    "get_dump_root_path",
    "add_special_init",
]
