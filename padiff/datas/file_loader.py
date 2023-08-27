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

import yaml
import os
import json


"""
    YamlLoader
"""


class YamlLoader:
    def __init__(self):
        yaml_path = os.path.join(os.path.dirname(__file__), "assign_weight.yaml")
        with open(yaml_path, "r") as yaml_file:
            self._assign_yaml = yaml.safe_load(yaml_file)
        self._options = {}

    def get_weight_settings(self, layer_names, framework_types, param_names):
        # only when paddle model compare with torch model, need to update settings
        if framework_types[0] == framework_types[1]:
            settings = {"transpose": False}
            return settings

        # settings are used to fix the diff between paddle and torch, so keep `paddle`` and `torch` infos here
        # currently, assign_weight.yaml only recorded transpose setting
        # transpose paddle or torch is the same, so it's no need to change current yaml file temporarily
        if framework_types[0] == "paddle":
            paddle_name = layer_names[0]
            torch_name = layer_names[1]
            param_name = param_names[0]
        else:
            paddle_name = layer_names[1]
            torch_name = layer_names[0]
            param_name = param_names[1]

        assign_config = self._assign_yaml.get(paddle_name, None)
        settings = {
            "transpose": False,
        }

        if assign_config is not None:
            assert (
                torch_name in assign_config["torch"]
            ), "Not correspond, paddle layer {}  vs torch module {}. check your __init__ to make sure every sublayer is corresponded, or view the model struct reports in diff_log.".format(
                paddle_name, torch_name
            )

        if (
            assign_config is None
            or assign_config.get("param", None) == None
            or param_name not in assign_config["param"]
        ):
            pass
        else:
            if assign_config["param"][param_name] == "transpose":
                settings["transpose"] = True

        return settings

    @property
    def assign_yaml(self):
        return self._assign_yaml

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, val):
        assert isinstance(val, dict)
        self._options.update(val)


global_yaml_loader = YamlLoader()


"""
    JsonLoader
"""


class JsonLoader:
    def __init__(self):
        self.TORCH_PATH = [
            "torch.nn.functional",
            "torch",
            "torch.linalg",
            "torch.fft",
        ]
        self.PADDLE_PATH = [
            "paddle.nn.functional",
            "paddle",
            "paddle.linalg",
            "paddle.fft",
            "paddle.incubate.sparse",
            "paddle.signal",
        ]

        json_path = os.path.join(os.path.dirname(__file__), "api_mapping.json")
        with open(json_path, "r") as file:
            api_mapping = json.load(file)

        self.torch_apis = {}
        self.paddle_apis = {}

        self.paddle_tensor_methods = set()
        self.torch_tensor_methods = set()

        for k, v in api_mapping.items():
            if "paddle_api" not in v.keys():
                continue

            torch_fullname = k
            paddle_fullname = v["paddle_api"]

            if torch_fullname.startswith("torch.Tensor.") and paddle_fullname.startswith("paddle.Tensor."):
                self.paddle_tensor_methods.add(paddle_fullname)
                self.torch_tensor_methods.add(torch_fullname)
                continue

            torch_module = torch_fullname.rpartition(".")[0]
            torch_api = torch_fullname.rpartition(".")[2]

            paddle_module = paddle_fullname.rpartition(".")[0]
            paddle_api = paddle_fullname.rpartition(".")[2]

            if torch_module not in self.TORCH_PATH or paddle_module not in self.PADDLE_PATH:
                continue

            if torch_module not in self.torch_apis.keys():
                self.torch_apis[torch_module] = {torch_api}
            else:
                self.torch_apis[torch_module].add(torch_api)

            if paddle_module not in self.paddle_apis.keys():
                self.paddle_apis[paddle_module] = {paddle_api}
            else:
                self.paddle_apis[paddle_module].add(paddle_api)

        # paddle.nn.Conv2D called _conv_nd, this layer is used frequently, so add it to ADDITIONAL_PATH
        self.ADDITIONAL_PATH = {"paddle.nn.functional.conv": {"_conv_nd"}}
        self.paddle_apis.update(self.ADDITIONAL_PATH)

        # Deprecated
        self.TORCH_IGNORE = {"torch.nn.functional": ["sigmoid"], "torch": ["as_tensor"]}
        self.PADDLE_IGNORE = {"paddle": ["to_tensor"]}
        for k, v in self.TORCH_IGNORE.items():
            for item in v:
                self.torch_apis[k].remove(item)
        for k, v in self.PADDLE_IGNORE.items():
            for item in v:
                self.paddle_apis[k].remove(item)

        self.MAGIC_METHOD = [
            "__add__",
            "__radd__",
            "__iadd__",
            "__sub__",
            "__rsub__",
            "__isub__",
            "__mul__",
            "__rmul__",
            "__div__",
            "__rdiv__",
            "__truediv__",
            "__rtruediv__",
            "__pow__",
            "__rpow__",
            "__floordiv__",
            "__mod__",
            "__matmul__",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__lt__",
            "__gt__",
            "__ge__",
        ]

        for magic_method in self.MAGIC_METHOD:
            self.paddle_tensor_methods.add("paddle.Tensor." + magic_method)
            self.torch_tensor_methods.add("torch.Tensor." + magic_method)

        self.IGNORE_METHOD = ["clone", "detach", "cpu", "gpu"]
        for ignore_method in self.IGNORE_METHOD:
            paddle_method = "paddle.Tensor." + ignore_method
            torch_method = "torch.Tensor." + ignore_method
            if paddle_method in self.paddle_tensor_methods:
                self.paddle_tensor_methods.remove(paddle_method)
            if torch_method in self.torch_tensor_methods:
                self.torch_tensor_methods.remove(torch_method)


if os.getenv("PADIFF_API_CHECK") == "ON":
    global_json_laoder = JsonLoader()
else:
    global_json_laoder = None
