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
import os
from ..utils import log
from ..weight_init.special_init.special_init_pool import global_special_init_pool as init_pool


class Marker:
    def __init__(self, proxy_model):
        self.proxy_model = proxy_model

        self.black_list = set()
        self.black_list_recursively = set()
        self.white_list = set()
        self.white_list_recursively = set()
        self.use_white_list = False
        self.unassigned_weights_list = set()
        self.unassigned_weights_list_recursively = set()
        self.layer_map = []

    def update_black_list(self, layers, mode="all"):
        assert mode in ("self", "sublayers", "all")
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        if mode in ("self", "all"):
            self.black_list.update(set(layers))
        if mode in ("sublayers", "all"):
            self.black_list_recursively.update(set(layers))

    def update_white_list(self, layers, mode="self"):
        assert mode in ("self", "sublayers", "all")
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        if mode in ("self", "all"):
            self.white_list.update(set(layers))
        if mode in ("sublayers", "all"):
            self.white_list_recursively.update(set(layers))
        self.use_white_list = True

    def update_unassigned_weights_list(self, layers, mode="all"):
        assert mode in ("self", "sublayers", "all")
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        if mode in ("self", "all"):
            self.unassigned_weights_list.update(set(layers))
        if mode in ("sublayers", "all"):
            self.unassigned_weights_list_recursively.update(set(layers))

    def set_layer_map(self, layer_map):
        _layer_map = []
        for layer in self.traversal_for_assign_weight():
            if layer.model in layer_map:
                self.unassigned_weights_list_recursively.add(layer.model)
                _layer_map.append(layer)

        self.layer_map = _layer_map

    def auto_layer_map(self, model_place):
        """
        Try to find components which support special init, and add them to layer_map automatically.
        NOTICE: this api suppose that all sublayers/submodules are defined in same order,
                if not, this may not work correctly.
        """
        _layer_map = []
        registered = init_pool.registered_base_models if model_place == "base" else init_pool.registered_raw_models

        log("Auto set layer_map start searching...")
        for layer in self.traversal_for_auto_layer_map():
            if layer.fullname in registered:
                print(f"++++    {model_place}_model found `{layer.fullname}` add to layer_map   ++++")
                _layer_map.append(layer)
                self.unassigned_weights_list.add(layer.model)
                self.unassigned_weights_list_recursively.add(layer.model)
        print()
        self.layer_map = _layer_map
        return True

    def update_black_list_with_class(self, layer_class, recursively=True):
        pass

    def update_white_list_with_class(self, layer_class, recursively=False):
        pass

    def traversal_for_hook(self):
        yield self.proxy_model
        for model in traversal_for_hook(self.proxy_model, self):
            if os.getenv("PADIFF_SIKP_WRAP_LAYER") == "OFF" and len(list(model.parameters(recursively=False))) == 0:
                continue
            yield model

    def traversal_for_assign_weight(self):
        yield self.proxy_model
        for model in traversal_for_assign_weight(self.proxy_model, self):
            if len(list(model.parameters(recursively=False))) == 0:
                continue
            yield model

    def traversal_for_auto_layer_map(self):
        yield self.proxy_model
        for model in traversal_for_assign_weight(self.proxy_model, self):
            yield model


def traversal_prototype(fn0, fn1):
    # if fn0 returns True, yield current model
    # if fn1 returns True, need traversal recursively
    def inner(model, marker):
        for child in model.children():
            if fn0(child, marker):
                yield child
            if fn1(child, marker):
                for sublayer in inner(child, marker):
                    yield sublayer

    return inner


traversal_all = traversal_prototype(
    fn0=lambda model, marker: True,
    fn1=lambda model, marker: True,
)
traversal_with_black_list = traversal_prototype(
    fn0=lambda model, marker: model.model not in marker.black_list,
    fn1=lambda model, marker: model.model not in marker.black_list_recursively,
)
traversal_layers_for_model_struct = traversal_prototype(
    fn0=lambda model, marker: model.model not in marker.black_list or model.model in marker.black_list_recursively,
    fn1=lambda model, marker: model.model not in marker.black_list_recursively,
)
traversal_layers_assign_weight = traversal_prototype(
    fn0=lambda model, marker: model.model not in marker.unassigned_weights_list,
    fn1=lambda model, marker: model.model not in marker.unassigned_weights_list_recursively,
)


def traversal_with_white_list(model, marker):
    for child in model.children():
        if child.model in marker.white_list:
            yield child
        if child.model in marker.white_list_recursively:
            for sublayer in traversal_all(child, marker):
                yield sublayer
        else:
            for sublayer in traversal_with_white_list(child, marker):
                yield sublayer


def traversal_for_hook(model, marker):
    if marker.use_white_list:
        for mod in traversal_with_white_list(model, marker):
            yield mod
    else:
        for mod in traversal_layers_for_model_struct(model, marker):
            yield mod


def traversal_for_assign_weight(model, marker):
    for mod in traversal_layers_assign_weight(model, marker):
        yield mod
