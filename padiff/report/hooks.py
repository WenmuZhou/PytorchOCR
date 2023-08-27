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

import contextlib
from functools import partial
from .report import current_report
from ..utils import (
    clone_tensors,
    map_structure_and_replace_key,
    flatten,
    for_each_grad_tensor,
)
import json
import numpy
import paddle
import torch


@contextlib.contextmanager
def register_hooker(model):
    marker = model.marker

    remove_handles = []
    idx = 0
    # traversal_for_hook includes layers which we need add pre and post hook
    # for model structure, but not info hook (they are in black list)
    models = list(marker.traversal_for_hook())
    for mod in models:
        pre_handle = mod.register_forward_pre_hook(partial(pre_structure_hook))
        if mod not in marker.black_list:
            handle = mod.register_forward_post_hook(partial(info_hook, net_id=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_post_hook(partial(post_structure_hook))
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


"""
    hooks used to build module structure
"""


def pre_structure_hook(layer, input):
    report = current_report()
    report.stack.push_layer(layer)
    if layer in report.marker.layer_map:
        report.stack._top().layer_type = "in map"
    return None


def post_structure_hook(layer, input, output):
    report = current_report()
    retval = report.stack.pop_layer(layer)
    if retval in report.marker.black_list:
        report.stack._top().children.pop()
    return None


"""
    hook for record forward infos
"""

# do not enter api layer which is triggered under info_hook
__in_info_hook__ = False


def info_hook(model, input, output, net_id):
    """
    Notice: the model is a origin layer/module, not ProxyModel
    """
    global __in_info_hook__
    if __in_info_hook__:
        return None

    report = current_report()

    if report is None or report.stack._top() is None:
        return None

    # if this api is not processing tensors, do not create report
    if output is None or all([not isinstance(x, (paddle.Tensor, torch.Tensor)) for x in flatten(output)]):
        return None

    # if an api under black_list_recursively, do not create report
    # a layer under black_list_recursively will not register this hook, except it is a mapped layer
    # report.stack._top().net can not be an api layer !!!
    if report.stack._top().net in report.marker.black_list_recursively and hasattr(model, "__api__"):
        return None

    # if this api is called under layer/module provided by framework, skip it
    python_module = report.stack._top().net.__module__
    if hasattr(model, "__api__") and (python_module.startswith("paddle.") or python_module.startswith("torch.")):
        return None

    __in_info_hook__ = True

    # if current model is an api layer, we do not want to hold it
    if hasattr(model, "__api__"):
        _model = padiff_layer_str(model)
    else:
        _model = model

    new_in = clone_tensors(input)
    new_out = clone_tensors(output)
    fwd_item = report.put_item("forward", new_in, new_out, _model, net_id)
    bwd_item = report.put_item("backward", new_in, new_out, _model, net_id)
    bwd_item.set_forward(fwd_item)

    report.stack.push_api(_model, fwd_item, bwd_item)

    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i, net_id=net_id))

    # if under single step forward guard
    if single_step_state() == "forward" and net_id != -1:
        # two report_item with same id, the step_idx should be corresponded
        step_idx = len(list(filter(lambda x: x.type == "forward" and x.net_id == net_id, report.items))) - 1
        base_report_node = find_base_report_node(net_id, step_idx)

        retval = map_structure_and_replace_key(replace_forward_output(base_report_node), output, output)
        __in_info_hook__ = False
        return retval
    else:
        __in_info_hook__ = False
        return None


"""
    hook for record backward infos
"""


def tensor_hook(x_grad, bwd_item, nth_tensor, net_id):
    new_grad = clone_tensors(x_grad)
    bwd_item.set_input_grads(nth_tensor, new_grad[0])

    if single_step_state() == "backward" and net_id != -1:
        report = current_report()
        step_idx = (
            list(filter(lambda x: x.type == "backward" and x.net_id == net_id, report.items)).index(bwd_item) - 1
        )
        base_report_node = find_base_report_node(net_id, step_idx)

        value = numpy.load(base_report_node["bwd_grads"][nth_tensor])
        if isinstance(x_grad, paddle.Tensor):
            return paddle.to_tensor(value)
        else:
            return torch.as_tensor(value, device=x_grad.device)

    return x_grad


"""
    utils
"""


def padiff_layer_str(model):
    if isinstance(model, paddle.nn.Layer):
        return PaddleLayerStr(model)
    else:
        return TorchModuleStr(model)


class PaddleLayerStr(paddle.nn.Layer):
    def __init__(self, net):
        super(PaddleLayerStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


class TorchModuleStr(torch.nn.Module):
    def __init__(self, net):
        super(TorchModuleStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


single_step_phase = ""
single_step_base = None


@contextlib.contextmanager
def SyncStepGuard(diff_phase, report_path):
    global single_step_phase, single_step_base
    try:
        old_phase = single_step_phase
        old_base = single_step_base

        with open(report_path + "/" + "report.json", "r") as report_file:
            report = json.load(report_file)

        single_step_phase = diff_phase
        single_step_base = split_by_net_id(report)

        yield
    finally:
        single_step_phase = old_phase
        single_step_base = old_base


def split_by_net_id(report):
    bucket = {}

    def _traversal(node, bucket):
        net_id = node["metas"]["net_id"]
        if net_id == -1:
            return
        if net_id not in bucket:
            bucket[net_id] = [node]
        else:
            bucket[net_id].append(node)

        for child in node["children"]:
            _traversal(child, bucket)

    for tree in report["tree"]:
        _traversal(tree, bucket)

    for key in bucket:
        bucket[key].sort(key=lambda x: x["metas"]["fwd_step"])

    return bucket


def single_step_state():
    return single_step_phase


def find_base_report_node(net_id, step_idx):
    global single_step_base
    return single_step_base[net_id][step_idx]


def replace_forward_output(node):
    numpy_file_list = node["fwd_outputs"]
    cur_idx = 0

    def inner(input_):
        if isinstance(input_, (paddle.Tensor, torch.Tensor)):
            if cur_idx >= len(numpy_file_list):
                raise RuntimeError(
                    "In single step mode, try to replace tensor by dumpped numpy value, but the number of tensors and numpy is not equal. Maybe the models are not corresponded."
                )
            value = numpy.load(numpy_file_list[cur_idx])
            if isinstance(input_, paddle.Tensor):
                return paddle.to_tensor(value)
            else:
                return torch.as_tensor(value, device=input_.device)
        else:
            return input_

    return inner
