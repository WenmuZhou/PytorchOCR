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

import json
import os, sys
import numpy
from .utils import Counter, reset_dir


dump_root_path = os.path.join(sys.path[0], "padiff_dump")


def set_dump_root_path(path):
    global dump_root_path
    dump_root_path = path


def get_dump_root_path():
    return dump_root_path


def numpy_dumper(path, prefix):
    reset_dir(path)
    counter = Counter()

    def dumper(value):
        id_ = counter.get_id()
        file_name = f"{path}/{prefix}_{id_}.npy"
        numpy.save(file_name, value)
        return file_name

    return dumper


"""
    dump tools for runtime reports
"""


def dump_report(model, dump_path):
    report = model.report
    tensor_path = dump_path + "/tensors"
    tensor_dumper = numpy_dumper(tensor_path, "tensor")
    report_info = {
        "model_name": model.name,
        "framework": model.framework,
        "file_path": f"{dump_path}/report.json",
        "layer_map": {
            "route": [mod.route for mod in model.marker.layer_map],
            "fullname": [mod.fullname for mod in model.marker.layer_map],
        },
        "tree": [dump_report_node(root, tensor_dumper) for root in report.stack.root],
    }
    with open(f"{dump_path}/report.json", "w") as fp:
        json.dump(report_info, fp, indent=4)


def dump_report_node(wrap_node, tensor_dumper):
    node_info = {
        "name": wrap_node.net_str,
        "route": wrap_node.net.route if hasattr(wrap_node.net, "route") else "",
        "type": wrap_node.layer_type,
        "fwd_outputs": [],
        "bwd_grads": [],
        "metas": {
            "fwd_step": wrap_node.fwd_report.step,
            "bwd_step": wrap_node.bwd_report.step,
            "net_id": wrap_node.fwd_report.net_id,
        },
        "children": [],
    }
    for tensor in wrap_node.fwd_report.tensors_for_compare():
        file_name = tensor_dumper(tensor.detach().numpy())
        node_info["fwd_outputs"].append(file_name)

    for tensor in wrap_node.bwd_report.tensors_for_compare():
        file_name = tensor_dumper(tensor.detach().numpy())
        node_info["bwd_grads"].append(file_name)

    for child in wrap_node.children:
        child_info = dump_report_node(child, tensor_dumper)
        node_info["children"].append(child_info)

    return node_info


"""
    dump tools for model parameters
"""


def dump_param_prototype(model, dump_fn, file_path):
    def dump_param_with_fn(model, fn, target_models):
        param_info = {
            "name": model.class_name,
            "route": model.route,
            "repr": model.model_repr_info(),
            "available": False,
            "weights": {},
            "grads": {},
            "children": [],
        }
        if model.model in target_models:  # only record sublayers specified by marker
            param_info["available"] = True
            for param_name, param in model.named_parameters(recursively=False):
                fn(param_name, param, param_info)
        for name, child in model.named_children():
            param_info["children"].append(dump_param_with_fn(child, fn, target_models))
        return param_info

    target_models = [layer.model for layer in model.marker.traversal_for_assign_weight()]
    param_info = dump_param_with_fn(model, dump_fn, target_models)

    model_info = {
        "model_name": model.name,
        "framework": model.framework,
        "file_path": file_path,
        "tree": param_info,
    }
    with open(file_path, "w") as fp:
        json.dump(model_info, fp, indent=4)


def dump_params(model, path):
    weight_dumper = numpy_dumper(path + "/weights", "weights")
    grad_dumper = numpy_dumper(path + "/grads", "grads")

    def _dump(param_name, param, param_info):
        file_name = weight_dumper(param.numpy())
        param_info["weights"][param_name] = file_name
        if param.main_grad() is not None:
            file_name = grad_dumper(param.main_grad())
            param_info["grads"][param_name] = file_name
        elif param.grad() is not None:
            file_name = grad_dumper(param.grad())
            param_info["grads"][param_name] = file_name
        else:
            param_info["grads"][param_name] = None

    dump_param_prototype(model, _dump, f"{path}/params.json")


def dump_weights(model, path):
    weight_dumper = numpy_dumper(path + "/weights", "weights")

    def _dump(param_name, param, param_info):
        file_name = weight_dumper(param.numpy())
        param_info["weights"][param_name] = file_name

    dump_param_prototype(model, _dump, f"{path}/weights.json")


def dump_grads(model, path):
    grad_dumper = numpy_dumper(path + "/grads", "grads")

    def _dump(param_name, param, param_info):
        if param.main_grad() is not None:
            file_name = grad_dumper(param.main_grad())
            param_info["grads"][param_name] = file_name
        elif param.grad() is not None:
            file_name = grad_dumper(param.grad())
            param_info["grads"][param_name] = file_name
        else:
            param_info["grads"][param_name] = None

    dump_param_prototype(model, _dump, f"{path}/grads.json")
