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

import numpy
from itertools import zip_longest

from .checker_utils import (
    struct_info_log,
    load_numpy,
    build_file_name,
    parse_cfg,
    load_json,
    traversal_node,
    get_all_valid_path,
)
from ..utils import log, log_file, log_path, assert_tensor_equal
from ..datas import global_yaml_loader as yamls


def check_params(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    log(f"Check params cfg: {cfg}")

    weight_rst = True
    grad_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "params.json"), load_json(path_1, "params.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        log(f"Checking params in {path_0} and {path_1}")
        weight_rst = weight_rst and check_target(assert_weight, node_lists, reports, "weights", cfg)
        grad_rst = grad_rst and check_target(assert_grad, node_lists, reports, "grads", cfg)
    return weight_rst and grad_rst


def check_weights(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    log(f"Check weights cfg: {cfg}")

    weight_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "weights.json"), load_json(path_1, "weights.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        log(f"Checking weights in {path_0} and {path_1}")
        weight_rst = weight_rst and check_target(assert_weight, node_lists, reports, "weights", cfg)
    return weight_rst


def check_grads(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    log(f"Check grads cfg: {cfg}")

    grad_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "grads.json"), load_json(path_1, "grads.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        log(f"Checking grads in {path_0} and {path_1}")
        grad_rst = grad_rst and check_target(assert_grad, node_lists, reports, "grads", cfg)
    return grad_rst


def check_target(fn, node_lists, reports, compare_target, cfg):
    flag = True
    log_name = build_file_name(reports[0], compare_target + "_diff")

    def checker(nodes, param_names, params, settings):
        try:
            fn(params, settings)
        except Exception as e:
            nonlocal flag
            flag = False
            info = (
                "=" * 25 + "\n" + "{} value is different.\n"
                "between base_model: {}\n"
                "        raw_model:  {}\n\n"
                "base_model param path:\n    {}\n"
                "raw_model param path:\n    {}\n\n"
                "{}\n\n".format(
                    compare_target,
                    nodes[0]["repr"],
                    nodes[1]["repr"],
                    nodes[0]["route"] + "." + param_names[0],
                    nodes[1]["route"] + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file(log_name, "a", info)

    try:
        process_each_param(checker, node_lists, reports, compare_target, cfg)
    except Exception as e:
        log("=" * 10 + f"Err occurs when compare {compare_target}!!!" + "=" * 10 + "\n")
        print(str(e))
        return False

    if flag == False:
        log(f"Diff found when compare {compare_target}, please check report \n        {log_path}/{log_name}")
    else:
        log(f"{compare_target} compared.")

    return flag


def process_each_param(process, node_lists, reports, compare_target, cfg):
    for node_0, node_1 in zip_longest(node_lists[0], node_lists[1], fillvalue=None):
        if node_0 is None or node_1 is None:
            raise RuntimeError("Found model with difference number of sublayers. Check your model.")
        for (param_name_0, param_path_0), (param_name_1, param_path_1) in zip(
            node_0[compare_target].items(),
            node_1[compare_target].items(),
        ):
            try:
                settings = yamls.get_weight_settings(
                    (node_0["name"], node_1["name"]),
                    (reports[0]["framework"], reports[1]["framework"]),
                    (param_name_0, param_name_1),
                )
                settings.update(cfg)
                param_0 = load_numpy(param_path_0)
                param_1 = load_numpy(param_path_1)
                process([node_0, node_1], [param_name_0, param_name_1], [param_0, param_1], settings)
            except Exception as e:
                err_str = f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += f"Error occured between:\n"
                err_str += f"    (base_model):  {node_0['route'] + '.' + param_name_0}\n"
                err_str += f"    (raw_model):   {node_1['route'] + '.' + param_name_1}\n\n"

                err_str += struct_info_log(reports, (compare_target, compare_target), compare_target)

                raise RuntimeError(err_str)


def assert_shape(params, settings):
    shape_0 = list(params[0].shape)
    shape_1 = list(params[1].shape)
    if settings["transpose"]:
        shape_1.reverse()
    assert shape_0 == shape_1, f"Shape not same. {shape_0} vs {shape_1}\n"


def assert_weight(params, settings):
    assert_shape(params, settings)
    if settings["transpose"]:
        params[1] = numpy.transpose(params[1])

    assert_tensor_equal(params[0], params[1], settings)


def assert_grad(params, settings):
    if params[0] is None and params[1] is None:
        return
    elif params[0] is None:
        raise RuntimeError(f"Found grad in base_model is `None`, when another is not!")
    elif params[1] is None:
        raise RuntimeError(f"Found grad in raw_model is `None`, when another is not!")
    assert_shape(params, settings)

    if settings["transpose"]:
        params[1] = numpy.transpose(params[1])

    assert_tensor_equal(params[0], params[1], settings)
