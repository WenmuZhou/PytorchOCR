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

from ..utils import log_path, log_file, log
from ..weight_init.special_init.special_init_pool import global_special_init_pool as init_pool

from itertools import zip_longest
import numpy
import os
import json


global_compare_configs = {
    "atol": 0,
    "rtol": 1e-7,
    "compare_mode": "mean",
}


def update_configs(cfg):
    global global_compare_configs
    assert isinstance(cfg, dict)
    for k in cfg.keys():
        assert k in global_compare_configs
    global_compare_configs.update(cfg)
    return global_compare_configs


def clone_dict_tree(root):
    new_root = {}
    new_root.update(root)
    new_root["origin_node"] = root
    new_root["reordered"] = False
    new_root["children"] = []
    for child in root["children"]:
        new_root["children"].append(clone_dict_tree(child))
    return new_root


def _get_all_rank_path(path):
    all_files = sorted(os.listdir(path))
    all_files_check = ["rank_" in file for file in all_files]
    if all(all_files_check):
        return [os.path.join(path, file) for file in all_files]
    else:
        return [path]


def _get_all_step_path(path):
    all_files = sorted(os.listdir(path))
    all_files_check = ["step_" in file for file in all_files]
    if all(all_files_check):
        return [os.path.join(path, file) for file in all_files]
    else:
        return [path]


def get_all_valid_path(report_path_0, report_path_1):
    all_steps_path_0 = _get_all_step_path(report_path_0)
    all_steps_path_1 = _get_all_step_path(report_path_1)
    assert len(all_steps_path_0) == len(all_steps_path_1)
    for step_path_0, step_path_1 in zip(all_steps_path_0, all_steps_path_1):
        all_ranks_path_0 = _get_all_rank_path(step_path_0)
        all_ranks_path_1 = _get_all_rank_path(step_path_1)
        assert len(all_ranks_path_0) == len(all_ranks_path_1)
        return all_ranks_path_0, all_ranks_path_1
    raise ValueError("reach illegal code, concat the developer")


def parse_cfg(cfg):
    global global_compare_configs
    if cfg is None:
        return global_compare_configs
    return update_configs(cfg)


def print_report_info(nodes, reports, exc, stage, msg=None):

    log("FAILED !!!")

    if msg is not None:
        log("ADDITIONAL MESSAGE:")
        print(msg + "\n")
        log("DIFF DETAILS:")
    log(f"    Diff found in {stage} Stage")
    log(f"    Type of layer is: {nodes[0]['name']} vs {nodes[1]['name']}")
    log(f"    Route: {nodes[0]['route']}")
    log(f"           {nodes[1]['route']}\n")

    print(f"{type(exc).__name__}: {str(exc)} \n")

    log("Check model struct:")
    retstr = struct_info_log(reports, [node["origin_node"] for node in nodes], "report")
    print(retstr)


def struct_info_log(reports, nodes, file_prefix):
    file_names = []
    for idx in range(2):
        node = nodes[idx]
        report = reports[idx]
        file_name = build_file_name(report, file_prefix + "_" + report["model_name"])
        file_names.append(file_name)
        title = f"{report['model_name']}\n" + "=" * 40 + "\n"
        retval = []
        for tree in report["tree"]:
            retval.extend(tree_print(tree, mark=node, prefix=[" " * 4]))
        info = title + "\n".join(retval)
        log_file(file_name, "w", info)

    retval = f"Logs: {log_path}/{file_names[0]}\n"
    retval += f"      {log_path}/{file_names[1]}\n"
    return retval


def tree_print(node, mark=None, prefix=[]):
    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += node["name"]
    if "available" in node and node["available"] == False:
        cur_str += " (skip)"
    if os.getenv("PADIFF_PATH_LOG") == "ON":
        cur_str += "  (" + node["route"] + ")"
    if mark is node:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]
    for i, child in enumerate(node["children"]):
        pre = " |--- "
        if i == len(node["children"]) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = tree_print(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


# reorder second tree based on the first one
def reorder_and_match_sublayers(nodes, reports):
    if len(nodes[0]["children"]) == 0 and len(nodes[1]["children"]) == 0:
        return

    # split children to 3 parts
    base_apis = list(filter(lambda x: x["type"] == "api", nodes[0]["children"]))
    base_opaque_layers = list(filter(lambda x: x["type"] == "in map", nodes[0]["children"]))
    base_layers = list(filter(lambda x: x["type"] == "net", nodes[0]["children"]))

    raw_apis = list(filter(lambda x: x["type"] == "api", nodes[1]["children"]))
    raw_opaque_layers = list(filter(lambda x: x["type"] == "in map", nodes[1]["children"]))
    raw_layers = list(filter(lambda x: x["type"] == "net", nodes[1]["children"]))

    try:
        assert len(base_apis) == len(raw_apis), "number of api is different"
        assert len(base_opaque_layers) == len(raw_opaque_layers), "number of opaque_layers is different"
        assert len(base_layers) == len(raw_layers), "number of normal layer is different"

        # reset orders
        reorder_api(base_apis, raw_apis)
        layer_map = dict(zip(reports[0]["layer_map"], reports[1]["layer_map"]))
        reorder_opaque_layers(base_opaque_layers, raw_opaque_layers, layer_map)
        reorder_normal_layers(base_layers, raw_layers)

        # for every child in nodes[0], find correspond child in nodes[1]
        new_children = []
        for child in nodes[0]["children"]:
            if child["type"] == "api":
                new_children.append(raw_apis[0])
                raw_apis.pop(0)
            elif child["type"] == "in map":
                new_children.append(raw_opaque_layers[0])
                raw_opaque_layers.pop(0)
            elif child["type"] == "net":
                new_children.append(raw_layers[0])
                raw_layers.pop(0)
            else:
                raise RuntimeError("Invalid node type")

        nodes[1]["children"] = new_children
        nodes[1]["reordered"] = True

    except Exception as e:
        raise e


def reorder_api(apis, base):
    """
    reorder apis based on base
    TODO(wuzhafnei): need better match logic there
    Temporarily, just keep in order
    """
    return


def swap(seq, l, r):
    temp = seq[l]
    seq[l] = seq[r]
    seq[r] = temp
    return


def reorder_opaque_layers(base_nodes, raw_nodes, layer_map):
    for idx, base_node in enumerate(base_nodes):
        # an api layer can not have in_layer_map mark, so node.net is save
        correspond_route = layer_map["route"][base_node["route"]]
        correspond_node = next(node for node in raw_nodes if node["route"] == correspond_route)
        item_idx = raw_nodes.index(correspond_node)
        if item_idx == idx:
            continue
        elif item_idx > idx:
            swap(raw_nodes, item_idx, idx)
        else:
            raise RuntimeError("Duplicate key or values, check your LayerMap")

    return


def reorder_normal_layers(base_nodes, raw_nodes):
    # we suppose that: corresponding layers have same net_id
    bucket = {}
    for node in raw_nodes:
        key = node["metas"]["net_id"]
        if key not in bucket:
            bucket[key] = [node]
        else:
            bucket[key].append(node)

    raw_nodes.clear()
    for node in base_nodes:
        correspond_node = bucket[node["metas"]["net_id"]].pop(0)
        raw_nodes.append(correspond_node)


def traversal_node(node, node_list=[]):
    if node["available"]:
        node_list.append(node)
    for child in node["children"]:
        traversal_node(child, node_list)
    return node_list


def build_file_name(report, file_name):
    strs = report["file_path"].split("/")
    for s in reversed(strs):
        if "step_" in s:
            return file_name + "_" + s + ".log"
    return file_name


def load_numpy(path):
    if path is None:
        return None
    return numpy.load(path)


def load_json(path, report_name):
    with open(path + "/" + report_name, "r") as f:
        retval = json.load(f)
    return retval


def check_layer_map(reports):
    if len(reports[0]["layer_map"]["route"]) == 0 and len(reports[1]["layer_map"]["route"]) == 0:
        return True
    log("Start check layer_map:")
    layer_maps = [zip(rep["layer_map"]["route"], rep["layer_map"]["fullname"]) for rep in reports]
    for base_info, raw_info in zip_longest(layer_maps[0], layer_maps[1], fillvalue=None):
        if raw_info is None or base_info is None:
            print(
                "\nError: The number of submodels which need special init is not the same! Check your layer_map first!"
            )
            return False

        base_route, base_fullname = base_info
        raw_route, raw_fullname = raw_info
        func_key = base_fullname + "###" + raw_fullname

        if func_key in init_pool.funcs.keys():
            print(
                f"++++    base_model `{base_fullname}` at `{base_route}` <==>  raw_model `{raw_fullname}` at `{raw_route}`   ++++"
            )
        else:
            print("\nError: When check layer_map in order, find that raw_model can not matchs base_model.")
            print(f"    base_model:  `{base_fullname}` at `{base_route}`")
            print(f"    raw_model: `{raw_fullname}` at `{raw_route}`")
            log("Check layer_map FAILED!!!\n")
            return False

    log("Check layer_map SUCCESS!!!\n")
    return True
