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

from itertools import zip_longest
import numpy
import os

from ..utils import log, log_file, log_path
from ..datas import global_yaml_loader as yamls
from .special_init import build_name, global_special_init_pool as init_pool


# this interface is exposed, so it takes two models as inputs
def assign_weight_(base_model, raw_model):
    """
    Set weights in raw_model to the same as the values in base_model
    """

    models = (base_model, raw_model)
    for model in models:
        for mod in model.marker.traversal_for_assign_weight():
            setattr(mod.model, "need_init", True)

    for base_submodel, raw_submodel in zip(base_model.marker.layer_map, raw_model.marker.layer_map):
        key_name = build_name(
            base_model.framework, base_submodel.class_name, raw_model.framework, raw_submodel.class_name
        )
        if key_name not in init_pool.funcs.keys():
            log(
                "*** Special init `{}` and `{}` is not supported ***".format(
                    base_submodel.fullname, raw_submodel.fullname
                )
            )
            log("    Checkout the parameters are inited by yourself,")
            log("    or call `add_special_init` to register your init logic!")
        else:
            try:
                init_pool.funcs[key_name](base_submodel.model, raw_submodel.model)
            except Exception as e:
                print(f"Special init `{base_submodel.fullname}` and `{raw_submodel.fullname}` failed.")
                print(type(e).__name__ + ":  " + str(e))
                log("Assign weight Failed !!!")
                return False

    def _assign_weight(submodels, param_names, params, settings):
        check_shape(submodels, param_names, params, settings)
        np_value = params[1].numpy()
        if settings["transpose"]:
            np_value = numpy.transpose(np_value)

        params[0].set_data(np_value)

    try:
        process_each_weight(_assign_weight, models)
        log("Assign weight success !!!")
        return True
    except Exception as e:
        log("Assign weight Failed !!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False


def process_each_weight(process, models):
    submodels_0 = models[0].marker.traversal_for_assign_weight()
    submodels_1 = models[1].marker.traversal_for_assign_weight()

    for submodel_0, submodel_1 in zip_longest(submodels_0, submodels_1, fillvalue=None):
        if submodel_0 is None or submodel_1 is None:
            raise RuntimeError("Given models return difference number of sublayers. Check your model.")

        for (param_name_0, param_0), (param_name_1, param_1) in zip(
            submodel_0.named_parameters(recursively=False),
            submodel_1.named_parameters(recursively=False),
        ):
            try:
                settings = yamls.get_weight_settings(
                    (submodel_0.class_name, submodel_1.class_name),
                    (submodel_0.framework, submodel_1.framework),
                    (param_name_0, param_name_1),
                )
                process((submodel_0, submodel_1), (param_name_0, param_name_1), (param_0, param_1), settings)
            except Exception as e:
                err_str = f"Error occured when trying init weights, between:\n"
                err_str += f"    base_model: `{submodel_0.model_repr_info()}`\n"
                err_str += f"                `{submodel_0.route + '.' + param_name_0}`\n"
                err_str += f"    raw_model: `{submodel_1.model_repr_info()}`\n"
                err_str += f"               `{submodel_1.route + '.' + param_name_1}`\n"
                err_str += f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += fail_init_weight_log(models, (submodel_0, submodel_1))
                raise RuntimeError(err_str)


def check_shape(submodels, param_names, params, settings):
    shape_0 = params[0].shape()
    shape_1 = params[1].shape()
    if settings["transpose"]:
        shape_1.reverse()
    assert (
        shape_0 == shape_1
    ), f"Shape of param `{param_names[0]}` in {submodels[0].fullname} and param `{param_names[1]}` in {submodels[1].fullname} is not the same. {shape_0} vs {shape_1}\n"


def fail_init_weight_log(models, submodels):
    file_names = []
    for idx in range(2):
        model = models[idx]
        file_name = "weight_init_" + model.name + ".log"
        file_names.append(file_name)
        title = f"{model.name}\n" + "=" * 40 + "\n"
        retval = weight_struct_string(model, mark=submodels[idx], prefix=[" " * 4])
        info = title + "\n".join(retval)
        log_file(file_name, "w", info)

    retval = f"Weight init log saved to \n"
    retval += f"    {log_path}/{file_names[0]}\n"
    retval += f"    {log_path}/{file_names[1]}\n\n"
    retval += "Please view the reports and checkout the layer marked with `<---  *** HERE ***` !"

    retval += "\nHint:\n"
    retval += "    1. Check the definition order of params is same in submodels.\n"
    retval += "    2. Check the corresponding submodel have the same style:\n"
    retval += "       param <=> param, buffer <=> buffer, embedding <=> embedding ...\n"
    retval += "       cases like param <=> buffer, param <=> embedding are not allowed.\n"
    retval += "    3. If can not change model codes, try to use a `LayerMap`\n"
    retval += "       which can solve most problems.\n"
    retval += "    4. (skip) means this layer is skipped because it is under black_list, or it has no param.\n"
    retval += "    0. Visit `https://github.com/PaddlePaddle/PaDiff` to find more infomation.\n"

    return retval


def weight_struct_string(model, mark=None, prefix=[]):
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

    cur_str += str(model.class_name)

    if not hasattr(model.model, "need_init"):
        cur_str += "  (skip)"

    if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(model.model, "path_info"):
        cur_str += "  (" + model.path_info + ")"

    if mark.model is model.model:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]

    children = list(model.children())
    for i, child in enumerate(children):
        pre = " |--- "
        if i == len(children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = weight_struct_string(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs
