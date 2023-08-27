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

from .diff_utils import pipeline, init_options, OptimizerHelper
from ..abstracts import ProxyModel
from ..abstracts.proxy_utils import init_route, remove_inplace
from ..weight_init import assign_weight_
from ..utils import assert_tensor_equal, log, reset_dir

from itertools import zip_longest
import paddle
import torch


def create_model(model, name=None, dump_freq=1):
    retval = ProxyModel.create_from(model, name, dump_freq)
    init_route(retval)
    if retval.framework == "paddle" and paddle.distributed.get_rank() % 8 == 0:
        # Only reset the root path once for each machine, here we assume each machine has 8 GPUs
        reset_dir(retval.dump_path)
    if retval.framework == "torch":
        reset_dir(retval.dump_path)
        remove_inplace(retval)
    return retval


def assign_weight(base_model, raw_model):
    """
    Set weights in raw_model to the same as the values in base_model
    """
    if not isinstance(raw_model, ProxyModel):
        raw_model = create_model(raw_model)
    if not isinstance(base_model, ProxyModel):
        base_model = create_model(base_model)

    return assign_weight_(base_model, raw_model)


def auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, **kwargs):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        base_model: paddle.nn.Layer or torch.nn.Module, provides the baseline of data precision.
        raw_model: paddle.nn.Layer or torch.nn.Module, which need to compare with base_model.
        inputs: input data for models, it should be a list of dict.
        loss_fns (list, optional): list of loss function for models.
        optimizers (list, optional): list of optimizers for models.
        kwargs: other options, view `https://github.com/PaddlePaddle/PaDiff` to learn more infomations
    Returns:
        True for success, False for failed.
    """

    options = kwargs

    if not isinstance(base_model, ProxyModel):
        base_model = create_model(base_model, base_model.__class__.__name__ + "_base_model")
    if not isinstance(raw_model, ProxyModel):
        raw_model = create_model(raw_model, raw_model.__class__.__name__ + "_raw_model")
    assert isinstance(inputs, (tuple, list)), "Invalid Argument."

    for input_ in inputs:
        assert isinstance(input_, dict), "Invalid Argument."

    if loss_fns is not None:
        options["use_loss"] = True
        assert len(loss_fns) == 2
        for loss in loss_fns:
            assert callable(loss), "Invalid loss function"
    else:
        loss_fns = [None, None]

    if optimizers is not None:
        options["use_opt"] = True
        assert len(optimizers) == 2
        for opt in optimizers:
            assert isinstance(opt, (paddle.optimizer.Optimizer, torch.optim.Optimizer)) or callable(
                opt
            ), "Invalid optimizer"
        optimizers = [OptimizerHelper(opt) for opt in optimizers]
    else:
        optimizers = [None, None]

    init_options(options)
    cfg = {}
    for key in ("atol", "rtol", "compare_mode"):
        cfg[key] = options[key]
        del options[key]

    if options["auto_init"] and not assign_weight(base_model, raw_model):
        return False

    result = pipeline((base_model, raw_model), inputs, loss_fns, optimizers, options, cfg)

    if result:
        log("SUCCESS !!!\n")
    else:
        log("FAILED !!!\n")

    return result


def check_dataloader(first_loader, second_loader, **kwargs):
    def get_numpy(data):
        if isinstance(data, (paddle.Tensor, torch.Tensor)):
            return data.detach().cpu().numpy()
        return data

    options = {
        "atol": 0,
        "rtol": 1e-7,
        "compare_mode": "mean",
    }
    options.update(kwargs)

    for data_0, data_1 in zip_longest(first_loader, second_loader, fillvalue=None):
        if data_0 is None or data_1 is None:
            raise RuntimeError("Given dataloader return difference number of datas.")
        try:
            assert_tensor_equal(get_numpy(data_0), get_numpy(data_1), options)
        except Exception as e:
            log("check dataloader failed!!!")
            print(f"{type(e).__name__ + ':  ' + str(e)}")
            return False
    return True
