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

import os
import sys
import shutil

import numpy as np
import paddle
import torch


from paddle.utils import flatten, pack_sequence_as, map_structure


"""
    clone tensor
"""


def is_tensor(x):
    return isinstance(x, (paddle.Tensor, torch.Tensor))


def is_tensors(*x):
    ret = True
    for i in x:
        ret = ret and is_tensor(i)
    return ret


def is_require_grad(x):
    if hasattr(x, "requires_grad"):
        return x.requires_grad
    if hasattr(x, "stop_gradient"):
        return not x.stop_gradient
    return False


def set_require_grad(x):
    if hasattr(x, "requires_grad"):
        x.requires_grad = True
    if hasattr(x, "stop_gradient"):
        x.stop_gradient = False


def _clone_tensor(inp):  # to cpu
    if isinstance(inp, (torch.Tensor, paddle.Tensor)):
        if inp.numel() == 0:
            if isinstance(inp, torch.Tensor):
                return torch.tensor([], dtype=inp.dtype)
            else:
                return paddle.to_tensor([], dtype=inp.dtype)
        new_t = inp.detach().cpu().clone()
        if is_require_grad(inp):
            set_require_grad(new_t)
        return new_t
    else:
        return inp


def clone_structure(inputs):
    return map_structure(_clone_tensor, inputs)


def clone_tensors(inputs):
    tensors = [_clone_tensor(t) for (t,) in for_each_tensor(inputs)]
    return tensors


"""
    traversal tools
"""


def for_each_tensor(*structure):
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    entries = filter(lambda x: is_tensors(*x), entries)
    for tensors in entries:
        yield tensors


def for_each_grad_tensor(*structure):
    def filter_fn(ts):
        return is_tensors(*ts) and is_require_grad(ts[0])

    for ts in filter(filter_fn, for_each_tensor(*structure)):
        yield ts


def map_structure_and_replace_key(func, structure1, structure2):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure1]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure2, [func(*x) for x in entries])


"""
    tensor compare or compute
"""


def assert_tensor_equal(tensor1, tensor2, cfg):
    """
    return None or raise Error.
    """
    atol = cfg.get("atol", 0)
    rtol = cfg.get("rtol", 1e-7)
    compare_mode = cfg.get("compare_mode", "mean")

    if compare_mode == "mean":
        np.testing.assert_allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    elif compare_mode == "strict":
        np.testing.assert_allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    elif compare_mode == "abs_mean":
        np.testing.assert_allclose(abs(tensor1).mean(), abs(tensor2).mean(), atol=atol, rtol=rtol)
    else:
        raise RuntimeError(f"Invalid compare_mode {compare_mode}")


"""
    process files
"""


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


"""
    log utils
"""
log_path = os.path.join(sys.path[0], "padiff_log")
__reset_log_dir__ = False  # reset log_path only once


def log_file(filename, mode, info):
    global __reset_log_dir__
    if not __reset_log_dir__:
        reset_dir(log_path)
        __reset_log_dir__ = True

    filepath = os.path.join(log_path, filename)
    with open(filepath, mode) as f:
        f.write(info)

    return filepath


def log(*args):
    print("[AutoDiff]", *args)


class Counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret
