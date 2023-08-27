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

from ..utils import assert_tensor_equal
from .checker_utils import load_numpy

import warnings


class ActionPool:
    def __init__(self):
        self.pool = []

    def register(self, cls):
        name = cls.__name__
        self.pool.append(cls())
        sorted(self.pool, key=lambda x: x.priority, reverse=True)  # high -> low
        return cls

    def find_actions(self, report_0, node_0, report_1, node_1):
        for act in self.pool:
            if act.match(report_0, node_0, report_1, node_1):
                return act
        raise RuntimeError("No action is matched, not expected.")


global_actions = ActionPool()


def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)


class Action:
    def match(self, report_0, node_0, report_1, node_1):
        raise NotImplementedError("")

    def __call__(self, base_item, raw_item, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register
class EqualAction(Action):
    def match(self, report_0, node_0, report_1, node_1):
        return True

    @property
    def priority(self):
        return 0

    def __call__(self, file_list_0, file_list_1, cfg):
        assert len(file_list_0) == len(
            file_list_1
        ), f"number of tensors for compare is not equal, {len(file_list_0)} vs {len(file_list_1)}"
        for path_0, path_1 in zip(file_list_0, file_list_1):
            tensor_0 = load_numpy(path_0)
            tensor_1 = load_numpy(path_1)
            if tensor_0.size == 0 or tensor_1.size == 0:
                if tensor_0.size != tensor_1.size:
                    raise RuntimeError("size of tensors is not equal")
                warnings.warn("Found nparray.size == 0, compare skipped!")
                continue
            assert_tensor_equal(tensor_0, tensor_1, cfg)
