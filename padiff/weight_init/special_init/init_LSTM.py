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
import torch

from .special_init_pool import global_special_init_pool as init_pool
import paddle


@init_pool.register("torch", "LSTM", "paddle", "LSTM")
def init_LSTM(module, layer):
    for (name, paddle_param), torch_param in zip(
        layer.named_parameters(prefix="", include_sublayers=False),
        module.parameters(recurse=False),
    ):
        p_shape = list(paddle_param.shape)
        t_shape = list(torch_param.shape)

        assert p_shape == t_shape, ("While init LSTM, shape of param `{}` is not the same. {} vs {}\n").format(
            name, p_shape, t_shape
        )
        np_value = paddle_param.detach().cpu().numpy()
        torch_param.data = torch.from_numpy(np_value)
