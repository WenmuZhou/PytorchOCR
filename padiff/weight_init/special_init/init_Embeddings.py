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

import paddle
import torch

from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register("torch", "Embeddings", "paddle", "Embeddings")
def init_Embeddings(module, layer):
    param_dict = {}
    for name, param in layer.state_dict().items():
        param_dict[name] = torch.from_numpy(param.cpu().detach().numpy())
    module.load_state_dict(param_dict)