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


from .special_init_pool import global_special_init_pool as init_pool
import numpy
import torch
import paddle


@init_pool.register("torch", "MultiheadAttention", "paddle", "MultiHeadAttention")
def init_MultiHeadAttention(module, layer):
    name_param_dict = {}
    for param in module.named_parameters():
        pname = param[0]
        if "multihead_attn" in pname:
            pname = pname.replace("multihead_attn", "cross_attn")
        elif "in_proj" not in pname:
            continue

        if "in_proj_" in pname:
            size = param[1].shape[0] // 3
            q, k, v = torch.split(param[1], [size for _ in range(3)])

            name_param_dict[pname.replace("in_proj_", "q_proj.")] = q.data.detach().cpu().numpy()
            name_param_dict[pname.replace("in_proj_", "k_proj.")] = k.data.detach().cpu().numpy()
            name_param_dict[pname.replace("in_proj_", "v_proj.")] = v.data.detach().cpu().numpy()
        else:
            name_param_dict[pname] = param[1].data.detach().cpu().numpy()

    for param in layer.named_parameters():
        pname = param[0]
        if "cross_attn" in pname or "q_proj" in pname or "k_proj" in pname or "v_proj" in pname:
            param_np = name_param_dict[pname]
        else:
            param_np = module.state_dict()[pname].detach().cpu().numpy()

        if pname.endswith("weight"):
            param_np = numpy.transpose(param_np)
        paddle.assign(paddle.to_tensor(param_np), param[1])


@init_pool.register("paddle", "MultiHeadAttention", "torch", "MultiheadAttention")
def init_MultiHeadAttention_(layer, module):
    name_param_dict = {}
    for i, param in enumerate(layer.named_parameters()):
        pname = param[0]
        if "cross_attn" in pname:
            pname = pname.replace("cross_attn", "multihead_attn")
        elif "q" not in pname and "k" not in pname and "v" not in pname:
            continue
        param_np = param[1].numpy()
        pname = pname.replace("q_proj.", "in_proj_")
        pname = pname.replace("k_proj.", "in_proj_")
        pname = pname.replace("v_proj.", "in_proj_")
        if pname not in name_param_dict:
            name_param_dict[pname] = param_np
        elif "_weight" in pname:
            name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=1)
        else:
            name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=0)

    for i, param in enumerate(module.named_parameters()):
        pname, pa = param[0], param[1]
        if "in_proj" in pname or "multihead_attn" in pname:
            param_np = name_param_dict[pname]
        else:
            param_np = layer.state_dict()[pname].numpy()
        if pname.endswith("weight"):
            param_np = numpy.transpose(param_np)
        device = param[1].device
        param[1].data = torch.from_numpy(param_np).to(device)
