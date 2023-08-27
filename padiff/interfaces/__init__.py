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

# this folder is used for support user interfaces

from .interfaces import create_model, assign_weight, auto_diff, check_dataloader
from ..checker import check_report, check_params, check_grads, check_weights
from ..dump_tools import set_dump_root_path, get_dump_root_path
from ..weight_init.special_init import add_special_init
