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

from .special_init_pool import global_special_init_pool, add_special_init, build_name

import os
import importlib

cur_dir = os.path.split(os.path.realpath(__file__))[0]
for filename in os.listdir(cur_dir):
    if filename.startswith("init_") and filename.endswith(".py"):
        module_name = filename.rpartition(".")[0]
        importlib.import_module(__name__ + "." + module_name)

__all__ = [
    "global_special_init_pool",
    "add_special_init",
    "build_name",
]
