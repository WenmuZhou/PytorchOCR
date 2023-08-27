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


# NOTICE: make sure params is in the same device after init


def build_name(base_framework, base_model_name, raw_framework, raw_model_name):
    name = base_framework + "::" + base_model_name + "###" + raw_framework + "::" + raw_model_name
    return name


class SpecialInitPool(object):
    def __init__(self):
        self.funcs = {}
        # used for LayerMap.auto
        self.registered_raw_models = set()
        self.registered_base_models = set()

    def register(self, base_framework, base_model_name, raw_framework, raw_model_name):
        name = build_name(base_framework, base_model_name, raw_framework, raw_model_name)
        self.registered_raw_models.add(raw_framework + "::" + raw_model_name)
        self.registered_base_models.add(base_framework + "::" + base_model_name)

        def do_reg(func):
            self.funcs[name] = func
            return func

        return do_reg


global_special_init_pool = SpecialInitPool()


def add_special_init(base_framework, base_model_name, raw_framework, raw_model_name, func):
    name = build_name(base_framework, base_model_name, raw_framework, raw_model_name)
    global_special_init_pool.registered_raw_models.add(raw_framework + "::" + raw_model_name)
    global_special_init_pool.registered_base_models.add(base_framework + "::" + base_model_name)
    global_special_init_pool.funcs[name] = func
