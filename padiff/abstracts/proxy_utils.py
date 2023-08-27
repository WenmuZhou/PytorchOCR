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


def init_route(model):
    def _set_route(model, path):
        for name, child in model.named_children():
            path.append(name)
            if not hasattr(child.model, "route"):
                setattr(child.model, "route", ".".join(path))
            _set_route(child, path)
            path.pop()

    if not hasattr(model, "route"):
        setattr(model.model, "route", model.name)
        _set_route(model, [model.name])


def remove_inplace(model):
    """
    Set `inplace` tag to `False` for torch module
    """
    for submodel in model.submodels():
        if hasattr(submodel, "inplace"):
            submodel.inplace = False


def deco_iter(iterator, fn):
    def new_fn(obj):
        try:
            return fn(obj)
        except:
            return obj

    def new_generator():
        for obj in iterator:
            if isinstance(obj, (tuple, list)):
                yield tuple(map(new_fn, obj))
            else:
                yield new_fn(obj)

    return new_generator()
