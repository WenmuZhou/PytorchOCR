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

import os


class LayerStack(object):
    def __init__(self):
        super(LayerStack, self).__init__()
        self.stack = []
        self.root = []

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        return self.stack.pop()

    def _top(self):
        if len(self.stack) == 0:
            return None
        return self.stack[-1]

    def _empty(self):
        return len(self.stack) == 0

    def push_layer(self, module):
        net = NetWrapper(module)
        if not self._empty():
            net.father = self._top()
            self._top().children.append(net)
        else:
            self.root.append(net)
        self._push(net)

    def pop_layer(self, module):
        assert id(self._top().net) == id(module)
        return self._pop()

    def push_api(self, api, fwd, bwd):
        if hasattr(api, "__api__"):
            net = NetWrapper(api)
            net.layer_type = "api"
            if not self._empty():
                self._top().children.append(net)
                net.father = self._top()
            net.set_report(fwd, bwd)
        else:
            net = self._top()
            net.set_report(fwd, bwd)


class NetWrapper(object):
    def __init__(self, net):
        self.net = net
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.children = []
        self.father = None

        self.layer_type = "net"  # "api" | "in map" | "net"

        self.fwd_report = None
        self.bwd_report = None

    def set_report(self, fwd, bwd):
        self.fwd_report = fwd
        self.bwd_report = bwd

    def pprint(self):
        ret = _tree_print(self)
        print("\n".join(ret))

    def __str__(self):
        return f"({self.layer_type}){self.net_str}"


def _tree_print(root, mark=None, prefix=[]):
    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += str(root)
    if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(root.net, "route"):
        cur_str += "  (" + root.net.route + ")"
    if mark is root:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]
    for i, child in enumerate(root.children):
        pre = " |--- "
        if i == len(root.children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = _tree_print(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs
