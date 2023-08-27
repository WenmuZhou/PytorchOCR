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

import contextlib

from ..utils import Counter, for_each_grad_tensor, for_each_tensor
from .module_struct import LayerStack


class Report:
    def __init__(self, marker):
        self.items = []
        self.counter = Counter()
        self.marker = marker
        self.stack = LayerStack()

    def put_item(self, type_, input_, output, net, net_id):
        step = self.counter.get_id()
        self.items.append(
            ReportItem(
                type_=type_,
                step=step,  # report order of layers/apis
                input_=input_,
                output=output,
                net=net,
                net_id=net_id,  # traversal order of sublayers
            )
        )
        return self.items[-1]

    def __str__(self):
        sorted(self.items, key=lambda x: x.step)
        strings = []
        strings.append("Report:")
        for item in self.items:
            strings.append("    " + str(item.step) + ": [{}]".format(item.net_str))
        return "\n".join(strings)


class ReportItem:
    def __init__(self, type_, step, input_, output, net, net_id):
        assert type_ in [
            "forward",
            "backward",
        ], f"type can only be one of ['forward', 'backward'], but{type_}"
        self.type = type_  # fwd or bwd
        self.step = step  # report order
        self.input = input_  # layer input (same for fwd or bwd)
        self.output = output  # layer output (same for fwd or bwd)

        self.net = net  # the layer ,if it is an api, this should be a str layer which is generated in hooks
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.net_id = net_id  # sublayer order
        self.fwd_item = None  # bound to another reportitem, if self.type is "backward"
        self.bwd_item = None  # bound to another reportitem, if self.type is "forward"
        self.input_grads = self._gen_input_grads()

    def set_forward(self, fwd):
        assert self.type == "backward", "can't set forward for non-backward item."
        fwd.bwd_item = self
        self.fwd_item = fwd

    def _gen_input_grads(self):
        if self.type == "forward":
            return None
        assert self.input is not None, "Backward while input is None, not expected."

        return [None for i in for_each_grad_tensor(self.input)]

    def set_input_grads(self, nth, value):
        assert nth < len(self.input_grads)
        self.input_grads[nth] = value

    def tensors_for_compare(self):
        if self.type == "forward":
            return [t for (t,) in for_each_tensor(self.output)]
        if self.type == "backward":
            return [t for (t,) in for_each_grad_tensor(self.input_grads)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        strings.append("ReportItem: \n    type={}".format(self.type))
        strings.append("    step_idx: {}".format(self.step))
        strings.append("    net: {}\n".format(self.net_str))
        return "\n".join(strings)


global_report = None


@contextlib.contextmanager
def report_guard(report):
    global global_report
    old_report = global_report
    try:
        global_report = report
        yield
    finally:
        global_report = old_report


def current_report():
    return global_report
