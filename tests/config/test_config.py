#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 
# Copyright 2016 The Python Authors All Rights Reserved.
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

# @Time    : 2020/5/20 11:16 上午
# @Author  : peichao.xu
# @Email   : 563853580@qq.com
# @File    : test_config.py

# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from torchocr.config import get_cfg, get_config_file
from torchocr.config import CfgNode as CN


class TestConfig(unittest.TestCase):

    def test_setup_config(self):
        """
            初始化配置用例，不同yaml文件对应不同实验
        """
        opts = ['MODEL.DEVICE', 'cpu']
        cfg = get_cfg()
        cfg.merge_from_file('east_R_50_FPN_1x.yaml')
        cfg.merge_from_list(opts)
        cfg.freeze()
        self.assertEqual(cfg['MODEL']['DEVICE'], 'cpu')
        # return cfg

    def test_get_config_file(self):
        '''
            从基础配置文件中加载配置
        '''
        cfg = get_config_file('Base-EAST.yaml')
        # return cfg

    def test_add_config(self):
        """
            在default.py的基础上，增加字段
        """
        cfg = get_cfg()
        _C = cfg
        _C.MODEL.EXAMPLE = CN()
        _C.MODEL.EXAMPLE.NUM_CLASSES = 3
        cfg.freeze()
        self.assertEqual(cfg['MODEL']['EXAMPLE']['NUM_CLASSES'], 3)
        # return cfg

    def test_custom_config(self):
        '''
            完全自定义配置，舍弃default.py中的字段
        '''
        cfg = CN()
        _C = cfg
        _C.MODEL = CN()
        _C.MODEL.DEVICE = "cuda"
        cfg.freeze()
        self.assertEqual(cfg, {
            'MODEL': {
                "DEVICE": "cuda"
            }
        })
        # return cfg

    def test_set_global_config(self):
        """
        Let the global config point to the given cfg.

        Assume that the given "cfg" has the key "KEY", after calling
        `set_global_cfg(cfg)`, the key can be accessed by:

        .. code-block:: python

            from torchpack.config import global_cfg
            print(global_cfg.KEY)

        By using a hacky global config, you can access these configs anywhere,
        without having to pass the config object or the values deep into the code.
        This is a hacky feature introduced for quick prototyping / research exploration.
        """


if __name__ == '__main__':
    unittest.main()
