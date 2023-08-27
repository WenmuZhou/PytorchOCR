# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 15:23
# @Author  : zhoujun
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.utility import ArgsParser
from torchocr import Config
from torchocr import Trainer


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    trainer = Trainer(cfg, mode='eval')
    metric = trainer.eval()
    trainer.logger.info('metric eval ***************')
    for k, v in metric.items():
        trainer.logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    main()
