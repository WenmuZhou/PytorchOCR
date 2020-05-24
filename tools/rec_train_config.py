# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

# 日志
from utils import Logger
logger = Logger()

# for train
resume_from = None
ckpt_dir = ""
device = 'cuda:0'

# ####################rec_train_options 参数说明##########################
# 识别训练参数
# base_lr：初始学习率
# fine_tune_stage:
#     if you want to freeze some stage, and tune the others.
#     ['backbone', 'neck', 'head'], 所有参数都参与调优
#     ['backbone'], 只调优backbone部分的参数
#     后续更新： 1、添加bn层freeze的代码
# optimizer 和 optimizer_step:
#     优化器的配置， 成对
#     example1： 'optimizer'：['SGD'], 'optimizer_step':[],表示一直用SGD优化器
#     example2:  'optimizer':['SGD', 'Adam'], 'optimizer_step':[160]， 表示前[0,160)个epoch使用SGD优化器，
#                [160,~]采用Adam优化器
# lr_scheduler和lr_scheduler_info：
#     学习率scheduler的设置
# ckpt_save_type作用是选择模型保存的方式
#      HighestAcc：只保存在验证集上精度最高的模型（还是在训练集上loss最小）
#      FixedEpochStep： 按一定间隔保存模型
###
rec_train_options = {
    'base_lr': 0.01,
    'batch_size': 2,
    'epochs': 200,
    'weight_decay': 1e-4,
    'fine_tune_stage': ['backbone', 'neck', 'head'],
    # 'optimizer': ['SGD', 'Adam', 'Radam', 'ASGD'],
    # 'optimizer_step': [140, 160, 180],
    'optimizer': ['Adam'],
    'optimizer_step': [],
    'lr_scheduler': 'ReduceLROnPlateau',
    'lr_scheduler_info': [],
    'print_interval': 200, # step为单位
    'val_interval': 1000, # step为单位
    'ckpt_save_type': 'HighestAcc', # 'FixedEpochStep'
    'ckpt_save_epoch': 4,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

SEED = 927


# if autoscale_lr:
#     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
#     optimizer['lr'] = optimizer['lr'] * len(gpu_ids) / 8
# autoscale_lr = True


# for model

class ArcConfig:
    def __init__(self):
        self.neck = {"type": 'PPaddleRNN'}
        self.backbone = {"type": "ResNet", 'layers': 34}
        self.head = {"type": "CTC", 'n_class': 27}
        self.in_channels = 3,
        self.labels = 1000


arc_config = ArcConfig()
#
# architecture = 'CRNNRes34'
# architecture_config = {
#     'in_channels': 3,
#     'labels': 1000
# }


# for dataset
# ##lable文件
label_file = ''

