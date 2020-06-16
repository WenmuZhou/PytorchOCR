# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

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
from addict import Dict

config = Dict()
config.exp_name = 'CRNN'
config.train_options = {
    # for train
    'resume_from': '',
    'third_party_name': 'paddle',
    'checkpoint_save_dir': "./out_dir/checkpoint",
    'device': 'cuda:0',
    'batch_size': 2,
    'epochs': 200,
    'fine_tune_stage': ['backbone', 'neck', 'head'],
    'print_interval': 1,  # step为单位
    'val_interval': 10,  # step为单位
    'ckpt_save_type': 'HighestAcc',  # 'FixedEpochStep'
    'ckpt_save_epoch': 4,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

config.SEED = 927
config.optimizer = [{
    'type': 'Adam',
    'lr': 0.01,
    'weight_decay': 1e-4,
}]

config.lr_scheduler = {
    'type': 'StepLR',
    'step_size': 50,
    'game': 0.5
}
config.model = {
    'type': "RecModel",
    'neck': {"type": 'PPaddleRNN'},
    'backbone': {"type": "ResNet", 'layers': 34},
    'head': {"type": "CTC", 'n_class': 92},
    'in_channels': 3,
}

config.loss = {
    'type': 'CTCLoss',
    'blank_idx': 0,
}

config.metric = {}
config.post_p = {}

# for dataset
# ##lable文件
### 存在问题，gt中str-->label 是放在loss中还是放在dataloader中
config.dataset = {
    'alphabet': r'D:\code\OCR\PytorchOCR\torchocr\datasets\alphabets\enAlphaNumPunc90.txt',
    'train': {
        'dataset': {
            'type': 'RecTextLineDataset',
            'data_dir': r'D:\dataset\converted_icdar2015_rec_data\converted_data',
            'input_h': 32,
            'mean': 0.588,
            'std': 0.193,
            'mode': 'train',
            'augmentation': False,
        },
        'loader': {
                'type': 'RecDataLoader',  # 使用torch dataloader只需要改为 DataLoader
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 1,
        }
    },
    'eval': {
        'dataset': {
            'type': 'RecTextLineDataset',
            'data_dir': r'D:\dataset\converted_icdar2015_rec_data\converted_data',
            'input_h': 32,
            'mean': 0.588,
            'std': 0.193,
            'mode': 'eval',
            'augmentation': False,
        },
        'loader': {
            'type': 'RecDataLoader',
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 1,
        }
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)
