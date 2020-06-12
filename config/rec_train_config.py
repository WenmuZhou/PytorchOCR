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
rec_train_options = {
    # for train
    'resume_from': '/home/novio/project/PytorchOCR/rec_r34_vd_none_bilstm_ctc/best_accuracy.pdparams',
    'third_party_name': 'paddle',
    'checkpoint_save_dir': "./out_dir/checkpoint",
    'device': 'cuda:0',
    'base_lr': 0.01,
    'batch_size': 2,
    'epochs': 200,
    'weight_decay': 1e-4,
    'fine_tune_stage': ['backbone', 'neck', 'head'],
    # 'optimizer': ['SGD', 'Adam', 'Radam', 'ASGD'],
    # 'optimizer_step': [140, 160, 180],
    'optimizer': ['Adam'],
    'optimizer_step': [],
    # 'lr_scheduler': 'LambdaLR',
    'lr_scheduler': 'StepLR',
    # 'lr_scheduler_info': {'burn_in': 1, 'steps': [50, 100]},
    'lr_scheduler_info': {'step_size':50,'gamma':0.5},
    'print_interval': 200,  # step为单位
    'val_interval': 1000,  # step为单位
    'ckpt_save_type': 'HighestAcc',  # 'FixedEpochStep'
    'ckpt_save_epoch': 4,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

SEED = 927


# if autoscale_lr:
#     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
#     optimizer['lr'] = optimizer['lr'] * len(gpu_ids) / 8
# autoscale_lr = True


# for model
# 建议都为dict

model = {
    'type': "RecModel",
    'neck': {"type": 'PPaddleRNN'},
    'backbone': {"type": "ResNet", 'layers': 34},
    'head': {"type": "CTC", 'n_class': 91},
    'in_channels': 3,
}
# class ArcConfig:
#     def __init__(self):
#         self.neck = {"type": 'PPaddleRNN'}
#         self.backbone = {"type": "ResNet", 'layers': 34}
#         self.head = {"type": "CTC", 'n_class': 27}
#         self.in_channels = 3,
#         self.labels = 1000


# arc_config = ArcConfig()
#
# architecture = 'CRNNRes34'
# architecture_config = {
#     'in_channels': 3,
#     'labels': 1000
# }

loss = {
    'type': 'CTCLoss',
    'blank_idx': 0,
}
# for dataset
# ##lable文件
### 存在问题，gt中str-->label 是放在loss中还是放在dataloader中
dataset = {
    'type': 'icdar15.ICDAR15RecDataset',
    'train': {
        'data_dir': '/data/OCR/ICDAR2015/converted_data',
        'input_h': 32,
        'mean': 0.588,
        'std': 0.193,
        'mode': 'train',
        'augmentation': False,
        'alphabet': '../datasets/alphabets/enAlphaNumPunc90.txt',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 1,
    },
    'eval': {
        'data_dir': '/data/OCR/ICDAR2015/converted_data',
        'input_h': 32,
        'mean': 0.588,
        'std': 0.193,
        'mode': 'eval',
        'augmentation': False,
        'alphabet': '../datasets/alphabets/enAlphaNumPunc90.txt',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 1,
    }

}
