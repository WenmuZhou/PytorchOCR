from addict import Dict

config = Dict()
config.exp_name = 'DBNet_icdar_distill'
config.train_options = {
    # for train
    'resume_from': '',  # 继续训练地址
    'checkpoint_save_dir': f"./output/{config.exp_name}/checkpoint",  # 模型保存地址，log文件也保存在这里
    'device': 'cuda:0',  # 不建议修改
    'epochs': 600,
    'fine_tune_stage': ['backbone', 'neck', 'head'],
    'print_interval': 5,  # step为单位
    'val_interval': 1,  # epoch为单位
    'ckpt_save_type': 'HighestAcc',  # HighestAcc：只保存最高准确率模型 ；FixedEpochStep：每隔ckpt_save_epoch个epoch保存一个
    'ckpt_save_epoch': 4,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

config.SEED = 927
config.optimizer = {
    'type': 'Adam',
    'lr': 0.0002,
    'weight_decay': 0,
}

config.model = {
    'type': 'DistillationModel',
    'algorithm': 'Distillation',
    'init_weight': False,  # 当不使用任何预训练模型（子网络或任意子网络backbone）时打开
    'models': {
        'Teacher': {
            'type': "DetModel",
            'freeze_params': True,
            'backbone': {"type": "ResNet", 'pretrained': False, 'layers': 18},
            'neck': {"type": 'DB_fpn', 'out_channels': 256},
            'head': {"type": "DBHead"},
            'in_channels': 3,
            'pretrained': '/path/to/your/workspace/work/PytorchOCR/models/dismodels/DBNet_icdar_res18_fast_pre.pth'
        },
        'Student': {
            'type': "DetModel",
            'freeze_params': False,
            'backbone': {"type": "MobileNetV3", 'pretrained': False, 'disable_se': False},
            'neck': {"type": 'DB_fpn', 'out_channels': 96},
            'head': {"type": "DBHead"},
            'in_channels': 3,
            'pretrained': '/path/to/your/workspace/work/PytorchOCR/models/dismodels/mbv3.pth'
        },
        'Student2': {
            'type': "DetModel",
            'freeze_params': False,
            'backbone': {"type": "MobileNetV3", 'pretrained': False, 'disable_se': False},
            'neck': {"type": 'DB_fpn', 'out_channels': 96},
            'head': {"type": "DBHead"},
            'in_channels': 3,
            'pretrained': '/path/to/your/workspace/work/PytorchOCR/models/dismodels/mbv3.pth'
        }

    }

}

config.loss = {
    'type': 'CombinedLoss',
    'combine_list': {
        'DistillationDilaDBLoss': {
            'weight': 2.0,
            'model_name_pairs': [("Student", "Teacher"), ("Student2", "Teacher")],
            # 'model_name_pairs': [("Student", "Teacher")],
            'key': 'maps',
            'balance_loss': True,
            'main_loss_type': 'DiceLoss',
            'alpha': 5,
            'beta': 10,
            'ohem_ratio': 3,
        },

        'DistillationDMLLoss': {
            'maps_name': "thrink_maps",
            'weight': 1.0,
            'model_name_pairs': ["Student", "Student2"],
            'key': 'maps'
        },

        'DistillationDBLoss': {
            'weight': 1.0,
            'model_name_list': ["Student"],
            'balance_loss': True,
            'main_loss_type': 'DiceLoss',
            'alpha': 5,
            'beta': 10,
            'ohem_ratio': 3}

    }

}

config.post_process = {
    'type': 'DistillationDBPostProcess',
    'model_name': ["Student", "Student2", "Teacher"],
    # 'model_name': ["Student", "Teacher"],
    'thresh': 0.3,  # 二值化输出map的阈值
    'box_thresh': 0.5,  # 低于此阈值的box丢弃
    'unclip_ratio': 1.5  # 扩大框的比例

}

config.metric = {
    'name': 'DistillationMetric',
    'base_metric_name': 'DetMetric',
    'main_indicator': 'hmean',
    'key': "Student"
}

# for dataset
# ##lable文件
### 存在问题，gt中str-->label 是放在loss中还是放在dataloader中
config.dataset = {
    'train': {
        'dataset': {
            'type': 'JsonDataset',
            'file': r'/path/to/your/workspace/dataset/icdar15-detection/train.json',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            # db 预处理，不需要修改
            'pre_processes': [{'type': 'IaaAugment', 'args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                              {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                                                              {'type': 'Resize', 'args': {'size': [0.5, 3]}}]},
                              {'type': 'EastRandomCropData', 'args': {'size': [640, 640], 'max_tries': 50, 'keep_ratio': True}},
                              {'type': 'MakeBorderMap', 'args': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}},
                              {'type': 'MakeShrinkMap', 'args': {'shrink_ratio': 0.4, 'min_text_size': 8}}],
            'filter_keys': ['img_name', 'text_polys', 'texts', 'ignore_tags', 'shape'],
            # 需要从data_dict里过滤掉的key
            'ignore_tags': ['*', '###', ' '],
            'img_mode': 'RGB'
        },
        'loader': {
            'type': 'DataLoader',  # 使用torch dataloader只需要改为 DataLoader
            'batch_size': 20,
            'shuffle': True,
            'num_workers': 20,
            'collate_fn': {
                'type': ''
            }
        }
    },
    'eval': {
        'dataset': {
            'type': 'JsonDataset',
            'file': r'/path/to/your/workspace/dataset/icdar15-detection/test.json',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pre_processes': [{'type': 'ResizeShortSize', 'args': {'short_size': 736, 'resize_text_polys': False}}],
            'filter_keys': [],  # 需要从data_dict里过滤掉的key
            'ignore_tags': ['*', '###', ' '],
            'img_mode': 'RGB'
        },
        'loader': {
            'type': 'DataLoader',
            'batch_size': 1,  # 必须为1
            'shuffle': False,
            'num_workers': 10,
            'collate_fn': {
                'type': 'DetCollectFN'
            }
        }
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)
