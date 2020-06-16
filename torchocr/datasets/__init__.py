import copy
from addict import Dict
from torch.utils.data import DataLoader

from .RecDataSet import RecDataLoader, RecTextLineDataset

__all__ = ['build_dataloader']

support_dataset = ['RecTextLineDataset', 'DetTextLineDataset']
support_loader = ['RecDataLoader', 'DataLoader']


def build_dataset(config):
    dataset_type = config.pop('type')
    assert dataset_type in support_dataset, f'{dataset_type} is not developed yet!, only {support_dataset} are support now'
    dataset_class = eval(dataset_type)(config)
    return dataset_class


def build_loader(dataset, config):
    copy_config = copy.deepcopy(config)
    dataset_type = copy_config.pop('type')
    assert dataset_type in support_loader, f'{dataset_type} is not developed yet!, only {support_loader} are support now'
    dataset_class = eval(dataset_type)(dataset=dataset, **copy_config)
    return dataset_class

def build_dataloader(config):
    """
    数据加载
    :return:
    """
    # Rec开头的dataset只能配合Rec开头的dataloader使用
    # build dataset
    config = Dict(config)
    dataset = build_dataset(config.dataset)
    # build loader
    loader = build_loader(dataset, config.loader)
    return loader
