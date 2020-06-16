import copy
from addict import Dict
from torch.utils.data import DataLoader

from .collate_fn import RecCollateFn
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
    dataloader_type = config.pop('type')
    assert dataloader_type in support_loader, f'{dataloader_type} is not developed yet!, only {support_loader} are support now'

    # build collate_fn
    if 'collate_fn' in config:
        collate_fn = build_collate_fn(config.pop('collate_fn'))
    else:
        collate_fn = None
    dataloader_class = eval(dataloader_type)(dataset=dataset, collate_fn=collate_fn, **config)
    return dataloader_class


def build_collate_fn(config):
    collate_fn_type = config.pop('type')
    if len(collate_fn_type) == 0:
        return None
    collate_fn_class = eval(collate_fn_type)(**config)
    return collate_fn_class


def build_dataloader(config):
    """
    数据加载
    :return:
    """
    # Rec开头的dataset只能配合Rec开头的dataloader使用
    # build dataset
    copy_config = copy.deepcopy(config)
    copy_config = Dict(copy_config)
    dataset = build_dataset(copy_config.dataset)

    # build loader
    loader = build_loader(dataset, copy_config.loader)
    return loader
