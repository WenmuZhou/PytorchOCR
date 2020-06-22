import copy
from addict import Dict
from torch.utils.data import DataLoader

from .RecDataSet import RecDataLoader, RecTextLineDataset, RecLmdbDataset
from .DetDataSet import JsonDataset
from .RecCollateFn import RecCollateFn
from .DetCollateFN import DetCollectFN

__all__ = ['build_dataloader']

support_dataset = ['RecTextLineDataset', 'RecLmdbDataset', 'DetTextLineDataset','JsonDataset']
support_loader = ['RecDataLoader', 'DataLoader']


def build_dataset(config):
    """
    根据配置构造dataset

    :param config: 数据集相关的配置，一般为 config['dataset']['train']['dataset] or config['dataset']['eval']['dataset]
    :return: 根据配置构造好的 DataSet 类对象
    """
    dataset_type = config.pop('type')
    assert dataset_type in support_dataset, f'{dataset_type} is not developed yet!, only {support_dataset} are support now'
    dataset_class = eval(dataset_type)(config)
    return dataset_class


def build_loader(dataset, config):
    """
    根据配置构造 dataloader, 包含两个步骤，1. 构造 collate_fn, 2. 构造 dataloader

    :param dataset: 继承自 torch.utils.data.DataSet的类对象
    :param config: loader 相关的配置，一般为 config['dataset']['train']['loader] or config['dataset']['eval']['loader]
    :return: 根据配置构造好的 DataSet 类对象
    """
    dataloader_type = config.pop('type')
    assert dataloader_type in support_loader, f'{dataloader_type} is not developed yet!, only {support_loader} are support now'

    # build collate_fn
    if 'collate_fn' in config:
        config['collate_fn']['dataset'] = dataset
        collate_fn = build_collate_fn(config.pop('collate_fn'))
    else:
        collate_fn = None
    dataloader_class = eval(dataloader_type)(dataset=dataset, collate_fn=collate_fn, **config)
    return dataloader_class


def build_collate_fn(config):
    """
    根据配置构造 collate_fn

    :param config: collate_fn 相关的配置
    :return: 根据配置构造好的 collate_fn 类对象
    """
    collate_fn_type = config.pop('type')
    if len(collate_fn_type) == 0:
        return None
    collate_fn_class = eval(collate_fn_type)(**config)
    return collate_fn_class


def build_dataloader(config):
    """
    根据配置构造 dataloader, 包含两个步骤，1. 构造 dataset, 2. 构造 dataloader
    :param config: 数据集相关的配置，一般为 config['dataset']['train'] or config['dataset']['eval']
    :return: 根据配置构造好的 DataLoader 类对象
    """
    # build dataset
    copy_config = copy.deepcopy(config)
    copy_config = Dict(copy_config)
    dataset = build_dataset(copy_config.dataset)

    # build loader
    loader = build_loader(dataset, copy_config.loader)
    return loader
