
__all__ = ['build_head']


def build_head(config):
    from .rec_multi_head import MultiHead
    from .rec_sar_head import SARHead

    support_dict = [
        'MultiHead', 'SARHead'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
