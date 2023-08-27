__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .rnn import SequenceEncoder
    support_dict = [
        'SequenceEncoder', 'DBFPN', 'RSEFPN', 'LKPAN'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))

    module_class = eval(module_name)(**config)
    return module_class
