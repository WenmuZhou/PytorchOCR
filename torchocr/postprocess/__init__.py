
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .east_postprocess import EASTPostProcess
from .sast_postprocess import SASTPostProcess
from .fce_postprocess import FCEPostProcess
from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, SRNLabelDecode, \
    DistillationCTCLabelDecode, NRTRLabelDecode, SARLabelDecode, \
    SEEDLabelDecode, PRENLabelDecode, ViTSTRLabelDecode, ABINetLabelDecode, \
    SPINLabelDecode, VLLabelDecode, RFLLabelDecode, SATRNLabelDecode
from .cls_postprocess import ClsPostProcess
from .pg_postprocess import PGPostProcess
from .table_postprocess import TableMasterLabelDecode, TableLabelDecode
from .picodet_postprocess import PicoDetPostProcess
from .ct_postprocess import CTPostProcess
from .drrg_postprocess import DRRGPostprocess
from .rec_postprocess import CANLabelDecode


def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'FCEPostProcess',
        'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode',
        'PGPostProcess', 'DistillationCTCLabelDecode', 'TableLabelDecode',
        'DistillationDBPostProcess', 'NRTRLabelDecode', 'SARLabelDecode',
        'SEEDLabelDecode', 'VQASerTokenLayoutLMPostProcess',
        'VQAReTokenLayoutLMPostProcess', 'PRENLabelDecode',
        'DistillationSARLabelDecode', 'ViTSTRLabelDecode', 'ABINetLabelDecode',
        'TableMasterLabelDecode', 'SPINLabelDecode',
        'DistillationSerPostProcess', 'DistillationRePostProcess',
        'VLLabelDecode', 'PicoDetPostProcess', 'CTPostProcess',
        'RFLLabelDecode', 'DRRGPostprocess', 'CANLabelDecode',
        'SATRNLabelDecode'
    ]

    if config['name'] == 'PSEPostProcess':
        from .pse_postprocess import PSEPostProcess
        support_dict.append('PSEPostProcess')

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
