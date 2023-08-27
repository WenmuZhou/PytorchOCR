# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 15:37
# @Author  : zhoujun
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config

def update_rec_head_out_channels(cfg, post_process_class):
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if cfg['Architecture']["algorithm"] in ["Distillation",
                                                     ]:  # distillation model
            for key in cfg['Architecture']["Models"]:
                if cfg['Architecture']['Models'][key]['Head']['name'] == 'MultiHead':  # for multi head
                    if cfg['PostProcess']['name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    if cfg['PostProcess']['name'] == 'DistillationNRTRLabelDecode':
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    # update SARLoss params
                    if list(cfg['Loss']['loss_config_list'][-1].keys())[0] == 'DistillationSARLoss':
                        cfg['Loss']['loss_config_list'][-1][
                            'DistillationSARLoss'][
                            'ignore_index'] = char_num + 1
                        out_channels_list['SARLabelDecode'] = char_num + 2
                    elif list(cfg['Loss']['loss_config_list'][-1].keys())[0] == 'DistillationNRTRLoss':
                        out_channels_list['NRTRLabelDecode'] = char_num + 3

                    cfg['Architecture']['Models'][key]['Head']['out_channels_list'] = out_channels_list
                else:
                    cfg['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif cfg['Architecture']['Head']['name'] == 'MultiHead':  # for multi head
            if cfg['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            if cfg['PostProcess']['name'] == 'NRTRLabelDecode':
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list['CTCLabelDecode'] = char_num
            # update SARLoss params
            if list(cfg['Loss']['loss_config_list'][1].keys())[0] == 'SARLoss':
                if cfg['Loss']['loss_config_list'][1]['SARLoss'] is None:
                    cfg['Loss']['loss_config_list'][1]['SARLoss'] = {
                        'ignore_index': char_num + 1
                    }
                else:
                    cfg['Loss']['loss_config_list'][1]['SARLoss'][
                        'ignore_index'] = char_num + 1
                out_channels_list['SARLabelDecode'] = char_num + 2
            elif list(cfg['Loss']['loss_config_list'][1].keys())[0] == 'NRTRLoss':
                out_channels_list['NRTRLabelDecode'] = char_num + 3
            cfg['Architecture']['Head']['out_channels_list'] = out_channels_list
        else:  # base rec model
            cfg['Architecture']["Head"]['out_channels'] = char_num

        if cfg['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
            cfg['Loss']['ignore_index'] = char_num - 1
