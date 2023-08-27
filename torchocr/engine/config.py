import os
from collections.abc import Mapping
import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter

__all__ = ['Config']


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
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


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def _merge_dict(config, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config

    Returns: dct
    """
    for key, value in merge_dct.items():
        sub_keys = key.split('.')
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {'.'.join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], dict) and isinstance(
                value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def print_dict(cfg, print_func=print, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, print_func, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        else:
            print_func("{}{} : {}".format(delimiter * " ", k, v))


class Config(object):
    def __init__(self, config_path, BASE_KEY='_BASE_'):
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)

    def _load_config_with_base(self, file_path):
        """
               Load config from file.

               Args:
                   file_path (str): Path of the config file to be loaded.

               Returns: global config
               """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        if self.BASE_KEY in file_cfg:
            all_base_cfg = AttrDict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                if base_yml.startswith("~"):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith('/'):
                    base_yml = os.path.join(
                        os.path.dirname(file_path), base_yml)

                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[
            0]
        return file_cfg

    def merge_dict(self, args):
        self.cfg = _merge_dict(self.cfg, args)

    def print_cfg(self, print_func=print):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        print_func('----------- Config -----------')
        print_dict(self.cfg, print_func)
        print_func('---------------------------------------------')

    def save(self, p, cfg=None):
        if cfg is None:
            cfg = self.cfg
        with open(p, 'w') as f:
            yaml.dump(
                dict(cfg), f, default_flow_style=False, sort_keys=False)