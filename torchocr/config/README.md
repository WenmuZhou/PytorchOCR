
# Usage:

## 方式一：简易模式

```python
'''
file1.py
'''
from torchocr.config import CfgNode as CN

def setup_config()
    cfg = CN()
    _C = cfg
    _C.MODEL = CN()
    _C.MODEL.DEVICE = "cuda"
    cfg.freeze()
    set_global_cfg(cfg)
    return cfg
setup_config()

'''
file2.py
'''
from torchocr.config import global_cfg
print(global_cfg.KEY)
```

## 方式二：通过参数传递

```python
form torchocr.config import get_cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    cfg = setup()
    model = build_model(cfg)
```

## 方式三：通过全局定义

```python

'''
file1.py
'''
form torchocr.config import set_global_cfg
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()
set_global_cfg(cfg)

'''
file2.py
'''
from torchocr.config import global_cfg
print(global_cfg.KEY)
```

