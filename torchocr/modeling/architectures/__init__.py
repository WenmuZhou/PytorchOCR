import copy
import importlib

from .base_model import BaseModel
from .distillation_model import DistillationModel

__all__ = ["build_model", "DistillationModel"]


def build_model(config):
    config = copy.deepcopy(config)
    if not "name" in config:
        arch = BaseModel(config)
    else:
        name = config.pop("name")
        mod = importlib.import_module(__name__)
        arch = getattr(mod, name)(config)
    return arch