from torch import nn
from .base_model import BaseModel
from torchocr.utils.ckpt import load_pretrained_params

__all__ = ['DistillationModel']


class DistillationModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        self.model_list = nn.ModuleDict()
        for key in config["Models"]:
            model_config = config["Models"][key]
            freeze_params = False
            pretrained = None
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            model = BaseModel(model_config)
            if pretrained is not None:
                load_pretrained_params(model, pretrained)
            if freeze_params:
                for param in model.parameters():
                    param.requires_grad = False
            self.model_list.add_module(key, model)

    def forward(self, x, data=None):
        result_dict = dict()
        for model_name in self.model_list:
            result_dict[model_name] = self.model_list[model_name](x, data)
        return result_dict
