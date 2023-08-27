__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        support_dict = [
        ]
    elif model_type == "rec" or model_type == "cls":
        from .rec_resnet_31 import ResNet31
        from .rec_mv1_enhance import MobileNetV1Enhance
        support_dict = [
            'MobileNetV1Enhance','ResNet31'
        ]
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class
