__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        from .det_mobilenet_v3 import MobileNetV3
        from .det_resnet_vd import ResNet_vd
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import PPHGNet_small
        support_dict = [
            'MobileNetV3', 'ResNet_vd', 'PPLCNetV3', 'PPHGNet_small'
        ]
    elif model_type == "rec" or model_type == "cls":
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet
        from .rec_resnet_31 import ResNet31
        from .rec_nrtr_mtb import MTB
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import PPHGNet_small
        support_dict = [
            'MobileNetV1Enhance', 'ResNet31', 'MobileNetV3', 'PPLCNetV3', 'PPHGNet_small', 'ResNet', 'MTB'
        ]
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class
