import numpy as np
import torch
from torchocr.networks import build_model


def pytorch2onnx(model,
                 input_shape,
                 output_file='a.onnx'):
    model.eval()

    one_img = torch.randn(input_shape)

    input_names = ["input_1"]
    output_names = ["output_1"]
    dynamic_axes = {'input_1': {0 : "b", 2 : "w", 3 : "h"}, 'output_1': {0 : "b", 2 : "w", 3 : "h"}}

    # register_extra_symbolics(11)
    torch.onnx.export(
        model,
        one_img,
        output_file,
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        opset_version=11,
        dynamic_axes=dynamic_axes
    )
    norm_out = np.linalg.norm(one_img)
    print(f'Successfully exported ONNX model: {output_file}')



if __name__ == '__main__':
    model_path = "./dbnet.pth"
    output_file = "./dbnet.onnx"
    batchsize = 1
    img_w = 640
    img_h = 640
    sshape = [batchsize, 3, img_w, img_h]

    ckpt = torch.load(model_path, map_location='cpu')
    cfg = ckpt['cfg']
    model = build_model(cfg['model'])

    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)

    model.eval()
    pytorch2onnx(
        model,
        sshape,
        output_file=output_file)