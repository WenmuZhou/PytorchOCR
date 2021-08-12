import torch
from torchocr.networks import build_model


def pytorch2onnx(model,
                 input_shape,
                 output_file='a.onnx'):
    model.eval()

    one_img = torch.randn(input_shape)

    # register_extra_symbolics(11)
    torch.onnx.export(
        model,
        one_img,
        output_file,
        input_names=['input_1'],
        output_names=['output_1'],
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        opset_version=11
    )

    print(f'Successfully exported ONNX model: {output_file}')



if __name__ == '__main__':
    model_path = "./dbnet.pth"
    output_file = "./dbnet.onnx"
    sshape = [1, 3, 640, 640]

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