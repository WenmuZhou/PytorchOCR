# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 11:34
# @Author  : zhoujun

import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import yaml
from padiff import assign_weight, add_special_init, create_model
import numpy as np
import paddle
import torch
from torchocr.modeling.architectures import build_model
from ppocr.modeling.architectures import build_model as build_model_paddle
from ppocr.postprocess import build_post_process


def init_Embeddings(module, layer):
    param_dict = {}
    for name, param in module.state_dict().items():
        param_dict[name] = paddle.to_tensor(param.cpu().detach().numpy())
    layer.set_state_dict(param_dict)


add_special_init("torch", "Embeddings", "paddle", "Embeddings", init_Embeddings)


def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def init_head(config):
    global_config = config['Global']
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                    'name'] == 'MultiHead':  # for multi head
                    if config['PostProcess'][
                        'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    if config['PostProcess'][
                        'name'] == 'DistillationNRTRLabelDecode':
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    # update SARLoss params
                    if list(config['Loss']['loss_config_list'][-1].keys())[
                        0] == 'DistillationSARLoss':
                        config['Loss']['loss_config_list'][-1][
                            'DistillationSARLoss'][
                            'ignore_index'] = char_num + 1
                        out_channels_list['SARLabelDecode'] = char_num + 2
                    elif list(config['Loss']['loss_config_list'][-1].keys())[
                        0] == 'DistillationNRTRLoss':
                        out_channels_list['NRTRLabelDecode'] = char_num + 3

                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
            'name'] == 'MultiHead':  # for multi head
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            if config['PostProcess']['name'] == 'NRTRLabelDecode':
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list['CTCLabelDecode'] = char_num
            # update SARLoss params
            if list(config['Loss']['loss_config_list'][1].keys())[
                0] == 'SARLoss':
                if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                    config['Loss']['loss_config_list'][1]['SARLoss'] = {
                        'ignore_index': char_num + 1
                    }
                else:
                    config['Loss']['loss_config_list'][1]['SARLoss'][
                        'ignore_index'] = char_num + 1
                out_channels_list['SARLabelDecode'] = char_num + 2
            elif list(config['Loss']['loss_config_list'][1].keys())[
                0] == 'NRTRLoss':
                out_channels_list['NRTRLabelDecode'] = char_num + 3
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

        if config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
            config['Loss']['ignore_index'] = char_num - 1
    return config


def load_paddle_weight_torch(model_config, paddle_params_path):
    torch_model = build_model(model_config)
    paddle_model = build_model_paddle(model_config)
    paddle_model.set_state_dict(paddle.load(paddle_params_path))
    torch_model.eval()
    paddle_model.eval()

    torch_model_warp = create_model(torch_model)
    paddle_model_warp = create_model(paddle_model)
    torch_model_warp.auto_layer_map('base')
    paddle_model_warp.auto_layer_map('raw')
    assign_weight(torch_model_warp, paddle_model_warp)
    torch.save({'state_dict':torch_model.state_dict()}, paddle_params_path.replace('.pdparams', '.pth'))


def get_model(config_path, paddle_params_path):
    config = load_config(config_path)
    config = init_head(config)
    model_config = config['Architecture']
    print(model_config)

    load_paddle_weight_torch(model_config, paddle_params_path)

    torch_model = build_model(model_config)
    paddle_model = build_model_paddle(model_config)
    torch_model.eval()
    paddle_model.eval()
    torch_model.load_state_dict(torch.load(paddle_params_path.replace('.pdparams', '.pth'))['state_dict'])
    paddle_model.set_state_dict(paddle.load(paddle_params_path))
    return torch_model, paddle_model


def get_input(w, h, color=True):
    img = cv2.imread('doc/imgs/1.jpg', 1 if color else 0)
    img = cv2.resize(img, (w, h))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, 0).transpose([0, 3, 1, 2])
    img = img.astype("float32")
    torch_x = torch.as_tensor(img)
    paddle_x = paddle.to_tensor(img)
    return torch_x, paddle_x

def diff_func(torch_out, paddle_out, prex=''):
    if isinstance(torch_out, dict):
        for k in torch_out:
            diff_func(torch_out[k], paddle_out[k], f'{prex}_{k}')
    elif isinstance(torch_out, list):
        for k in range(len(torch_out)):
            diff_func(torch_out[k], paddle_out[k], f'{prex}_{k}')
    elif isinstance(torch_out, torch.Tensor):
        diff = paddle_out.detach().cpu().numpy() - torch_out.detach().cpu().numpy()
        print(prex[1:], np.abs(diff).mean())

device = 'cpu'
paddle.device.set_device('cpu')
torch_x, paddle_x = get_input(320, 48, True)
torch_x = torch_x.to(device)

torch_model, paddle_model = get_model('configs/cls/cls_mv3.yml',
                                      r"D:\code\OCR\new_ocr\models\ch_ppocr_mobile_v2.0_cls_train\best_accuracy.pdparams")
torch_model.to(device)
torch_out = torch_model(torch_x)
paddle_out = paddle_model(paddle_x)

diff_func(torch_out, paddle_out)
# if isinstance(torch_out, dict):
#     for k in torch_out:
#         diff = paddle_out[k].detach().cpu().numpy() - torch_out[k].detach().cpu().numpy()
#         print(f'{k}: {np.abs(diff).mean()}')
# elif isinstance(torch_out, list):
#     for k in range(len(torch_out)):
#         diff = paddle_out[k].detach().cpu().numpy() - torch_out[k].detach().cpu().numpy()
#         print(f'{k}: {np.abs(diff).mean()}')
# elif isinstance(torch_out, torch.Tensor):
#     diff = paddle_out.detach().cpu().numpy() - torch_out.detach().cpu().numpy()
#     print(np.abs(diff).mean())
# inp = (
#     {'x': paddle.to_tensor(inp)},
#     {'x': torch.as_tensor(inp)}
# )
# auto_diff(paddle_model, torch_model, inp, options={'atol': 1e-4, 'single_step':True})
