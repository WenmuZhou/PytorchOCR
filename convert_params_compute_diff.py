# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 11:34
# @Author  : zhoujun

import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import yaml
import numpy as np
import paddle
import torch

from torchocr.modeling.architectures import build_model
from ppocr.modeling.architectures import build_model as build_model_paddle
from ppocr.postprocess import build_post_process


def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def init_head(config):
    global_config = config["Global"]
    post_process_class = build_post_process(config["PostProcess"], global_config)
    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                        config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                            list(config["Loss"]["loss_config_list"][-1].keys())[0]
                            == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif (
                            list(config["Loss"]["loss_config_list"][-1].keys())[0]
                            == "DistillationNRTRLoss"
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                            char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1
    return config


def conver_params(model_config, paddle_params_path, tmp_dir, show_log=False):
    from padiff import assign_weight, create_model
    torch_model = build_model(model_config)
    paddle_model = build_model_paddle(model_config)
    if os.path.exists(paddle_params_path):
        paddle_model.set_state_dict(paddle.load(paddle_params_path))
    torch_model.eval()
    paddle_model.eval()

    torch_model_warp = create_model(torch_model)
    paddle_model_warp = create_model(paddle_model)
    torch_model_warp.auto_layer_map("base", show_log=show_log)
    paddle_model_warp.auto_layer_map("raw", show_log=show_log)
    assign_weight(torch_model_warp, paddle_model_warp)
    # for recv4 rec
    # torch2paddle(torch_model, paddle_model)
    if not os.path.exists(paddle_params_path):
        paddle_params_path = os.path.join(tmp_dir, 'paddle.pdparams')
        paddle.save(paddle_model.state_dict(), paddle_params_path)
        print(f"save default paddle params success to {paddle_params_path}")
    torch_params_path = paddle_params_path.replace(".pdparams", ".pth")
    torch.save({"state_dict": torch_model.state_dict()}, torch_params_path)
    print(f"save convert torch params to {torch_params_path}")
    return paddle_params_path, torch_params_path

def torch2paddle(torch_model: torch.nn.Module, paddle_model: paddle.nn.Layer):
    paddle_state_dict = paddle_model.state_dict()
    torch_dict = torch_model.state_dict()
    # paddle_state_dict = paddle.load(paddle_model)
    fc_names = ["qkv",'fc', 'kv', 'tgt_word_prj','q','out_proj','linear','proj']
    torch_state_dict = {}
    for k in paddle_state_dict:
        v = paddle_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("_variance", "running_var")
        k = k.replace("_mean", "running_mean")
        if torch_dict[k].numpy().shape != v.shape:
            print(torch_dict[k].numpy().shape, v.shape)
        torch_state_dict[k] = torch.from_numpy(v)

    for k in torch_state_dict:
        if k not in torch_model.state_dict():
            print(f'{k} is not in torch model')
    for k in torch_model.state_dict():
        if 'num_batches_tracked' in k:
            continue
        if k not in torch_state_dict:
            print(f'{k} is not in torch params')
    torch_model.load_state_dict(torch_state_dict)
    
def get_input(w, h, color=True):
    img = cv2.imread("doc/imgs/1.jpg", 1 if color else 0)
    img = cv2.resize(img, (w, h))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, 0).transpose([0, 3, 1, 2])
    img = img.astype("float32")
    img /= 255
    img-=0.5
    img/=0.5
    return img


def diff_func(torch_out, paddle_out, prex=""):
    if isinstance(torch_out, dict):
        for k in torch_out:
            diff_func(torch_out[k], paddle_out[k], f"{prex}_{k}")
    elif isinstance(torch_out, list):
        for k in range(len(torch_out)):
            diff_func(torch_out[k], paddle_out[k], f"{prex}_{k}")
    elif isinstance(torch_out, torch.Tensor):
        diff = paddle_out.detach().cpu().numpy() - torch_out.detach().cpu().numpy()
        print(prex[1:], np.abs(diff).mean())


def paddle_infer(config, input_np, device, params_path):
    print(f"paddle version: {paddle.__version__}")
    print(f"input shape of paddle is {input_np.shape}")
    paddle.device.set_device(device)
    x = paddle.to_tensor(input_np)
    model = build_model_paddle(config)
    model.set_state_dict(paddle.load(params_path))
    model.eval()

    y = model(x)
    return y


def torch_infer(config, input_np, device, params_path):
    print(f"torch version: {torch.__version__}")
    print(f"input shape of torch is {input_np.shape}")
    if device == 'gpu':
        device = 'cuda'
    x = torch.from_numpy(input_np)
    x = x.to(device)
    model = build_model(config)
    model.load_state_dict(torch.load(params_path)["state_dict"])
    model.eval()
    model = model.to(device)

    y = model(x)
    return y


def main():
    device = "cpu"
    input_np = get_input(320, 48, True)

    tmp_dir = './tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    config_path = "configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml"
    paddle_params_path = r''

    config = load_config(config_path)
    config = init_head(config)
    model_config = config["Architecture"]
    print(model_config)

    # step 1 convert params and  run paddle and save result
    paddle_params_path, torch_params_path = conver_params(model_config, paddle_params_path, tmp_dir, show_log=False)
    # step 2 run paddle
    paddle_out = paddle_infer(model_config, input_np, device, paddle_params_path)
    # step 2 run torch
    torch_out = torch_infer(model_config, input_np, device, torch_params_path)
    diff_func(torch_out, paddle_out)


if __name__ == "__main__":
    main()
