# PytorchOCR

## 简介
PytorchOCR旨在打造一套训练，推理，部署一体的OCR引擎库

## 更新日志
* 2020.06.25 检测模块的训练和预测ok
* 2020.06.18 更新README
* 2020.06.17 识别模块的训练和预测ok

## todo list
* [ ] 识别模型预训练模型
* [ ] DB通用模型
* [ ] 手机端部署

## 环境配置

需要的环境如下
* pytorch 1.4+
* torchvision 0.5+
* gcc 4.9+ (pse,pan会用到)

快速安装环境
```bash
pip3 install -r requirements.txt
```

## 模型下载

链接：https://pan.baidu.com/s/1oCWJVyEpGAeagE4EwoV0kA 
提取码：vvvx

## 文档教程
* [文字检测](doc/检测.md)
* [文字识别](doc/识别.md)

## 文本检测算法

PytorchOCR开源的文本检测算法列表：
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))

在ICDAR2015文本检测公开数据集上，算法效果如下：


| 模型 | 骨干网络 | precision | recall | Hmean | 下载链接 |
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
|DB|ResNet18_vd|90.56%|72.66%|80.63%|见百度网盘|

## 文本识别算法

PytorchOCR开源的文本识别算法列表：
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))

## 结果展示

![检测](doc/imgs/exampl1.png)

## 贡献代码
我们非常欢迎你为PytorchOCR贡献代码，也十分感谢你的反馈。

## 相关仓库
* https://github.com/WenmuZhou/OCR_DataSet