# PytorchOCR


- [PytorchOCR](#pytorchocr)
  - [模型对齐信息](#模型对齐信息)
    - [环境](#环境)
    - [对齐列表](#对齐列表)
  - [TODO](#todo)
  - [使用方式](#使用方式)
    - [数据准备](#数据准备)
    - [train](#train)
    - [eval](#eval)
    - [infer](#infer)
    - [export](#export)
    - [predict](#predict)

从PaddleOCR转换模型到PytorchOCR

## 模型对齐信息

### 环境

- torch: 2.0.1
- paddle: 2.5.1
- 系统：win10 cpu

### 对齐列表

注意：不在下述列表中的模型代表还未经过验证

模型下载地址 

百度云: 链接：https://pan.baidu.com/s/1rLYJt647EE0341mfHjSWMg?pwd=uyea 提取码：uyea

| 模型 | 是否对齐 | 对齐误差| 配置文件 |
|---|---|---|---|
| ch_PP-OCRv4_rec_distill   | X | 配置不一致 | [config](configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_distill.yml) |
| ch_PP-OCRv4_rec_teacher   | Y | 1.4605024e-10 | [config](configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml) |
| ch_PP-OCRv4_rec_student  | Y | 3.6277156e-06 | [config](configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml) |
| ch_PP-OCRv4_det_student  | Y | 0 | [config](configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml) |
| ch_PP-OCRv4_det_teacher  | Y | maps 7.811429e-07 <br> cbn_maps 1.0471307e-06 | [config](configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml) |
| ch_PP-OCRv4_det_cml  | Y | Student_res 0.0 <br> Student2_res 0.0 <br> Teacher_maps 1.1398747e-06 <br> Teacher_cbn_maps 1.2791393e-06 | [config](configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml) |
| ch_PP-OCRv3_rec  | Y | 4.615016e-11 | [config](configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml) |
| ch_PP-OCRv3_rec_distillation.yml  | Y | Teacher_head_out_res 7.470646e-10 <br> Student_head_out_res 4.615016e-11 | [config](configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) |
| ch_PP-OCRv3_det_student  | Y | 1.766314e-07 | [config](cconfigs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml) |
| ch_PP-OCRv3_det_cml  | Y | Student_res 1.766314e-07 <br> Student2_res 3.1212483e-07 <br> Teacher_res 8.829421e-08 | [config](configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) |
| ch_PP-OCRv3_det_dml  | Y | ok | [config](configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml) |
| cls_mv3  | Y | 5.9604645e-08 | [config](configs/cls/cls_mv3.ymll) |

## TODO

功能性：

- [x] 端到端推理
- [x] det推理
- [x] rec推理
- [x] cls推理
- [x] 导出为onnx
- [x] onnx推理
- [ ] tensorrt 推理
- [x] 训练，评估，测试

模型：

- [x] PP-OCRv4 det mobile
- [x] PP-OCRv4 det server
- [x] PP-OCRv4 rec mobile
- [x] PP-OCRv4 rec server
- [ ] DB
- [ ] DB ++
- [ ] CRNN

## 使用方式

### 数据准备

参考PaddleOCR

### train

```sh
# 单卡
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 tools/train.py --c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```


### eval

```sh
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```


### infer

```sh
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```

### export

```sh
python tools/export.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```
会将模型导出为onnx格式(默认，torch script未做测试)，同时导出后处理和预处理参数

### predict

```sh
# det + cls + rec
python .\tools\infer\predict_system.py --det_model_dir=path/to/det/export_dir  --cls_model_dir=path/to/cls/export_dir  --rec_model_dir=path/to/rec/export_dir  --image_dir=doc/imgs/1.jpg --use_angle_cls=true

# det
python .\tools\infer\predict_det.py --det_model_dir=path/to/det/export_dir --image_dir=doc/imgs/1.jpg

# cls
python .\tools\infer\predict_cls.py --cls_model_dir=path/to/cls/export_dir --image_dir=doc/imgs/1.jpg

# rec
python tools/infer/predict_rec.py --rec_model_dir=path/to/rec/export_dir --image_dir=doc/imgs_words/en/word_1.png

```

ref:

1. https://github.com/PaddlePaddle/PaddleOCR
2. https://github.com/frotms/PaddleOCR2Pytorch
