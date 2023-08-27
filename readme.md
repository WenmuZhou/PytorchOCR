# PytorchOCR


- [PytorchOCR](#pytorchocr)
  - [模型对齐信息](#模型对齐信息)
    - [环境](#环境)
    - [对齐列表](#对齐列表)
      - [识别模型](#识别模型)

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

#### 识别模型


| 模型 | 是否对齐 | 对齐误差| 配置文件 |
|---|---|---|---|
| ch_PP-OCRv3_rec  | Y | 4.615016e-11 | [config](configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml) |
| ch_PP-OCRv3_rec_distillation.yml  | Y | Teacher_head_out_res 7.470646e-10 <br> Student_head_out_res 4.615016e-11 | [config](configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) |


## 使用方式

### 数据准备

参考PaddleOCR

### train

```sh
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```


### eval

```sh
python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```


### infer

```sh
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```

### export

```sh
python tools/export.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```
会导出会处理和预处理参数

### predict

```sh
python tools/infer/predict_rec.py -c path/to/export/config.yaml --rec_model_path=path/to/export/model.onnx --image_dir=doc/imgs_words/en/word_1.png
```