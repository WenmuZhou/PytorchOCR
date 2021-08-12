#识别数据任意txt格式转  pytorchOCR TextLine格式    \t作为分隔符

# " 图像文件名                 图像标注信息 "

# train_data/train_0001.jpg   简单可依赖
# train_data/train_0002.jpg   用科技让复杂的世界更简单

import os
import datapre.datapre_utils as utils

pwd_path = "/home/wwe/ocr/chinesedata/1_data/"
input_txt = "{}crnndata.txt".format(pwd_path)
input_txt_split_symbol = " " # " " or "-" or ...
out_txt_pwd_path = "{}data/".format(pwd_path)

output_txt = "{}crnndata_out.txt".format(pwd_path)


f_out = open(output_txt,"w")
with open(input_txt, "r") as f:
    for line in f.readlines():
        datas = line.split(input_txt_split_symbol)
        f_out.write(out_txt_pwd_path + "\t".join(datas))
f_out.close()
