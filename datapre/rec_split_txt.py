#识别数据分割成train val

import os
import datapre.datapre_utils as utils

pwd_path = "/home/wwe/ocr/chinesedata/1_data/"
input_txt = "{}crnndata_out.txt".format(pwd_path)

output_train_txt = "{}crnndata_train.txt".format(pwd_path)
output_val_txt = "{}crnndata_val.txt".format(pwd_path)


f_out_train = open(output_train_txt,"w")
f_out_val = open(output_val_txt,"w")

split_scale = 500
with open(input_txt, "r") as f:
    for index, line in enumerate(f.readlines()):
        if index % split_scale == 0:
            f_out_val.write(line)
        else:
            f_out_train.write(line)
f_out_train.close()
f_out_val.close()