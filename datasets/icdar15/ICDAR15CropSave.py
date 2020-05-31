'''
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/5/21 18:34
@File:   ICDAR15CropSave.py
'''

import os
import cv2
from glob import glob
from tqdm import tqdm


class icdar2015CropSave:
    def __init__(self, img_dir :str, gt_dir :str, save_data_dir :str,
                 train_val_split_ratio: float or None=0.1):
        self.save_id = 1
        self.img_dir = os.path.abspath(img_dir)
        self.gt_dir = os.path.abspath(gt_dir)
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        self.save_data_dir = save_data_dir
        self.train_val_split_ratio = train_val_split_ratio

    def crop_save(self) -> None:
        all_img_paths = glob(os.path.join(self.img_dir, '*.jpg'))
        all_gt_paths = glob(os.path.join(self.gt_dir, '*.txt'))
        # check length
        assert len(all_img_paths) == len(all_gt_paths)
        # create lists to store text-line
        text_lines = list()
        # start to crop and save
        for img_path in tqdm(all_img_paths):
            img = cv2.imread(img_path)
            gt_path = os.path.join(self.gt_dir, 'gt_' + os.path.basename(img_path).replace('.jpg', '.txt'))
            with open(gt_path, 'r', encoding='utf-8-sig') as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip().split(',')
                # get points
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[: 8]))
                # get transcript
                trans = line[8]
                if trans in {'', '*', '###'}:
                    continue
                # check & make dir
                save_img_dir = os.path.join(self.save_data_dir, 'images')
                if not os.path.exists(save_img_dir):
                    os.mkdir(save_img_dir)
                # build save img path
                save_img_path = os.path.join(save_img_dir, f'textbox_{self.save_id}.jpg')
                # check if rectangle
                if len({x1, y1, x2, y2, x3, y3, x4, y4}) == 4:
                    # save rectangle
                    cv2.imwrite(save_img_path, img[y1: y4, x1: x2])
                # if polygon, save minimize circumscribed rectangle
                else:
                    x_min, x_max = min((x1, x2, x3, x4)), max((x1, x2, x3, x4))
                    y_min, y_max = min((y1, y2, y3, y4)), max((y1, y2, y3, y4))
                    cv2.imwrite(save_img_path, img[y_min: y_max, x_min: x_max])
                # save to text-line
                text_lines.append(f'textbox_{self.save_id}.jpg\t{trans}\n')
                # save_id self increase
                self.save_id += 1
        if self.train_val_split_ratio:
            train = text_lines[: int(round((1-self.train_val_split_ratio)*self.save_id))]
            val = text_lines[int(round((1-self.train_val_split_ratio)*self.save_id)): ]
            # save text-line file
            with open(os.path.join(self.save_data_dir, 'train.txt'), 'w') as save_file:
                save_file.writelines(train)
            with open(os.path.join(self.save_data_dir, 'val.txt'), 'w') as save_file:
                save_file.writelines(val)
            print(f'{self.save_id-1} text-box images and 2 text-line file are saved.')
        else:
            # save text-line file
            with open(os.path.join(self.save_data_dir, 'train.txt'), 'w') as save_file:
                save_file.writelines(text_lines)
            print(f'{self.save_id-1} text-box images and 1 text-line file are saved.')


if __name__ == '__main__':
    img_dir = '/data/disk7/private/szf/Datasets/ICDAR2015/train'
    gt_dir = '/data/disk7/private/szf/Datasets/ICDAR2015/train_local_trans'
    save_data_dir = '/data/disk7/private/szf/Datasets/ICDAR2015/data'
    icdar2015CropSave(img_dir, gt_dir, save_data_dir).crop_save()