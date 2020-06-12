import os
import cv2
import numpy as np

if __name__ == '__main__':
    icdar2015_directory = '/data/OCR/ICDAR2015'
    target_directory = '/data/OCR/ICDAR2015/converted_data'
    target_image_directory = os.path.join(target_directory, 'image')
    train_image_directory = os.path.join(icdar2015_directory, 'ch4_training_images')
    train_gt_directory = os.path.join(icdar2015_directory, 'ch4_training_localization_transcription_gt')
    test_image_directory = os.path.join(icdar2015_directory, 'ch4_test_images')
    test_gt_directory = os.path.join(icdar2015_directory, 'Challenge4_Test_Task4_GT')
    os.makedirs(target_directory, exist_ok=True)
    os.makedirs(target_image_directory, exist_ok=True)
    for m_name, m_image_directory, m_gt_directory in zip(['train', 'eval'],
                                                         [train_image_directory, test_image_directory],
                                                         [train_gt_directory, test_gt_directory]):
        m_index = 0
        with open(os.path.join(target_directory, m_name + '.txt'), mode='w', encoding='utf-8') as to_write:
            for m_image_file in os.listdir(m_image_directory):
                m_gt_file = os.path.join(m_gt_directory, 'gt_' + os.path.splitext(m_image_file)[0] + '.txt')
                m_img = cv2.imread(os.path.join(m_image_directory, m_image_file))
                with open(m_gt_file, mode='r', encoding='utf-8') as to_read:
                    # 识别阶段只考虑每行中非###的字段
                    for m_line in to_read:
                        m_line = m_line.strip('\ufeff\n')
                        if not m_line.endswith('###'):
                            # 前八个为从左上角开始的四个点的坐标，这里是四个点的多边形，可能是矩形罢了，用逗号进行了间隔
                            coordinates_and_transcript = m_line.split(',')
                            # 保留字符串中唯一的一个空格，去除多个空格
                            transcript = ' '.join(''.join(coordinates_and_transcript[8:]).split())
                            if len(transcript) == 0:
                                continue
                            np_coordinates = np.array([int(_) for _ in coordinates_and_transcript[:8]]).reshape((-1, 2))
                            min_x, min_y = np.min(np_coordinates, axis=0)
                            max_x, max_y = np.max(np_coordinates, axis=0)
                            m_width = max_x - min_x + 1
                            m_height = max_y - min_y + 1
                            m_target_roi = np.zeros((m_height, m_width, m_img.shape[2]), dtype=np.uint8)
                            m_region = np.array([np_coordinates - [min_x, min_y]], dtype=np.int32)
                            m_target_roi = cv2.fillPoly(m_target_roi,
                                                        m_region,
                                                        (255,) * m_img.shape[2])
                            m_target_roi = cv2.bitwise_and(m_img[min_y:max_y + 1, min_x:max_x + 1, ...], m_target_roi)
                            target_image_name = f'{m_name}_{m_index}.jpg'
                            cv2.imwrite(os.path.join(target_image_directory, target_image_name), m_target_roi)
                            m_index += 1
                            to_write.write(f'{target_image_name}\t{transcript}\n')
                    to_write.flush()
