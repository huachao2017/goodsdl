from tradition.matcher.matcher import Matcher
import os
import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='matcher demo.')
parser.add_argument('image_dir', help='path', default='images/matrix')

class MatrixTest:
    def __init__(self, sample_dir):
        self._matcher = Matcher(visual=True)

        filelist = os.listdir(sample_dir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(filelist)):
            sample_image_path = os.path.join(sample_dir, filelist[i])
            if os.path.isfile(sample_image_path):
                upc, ext = os.path.splitext(os.path.basename(sample_image_path))
                if upc[0] != 'v':
                    self._matcher.add_baseline_image(sample_image_path, upc) # 基准样本不能被删除，所以直接调用_matcher的方法

    def detect_one_with_path(self,image_path):
        name, score = self._matcher.match_image_best_one(image_path)
        return name, score


if __name__ == '__main__':
    args = parser.parse_args()
    sample_dir = args.image_dir
    mt = MatrixTest(sample_dir)
    filelist = os.listdir(sample_dir)  # 列出文件夹下所有的目录与文件

    count = 0
    true_count = 0
    for i in range(0, len(filelist)):
        sample_image_path = os.path.join(sample_dir, filelist[i])
        if os.path.isfile(sample_image_path):
            upc, ext = os.path.splitext(os.path.basename(sample_image_path))
            if upc[0] != 'v':
                test_upc, score = mt.detect_one_with_path(sample_image_path)
                count += 1
                if upc == test_upc:
                    print('true:{},{}--{}'.format(test_upc, score, upc))
                    true_count += 1
                else:
                    print('false:{},{}--{}'.format(test_upc, score, upc))

    print('matrix: {},{}--{}%'.format(count,true_count,true_count/count * 100))
