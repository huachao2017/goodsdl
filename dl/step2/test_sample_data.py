import logging
import os

import cv2
import django
import numpy as np

from dl import common
from tradition.matcher.matcher import Matcher
import shutil

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import ExportAction, SampleImageClass
from django.conf import settings


import tensorflow as tf

# from datasets import dataset_utils

def test_one_class(matcher,
                   class_dir,
                   class_name,
                   output_dir
                   ):
    if class_name == 'ziptop-drink-stand' or class_name == 'bottled-drink-stand':
        return 0
    error_cnt = 0
    filelist = os.listdir(class_dir)
    for j in range(0, len(filelist)):
        image_path = os.path.join(class_dir, filelist[j])
        prefix = filelist[j].split('_')[0]
        example, ext = os.path.splitext(image_path)
        if ext != ".jpg" or prefix == 'visual':
            continue

        logging.info('test image:{}'.format(image_path))
        upc, score = matcher.match_image_best_one(image_path,filter_upcs=[class_name],visual=False,debug=False)
        if upc != class_name:
            error_cnt += 1
            output_image_path = os.path.join(output_dir, '{}_{}.jpg'.format(class_name,error_cnt))
            shutil.copy(image_path, output_image_path)
            matcher.match_image_best_one(image_path, filter_upcs=[class_name],debug=True)
    return error_cnt

def test_sample(data_dir, output_dir):
    error_total = 0
    dirlist = os.listdir(data_dir)  # 列出文件夹下所有的目录与文件
    matcher = Matcher()
    samples = SampleImageClass.objects.filter(deviceid='')
    for sample in samples:
        if os.path.isfile(sample.source.path):
            matcher.add_baseline_image(sample.source.path, sample.upc)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    for i in range(0, len(dirlist)):
        class_name = dirlist[i]
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            logging.info('test class:{}'.format(class_name))
            error_total += test_one_class(
                matcher,
                class_dir,
                class_name,
                output_dir
            )

    logging.info("sample test complete, error count: {}".format(error_total))

tf.app.flags.DEFINE_string(
    'source_dir_serial', '',
    'source dir serial')
tf.app.flags.DEFINE_string(
    'dest_dir_serial', '',
    'dest dir serial')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # if not FLAGS.day_hour:
    #     raise ValueError('You must supply day and hour --day_hour')
    logger = logging.getLogger()
    logger.setLevel('INFO')
    dataset_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME)
    sample_dir = os.path.join(dataset_dir, common.SAMPLE_PREFIX if FLAGS.dest_dir_serial=='' else common.SAMPLE_PREFIX+'_'+FLAGS.dest_dir_serial)
    test_sample_dir = os.path.join(dataset_dir,'test_sample')

    test_sample(sample_dir, test_sample_dir)

if __name__ == '__main__':
    tf.app.run()
