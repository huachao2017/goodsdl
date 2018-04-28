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
                   output_class_dir
                   ):
    f_error_cnt = 0
    t_error_cnt = 0
    filelist = os.listdir(class_dir)
    for j in range(0, len(filelist)):
        image_path = os.path.join(class_dir, filelist[j])
        prefix = filelist[j].split('_')[0]
        example, ext = os.path.splitext(image_path)
        if ext != ".jpg" or prefix == 'visual':
            continue

        logging.info('test image:{}'.format(image_path))
        f_upc, f_score = matcher.match_image_best_one(image_path,filter_upcs=[class_name],visual=False,debug=False)

        if f_upc is not None:
            f_error_cnt += 1
            output_image_path = os.path.join(output_class_dir, '{}_{}_f.jpg'.format(class_name,f_error_cnt))
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)
            shutil.copy(image_path, output_image_path)
            matcher.match_image_best_one(output_image_path, filter_upcs=[class_name],visual=True,debug=False)

        t_upc, t_score = matcher.match_image_best_one(image_path,within_upcs=[class_name],visual=False,debug=False)
        if t_score < 0.8:
            t_error_cnt += 1
            output_image_path = os.path.join(output_class_dir, '{}_{}_t.jpg'.format(class_name,t_error_cnt))
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)
            shutil.copy(image_path, output_image_path)
            matcher.match_image_best_one(output_image_path, within_upcs=[class_name],visual=True,debug=False)

    return f_error_cnt, t_error_cnt

def test_sample(data_dir, output_dir):
    f_error_total = 0
    t_error_total = 0
    dirlist = os.listdir(data_dir)  # 列出文件夹下所有的目录与文件
    matcher = Matcher()
    samples = SampleImageClass.objects.filter(deviceid='')
    for sample in samples:
        if os.path.isfile(sample.source.path):
            matcher.add_baseline_image(sample.source.path, sample.upc)
    for i in range(0, len(dirlist)):
        class_name = dirlist[i]
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            if class_name == 'ziptop-drink-stand' or class_name == 'bottled-drink-stand':
                continue
            logging.info('test class:{}'.format(class_name))
            output_class_dir = os.path.join(output_dir,class_name)
            f_error, t_error= test_one_class(
                matcher,
                class_dir,
                class_name,
                output_class_dir
            )
            f_error_total += f_error
            t_error_total += t_error

    logging.info("sample test complete, error count: ({}+{})/{}={}".format(f_error_total, t_error_total, matcher.get_baseline_cnt(),(f_error_total+t_error_total)/matcher.get_baseline_cnt()))

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
