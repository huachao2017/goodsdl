import logging
import os

import cv2
import django
import shutil
from dl import common

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import ExportAction, SampleImageClass
from django.conf import settings


import tensorflow as tf

# from datasets import dataset_utils

def solves_one_class(class_dir,
                     class_name,
                     output_class_dir
                     ):
    if class_name == 'ziptop-drink-stand' or class_name == 'bottled-drink-stand':
        return 0
    sample_cnt = 0
    filelist = os.listdir(class_dir)
    for j in range(0, len(filelist)):
        image_path = os.path.join(class_dir, filelist[j])
        prefix = filelist[j].split('_')[0]
        postfix = filelist[j].split('_')[-1]
        if prefix == 'visual' or postfix != 'augment0.jpg':
            continue

        logging.info('solve image:{}'.format(image_path))

        output_image_path = os.path.join(output_class_dir, os.path.basename(image_path))
        shutil.copy(image_path,output_image_path)

        SampleImageClass.objects.create(
            source='{}/{}/{}/{}'.format(settings.DATASET_DIR_NAME, common.SAMPLE_PREFIX+'_'+common.STEP2S_PREFIX, class_name,os.path.basename(output_image_path)),
            deviceid=common.STEP2S_PREFIX,
            upc=class_name,
            name=class_name,
        )
        sample_cnt += 1
    return sample_cnt

def create_sample(data_dir, output_dir, step1_model_path):

    # class_names = get_class_names(os.path.join(os.path.dirname(step1_model_path), dataset_utils.LABELS_FILENAME))
    """返回所有图片文件路径"""

    sample_total = 0
    dirlist = os.listdir(data_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirlist)):
        # 根据step1的classname确定进入step2的类别
        # 不再需要，step1的检测应该可以泛化
        # if dirlist[i] not in class_names:
        #     continue

        class_name = dirlist[i]
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            logging.info('solve class:{}'.format(class_name))
            output_class_dir = os.path.join(output_dir, class_name)
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)

            sample_total += solves_one_class(
                class_dir,
                class_name,
                output_class_dir,
            )

    logging.info("sample create complete: {}".format(sample_total))

tf.app.flags.DEFINE_string(
    'device', "0",
    'gpu device id')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # if not FLAGS.day_hour:
    #     raise ValueError('You must supply day and hour --day_hour')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device
    logger = logging.getLogger()
    logger.setLevel('INFO')
    dataset_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME)
    source_dir = os.path.join(dataset_dir, common.STEP2S_PREFIX)
    sample_dir = os.path.join(dataset_dir, common.SAMPLE_PREFIX+'_'+common.STEP2S_PREFIX)
    export1s = ExportAction.objects.filter(train_action__action='T1').order_by('-update_time')[:1]
    step1_model_path = os.path.join('/home/src/goodsdl/dl/model', str(export1s[0].pk), 'frozen_inference_graph.pb')

    create_sample(source_dir, sample_dir, step1_model_path)

if __name__ == '__main__':
    tf.app.run()
