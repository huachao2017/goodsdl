import os
import logging
from dl import common

import tensorflow as tf

# from datasets import dataset_utils

def create_step20_goods(source_dir, output_dir):
    dirlist = os.listdir(source_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirlist)):
        class_dir = os.path.join(source_dir, dirlist[i])
        if os.path.isdir(class_dir):
            class_sep = dirlist[i].split('-')
            if len(class_sep) != 2:
                logging.error('class dir error: {}'.format(dirlist[i]))
                continue
            cluster_class = class_sep[1]
            child_class = class_sep[0]
            logging.info('copy class:{}-{}'.format(cluster_class, child_class))
            output_cluster_class_dir = os.path.join(output_dir, cluster_class)
            if not tf.gfile.Exists(output_cluster_class_dir):
                tf.gfile.MakeDirs(output_cluster_class_dir)

            output_class_dir = os.path.join(output_cluster_class_dir, child_class)
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)
            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                example, ext = os.path.splitext(image_path)
                if ext == ".jpg":
                    logging.info('copy image:{}'.format(image_path))
                    output_image_path = os.path.join(output_class_dir, filelist[j])
                    if tf.gfile.Exists(output_image_path):
                        # 文件存在不再重新生成，从而支持增量生成
                        continue
                    tf.gfile.Copy(image_path,output_image_path)

    tf.gfile.Copy(os.path.join(source_dir, common.CLUSTER_FILE_NAME), os.path.join(output_dir, common.CLUSTER_FILE_NAME))
    logging.info("step20 copy complete")

tf.app.flags.DEFINE_string(
    'source_dir_name', 'step20_from_step2',
    'source dir name')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # if not FLAGS.day_hour:
    #     raise ValueError('You must supply day and hour --day_hour')
    logger = logging.getLogger()
    logger.setLevel('INFO')
    dataset_dir = '/home/src/goodsdl/media/dataset'
    source_dir = os.path.join(dataset_dir, FLAGS.source_dir_name)
    step20_dir = os.path.join(dataset_dir, common.STEP20_PREFIX)

    create_step20_goods(source_dir, step20_dir)

if __name__ == '__main__':
    tf.app.run()
