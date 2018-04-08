import os
from dl.step2.cluster import ClusterSettings
from dl import common
import tensorflow as tf
import shutil
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import TrainTask

tf.app.flags.DEFINE_string(
    'dir_serial', '',
    'dir serial')
FLAGS = tf.app.flags.FLAGS

def create_one_task(dataset_dir):
    TrainTask.objects.create(
        dataset_dir=dataset_dir,
    )

def main(_):
    dataset_dir = '/home/src/goodsdl/media/dataset'
    step20_dir = os.path.join(dataset_dir, common.STEP20_PREFIX if FLAGS.dir_serial=='' else common.STEP20_PREFIX+'_'+FLAGS.dir_serial)
    step30_dir = os.path.join(dataset_dir, common.STEP30_PREFIX if FLAGS.dir_serial=='' else common.STEP30_PREFIX+'_'+FLAGS.dir_serial)
    # cluster_filepath = os.path.join(step20_dir, common.CLUSTER_FILE_NAME)
    # cluster_settings = ClusterSettings(cluster_filepath)
    # traintype_to_class_names = cluster_settings.get_traintype_to_class_names()
    # print(traintype_to_class_names)
    shutil.copytree(step20_dir,step30_dir)

    for name in os.listdir(step30_dir):
        dataset_dir = os.path.join(step30_dir, name)
        if os.path.isdir(dataset_dir):
            create_one_task(dataset_dir)

if __name__ == '__main__':
    tf.app.run()
