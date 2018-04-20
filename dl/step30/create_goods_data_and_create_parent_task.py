import os
from dl.step2.cluster import ClusterSettings
from dl import common
import tensorflow as tf
import shutil
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import TrainTask,ClusterStructure,ClusterEvalStep,ClusterEvalData,ClusterSampleScore,ClusterUpcScore

tf.app.flags.DEFINE_string(
    'dir_serial', '',
    'dir serial')
FLAGS = tf.app.flags.FLAGS

def create_one_task(upc_name,dataset_dir):
    TrainTask.objects.create(
        dataset_dir=dataset_dir,
    )
    father_cluster = ClusterStructure.objects.create(
        upc=upc_name
    )

    for upc in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir,upc)):
            ClusterStructure.objects.create(
                upc=upc,
                f_upc=father_cluster.upc,
            )

def main(_):
    dataset_dir = '/home/src/goodsdl/media/dataset'
    step20_dir = os.path.join(dataset_dir, common.STEP20_PREFIX if FLAGS.dir_serial=='' else common.STEP20_PREFIX+'_'+FLAGS.dir_serial)
    step30_dir = os.path.join(dataset_dir, common.STEP30_PREFIX if FLAGS.dir_serial=='' else common.STEP30_PREFIX+'_'+FLAGS.dir_serial)
    # cluster_filepath = os.path.join(step20_dir, common.CLUSTER_FILE_NAME)
    # cluster_settings = ClusterSettings(cluster_filepath)
    # traintype_to_class_names = cluster_settings.get_traintype_to_class_names()
    # print(traintype_to_class_names)
    if os.path.isdir(step30_dir):
        os.rmdir(step30_dir)

    shutil.copytree(step20_dir,step30_dir)
    ret = os.popen('chmod 777 -R {}'.format(step30_dir)).readline()

    TrainTask.objects.all().delete()
    ClusterStructure.objects.all().delete()
    ClusterEvalStep.objects.all().delete()
    ClusterEvalData.objects.all().delete()
    ClusterSampleScore.objects.all().delete()
    ClusterUpcScore.objects.all().delete()
    for upc_name in os.listdir(step30_dir):
        if upc_name == 'lack_info_stand':
            continue
        dataset_dir = os.path.join(step30_dir, upc_name)
        if os.path.isdir(dataset_dir):
            create_one_task(upc_name, dataset_dir)

if __name__ == '__main__':
    tf.app.run()
