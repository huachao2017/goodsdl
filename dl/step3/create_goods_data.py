import os
from dl.step2.cluster import ClusterSettings
from dl import common
import tensorflow as tf
import shutil

tf.app.flags.DEFINE_integer(
    'traintype', 0,
    'point traintype to create, if 0 then create all')
tf.app.flags.DEFINE_string(
    'dir_serial', '',
    'dir serial')
FLAGS = tf.app.flags.FLAGS

def main(_):
    dataset_dir = '/home/src/goodsdl/media/dataset'
    step2_dir = os.path.join(dataset_dir, common.STEP2_PREFIX if FLAGS.dir_serial=='' else common.STEP2_PREFIX+'_'+FLAGS.dir_serial)
    step3_dir = os.path.join(dataset_dir, common.STEP3_PREFIX if FLAGS.dir_serial=='' else common.STEP3_PREFIX+'_'+FLAGS.dir_serial)
    cluster_filepath = os.path.join(step2_dir, common.CLUSTER_FILE_NAME)
    print(cluster_filepath)
    cluster_settings = ClusterSettings(cluster_filepath)
    traintype_to_class_names = cluster_settings.get_traintype_to_class_names()
    print(traintype_to_class_names)

    do_traintype = FLAGS.traintype
    for traintype in traintype_to_class_names:
        if do_traintype > 0 and traintype != do_traintype:
            continue
        output_dir = os.path.join(step3_dir,str(traintype))
        tf.gfile.MakeDirs(output_dir)
        class_names = traintype_to_class_names[traintype]
        for class_name in class_names:
            source_dir = os.path.join(step2_dir, class_name)
            if os.path.isdir(source_dir):
                if os.path.isdir(os.path.join(output_dir,class_name)):
                    # 防止重复目录拷贝，可以增量使用
                    # print('folder exist: {}'.format(os.path.join(output_dir,class_name)))
                    continue
                shutil.copytree(source_dir, os.path.join(output_dir,class_name))
                print('{}-->{}'.format(source_dir, output_dir))

if __name__ == '__main__':
    tf.app.run()
