import tensorflow as tf
import os
import shutil

class ClusterSettings:
    def __init__(self, cluster_filepath):
        """

        :param cluster_filepath:
        :example file content:
        1:111111111:222222222
        1:111111111:333333333
        2:444444444:555555555
        2:444444444:555555555
        """
        with tf.gfile.Open(cluster_filepath, 'rb') as f:
            lines = f.read().decode()
        lines = lines.split('\r\n')  # TODO use windows to edit
        self.lines = []
        for line in self.lines:
            sep = line.split(':')
            if len(sep) >= 3:
                self.lines.append(line)

    def get_class_names_to_cluster_class_names(self):
        # class_names_to_cluster_class_names={'222222222':'111111111','333333333':'111111111'}
        class_names_to_cluster_class_names = {}
        for line in self.lines:
            sep = line.split(':')
            class_names_to_cluster_class_names[sep[2]] = line[sep[1]]

        return class_names_to_cluster_class_names

    def get_traintype_to_class_names(self):
        traintype_to_class_names = {}
        for line in self.lines:
            sep = line.split(':')
            traintype = int(sep[0])
            if traintype not in traintype_to_class_names:
                traintype_to_class_names[traintype] = self.get_class_names_from_traintype(traintype)

        return traintype_to_class_names

    def get_class_names_from_traintype(self, traintype):
        class_names = []
        first = False
        for line in self.lines:
            sep = line.split(':')
            if traintype == int(sep[0]):
                if not first:
                    class_names.append(sep[1])
                    first = True
                class_names.append(sep[2])
        return class_names

    def get_main_class_name_from_traintype(self, traintype):
        for line in self.lines:
            sep = line.split(':')
            if traintype == int(sep[0]):
                return sep[1]


tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/src/goodsdl/media/dataset', 'The path of the dataset dir.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    step2_dir = os.path.join(FLAGS.dataset_dir, 'step2')
    step3_dir = os.path.join(FLAGS.dataset_dir, 'step3')
    cluster_filepath = os.path.join(step2_dir, 'cluster.txt')
    print(cluster_filepath)
    cluster_settings = ClusterSettings(cluster_filepath)
    traintype_to_class_names = cluster_settings.get_traintype_to_class_names()
    print(traintype_to_class_names)

    for traintype in traintype_to_class_names:
        output_dir = os.path.join(step3_dir,str(traintype))
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
            class_names = traintype_to_class_names[traintype]
            for class_name in class_names:
                source_dir = os.path.join(step2_dir, class_name)
                shutil.copytree(source_dir, os.path.join(output_dir,class_name))
                print('{}-->{}'.format(source_dir, output_dir))


if __name__ == '__main__':
    tf.app.run()

