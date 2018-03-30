import tensorflow as tf

class ClusterSettings:
    def __init__(self, cluster_filepath):
        """

        :param cluster_filepath:
        :example file content:
        cylinder_stand
        cylinder_hardbox
        cylinder_softbox
        cylinder_bottled
        rectangle_hardbox
        rectangle_softbox
        rectangle_thinedge
        rectangle_special
        bottled_drink_stand
        lack_info_stand
        """
        with tf.gfile.Open(cluster_filepath, 'rb') as f:
            lines = f.read().decode()
        lines = lines.split('\r\n')  # TODO use windows to edit
        self.classnames = []
        for line in lines:
            self.classnames.append(line)

    def get_class_names(self):
        return self.classnames



