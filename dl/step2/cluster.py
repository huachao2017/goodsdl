import tensorflow as tf

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
        self.lines = filter(None, lines)

    def get_class_names_to_cluster_class_names(self):
        # class_names_to_cluster_class_names={'222222222':'111111111','333333333':'111111111'}
        class_names_to_cluster_class_names = {}
        for line in self.lines:
            sep = line.split(':')
            class_names_to_cluster_class_names[sep[2]] = line[sep[1]]

        return class_names_to_cluster_class_names

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


# def get_class_names_to_cluster_class_names(cluster_filepath):
#     """
#
#     :param cluster_filepath:
#     :return: class_names_to_class_name
#     :example file content:
#     111111111:222222222
#     111111111:333333333
#     class_names_to_class_name={'222222222':'111111111','333333333':'111111111'}
#     """
#     with tf.gfile.Open(cluster_filepath, 'rb') as f:
#         lines = f.read().decode()
#     lines = lines.split('\r\n') # TODO use windows to edit
#     lines = filter(None, lines)
#
#     class_names_to_cluster_class_names = {}
#     for line in lines:
#         index = line.index(':')
#         class_names_to_cluster_class_names[line[index + 1:]] = line[:index]
#
#     return class_names_to_cluster_class_names
