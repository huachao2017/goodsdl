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
        self.lines = []
        for line in lines:
            sep = line.split(':')
            if len(sep) >= 3:
                self.lines.append(line)

    def get_class_names_to_cluster_class_names(self):
        # class_names_to_cluster_class_names={'222222222':'111111111','333333333':'111111111'}
        class_names_to_cluster_class_names = {}
        for line in self.lines:
            sep = line.split(':')
            class_names_to_cluster_class_names[sep[2]] = sep[1]

        return class_names_to_cluster_class_names

    def get_traintype_to_class_names(self):
        traintype_to_class_names = {}
        for line in self.lines:
            sep = line.split(':')
            traintype = int(sep[0])
            if traintype not in traintype_to_class_names:
                traintype_to_class_names[traintype] = self.get_class_names_from_traintype(traintype)

        return traintype_to_class_names

    def get_main_class_name_to_traintype(self):
        main_class_name_to_traintype = {}
        for line in self.lines:
            sep = line.split(':')
            traintype = int(sep[0])
            main_class_name = sep[1]
            if main_class_name not in main_class_name_to_traintype:
                main_class_name_to_traintype[main_class_name] = traintype

        return main_class_name_to_traintype

    def get_max_traintype(self):
        lastline = self.lines[-1]
        sep = lastline.split(':')
        return int(sep[0])

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


