import tensorflow as tf

def get_class_names_to_cluster_class_names(cluster_filepath):
    """

    :param cluster_filepath:
    :return: class_names_to_class_name
    :example file content:
    111111111:222222222
    111111111:333333333
    class_names_to_class_name={'222222222':'111111111','333333333':'111111111'}
    """
    with tf.gfile.Open(cluster_filepath, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\r\n') # TODO use windows to edit
    lines = filter(None, lines)

    class_names_to_cluster_class_names = {}
    for line in lines:
        index = line.index(':')
        class_names_to_cluster_class_names[line[index + 1:]] = line[:index]

    return class_names_to_cluster_class_names
