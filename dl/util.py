import tensorflow as tf

def get_labels_to_names(labels_filepath):
    with tf.gfile.Open(labels_filepath, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_names[int(line[:index])] = line[index + 1:]

    return labels_to_names
