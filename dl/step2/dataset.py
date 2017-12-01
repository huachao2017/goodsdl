import os
import tensorflow as tf

slim = tf.contrib.slim


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer',
}

def _get_split(split_name, dataset_dir, example_num=2000, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in ('train', 'validation'):
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = 'goods_recogonize_%s.tfrecord'
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_filename = os.path.join(dataset_dir, 'labels.txt')
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_names[int(line[:index])] = line[index + 1:]

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=int(example_num),
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=len(labels_to_names),
        labels_to_names=labels_to_names)

