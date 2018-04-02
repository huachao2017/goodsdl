r"""and converts Step1 data to TFRecords of TF-Example protos.

This module reads the files
that make up the goods data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
from PIL import Image as im
import logging
import io
import math
import shutil
from dl.step2 import cluster

import tensorflow as tf

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 300

# Seed for repeatability.
_RANDOM_SEED = 0

logger = logging.getLogger("dataset")


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(dir_name)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

def _get_split_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(dir_name)

    train_photo_filenames = []
    validation_photo_filenames = []
    for directory in directories:
        local_filenames = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                local_filenames.append(path)

        count = len(local_filenames)
        if count>0:
            validation_indexes = []
            for c in range(5):
                validation_indexes.append(random.randint(0, count - 1))

            for i in range(count):
                if i in validation_indexes:
                    validation_photo_filenames.append(local_filenames[i])
                    if count < 20:# 样本太少的不减少训练样本
                        train_photo_filenames.append(local_filenames[i])
                else:
                    train_photo_filenames.append(local_filenames[i])

    return train_photo_filenames, validation_photo_filenames, sorted(class_names)

def _get_test_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(dir_name)

    validation_photo_filenames = []
    for directory in directories:
        local_filenames = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                local_filenames.append(path)

        count = len(local_filenames)
        if count>0:
            validation_indexes = []
            for c in range(5):
                validation_indexes.append(random.randint(0, count-1))

            for i in range(count):
                if i in validation_indexes:
                    validation_photo_filenames.append(local_filenames[i])

    return validation_photo_filenames, sorted(class_names)

def _get_tfrecord_filename(output_dir, split_name):
    output_filename = 'goods_recogonize_%s.tfrecord' % (
        split_name)
    return os.path.join(output_dir, output_filename)


def _convert_dataset(split_name, filenames, names_to_labels, output_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      names_to_labels: A dictionary from class names (strings) to ids
        (integers).
      output_dir: The directory where the converted tfrecord are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(len(names_to_labels))))

    output_filename = _get_tfrecord_filename(
        output_dir, split_name)
    if tf.gfile.Exists(output_filename):
        tf.gfile.Remove(output_filename)
    writer = tf.python_io.TFRecordWriter(output_filename)
    for shard_id in range(len(names_to_labels)):
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
        for i in range(start_ndx, end_ndx):
            # logger.info('\r>> Converting image %d/%d shard %d' % (
            #     i + 1, len(filenames), shard_id))

            # Read the filename:
            with tf.gfile.GFile(filenames[i], 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = im.open(encoded_jpg_io)
            width, height = image.size

            name = os.path.basename(os.path.dirname(filenames[i]))
            label = names_to_labels[name]
            print('{}:{}'.format(filenames[i],label))
            example = dataset_utils.image_to_tfexample(
                encoded_jpg, b'jpg', height, width, label)
            writer.write(example.SerializeToString())
    writer.close()
    # print('generate tfrecord:{}'.format(output_filename))
    logger.info('generate tfrecord:{}'.format(output_filename))


def _clean_up_temporary_files(dataset_dir):
    if tf.gfile.Exists(dataset_dir):
        tf.gfile.DeleteRecursively(dataset_dir)


def _remove_tfrecord_ifexists(output_dir):
    for split_name in ['train', 'validation']:
        output_filename = _get_tfrecord_filename(
            output_dir, split_name)
        if tf.gfile.Exists(output_filename):
            tf.gfile.Remove(output_filename)


def prepare_train(dataset_dir, output_dir):
    """Runs the conversion operation.

    Args:
      dataset_dir: The source directory where the step2 dataset is stored.
      output_dir: tfrecord will be stored.
    """

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    training_filenames, validation_filenames, class_names = _get_split_filenames_and_classes(dataset_dir)

    names_to_labels = None
    if len(class_names) > 1:
        names_to_labels = dict(zip(class_names, range(len(class_names))))

        # Divide into train and test:
        random.seed(_RANDOM_SEED)
        random.shuffle(training_filenames)
        # training_filenames = photo_filenames[_NUM_VALIDATION:]
        # validation_filenames = photo_filenames[:_NUM_VALIDATION]

        # First, convert the training and validation sets.
        _convert_dataset('train', training_filenames, names_to_labels,
                         output_dir)
        _convert_dataset('validation', validation_filenames, names_to_labels,
                         output_dir)

        # Finally, write the labels file:
        labels_to_names = dict(zip(range(len(class_names)), class_names))
        dataset_utils.write_label_file(labels_to_names, output_dir)

        logger.info('Finished converting the goods dataset!')
    return names_to_labels, training_filenames, validation_filenames

if __name__ == '__main__':
    dataset_dir = '/home/src/goodsdl/media/dataset/step20/rectangle_softbox'
    output_dir = '/home/src/goodsdl/train/51'
    test_photo_filenames, class_names = _get_test_filenames_and_classes(dataset_dir)
    names_to_labels = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    validation_filenames = test_photo_filenames
    # validation_filenames = validation_filenames[:10]

    # First, convert the training and validation sets.
    _convert_dataset('validation', validation_filenames, names_to_labels,
                     output_dir)
    print(len(validation_filenames))
    labels_to_names = dict(zip(range(len(class_names)), class_names))
    print(labels_to_names)
