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
from PIL import Image as im
import logging
import io
import math

import tensorflow as tf

from datasets import dataset_utils
from goods2.models import TrainImage, TrainUpc, TrainActionUpcs, TrainAction, TrainModel
# Seed for repeatability.
_RANDOM_SEED = 42

logger = logging.getLogger("cron")


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
            # print('{}:{}'.format(filenames[i],label))
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


def prepare_train_TA(train_action):
    upcs = []
    train_upcs = TrainUpc.objects.all()
    for train_upc in train_upcs:
        upcs.append(train_upc.upc)

    upcs = sorted(upcs)

    training_filenames = []
    train_images = TrainImage.objects.all()
    for train_image in train_images:
        if train_image.upc in upcs:
            training_filenames.append(train_image.source.path)

    return upcs, training_filenames, None

def prepare_train_TF(train_action):
    upcs = []
    train_upcs = TrainUpc.objects.all()
    f_model = TrainModel.objects.get(id=train_action.f_model.pk)
    f_train = TrainAction.objects.get(id=f_model.train_action.pk)
    for train_upc in train_upcs:
        upcs.append(train_upc.upc)

    upcs = sorted(upcs)

    training_filenames = []
    validation_filenames = []
    old_training_filenames_to_upc = {}
    old_training_filenames = []
    train_images = TrainImage.objects.all()
    # 每类样本数
    upc_to_cnt = {}
    max_cnt = 0
    for train_image in train_images:
        if train_image.upc in upcs:
            if train_image.create_time > f_train.create_time:
                # 根据f_train.create_time增加增量样本
                training_filenames.append(train_image.source.path)
                validation_filenames.append(train_image.source.path)
                upc_to_cnt[train_image.upc] += 1
                if max_cnt < upc_to_cnt[train_image.upc]:
                    max_cnt = upc_to_cnt[train_image.upc]
            else:
                old_training_filenames_to_upc[train_image.source.path] = train_image.upc
                old_training_filenames.append(train_image.source.path)

    random.shuffle(old_training_filenames)

    # 存量样本数量上限暂时用两倍策略
    upc_cnt_thresh = max_cnt*2

    # 增加存量样本
    for upc in upcs:
        upc_to_cnt[upc] = 0
    for training_filename in old_training_filenames:
        upc = old_training_filenames_to_upc[training_filename]
        if upc_to_cnt[upc] < upc_cnt_thresh:
            training_filenames.append(training_filename)
            upc_to_cnt[upc] += 1
        elif upc_to_cnt[upc] < upc_cnt_thresh*1.5:
            validation_filenames.append(training_filename)
            upc_to_cnt[upc] += 1

    return upcs, training_filenames, validation_filenames


def prepare_train_TC(train_action):
    output_dir = train_action.train_path
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    train_upcs = TrainUpc.objects.all()
    f_model = TrainModel.objects.get(id=train_action.f_model.pk)
    f_train = TrainAction.objects.get(id=f_model.train_action.pk)
    f_train_upcs = f_train.upcs.all()

    upcs = []
    for train_upc in train_upcs:
        upcs.append(train_upc.upc)

    upcs = sorted(upcs)

    # 根据train_upcs和f_train_upcs进行样本筛选
    f_upcs = []
    for train_upc in f_train_upcs:
        f_upcs.append(train_upc.upc)

    append_upcs = []
    for upc in upcs:
        if upc not in f_upcs:
            append_upcs.append(upc)

    training_filenames = []
    validation_filenames = []
    old_training_filenames_to_upc = {}
    old_training_filenames = []
    train_images = TrainImage.objects.all()

    # 增加增量upc样本
    for train_image in train_images:
        if train_image.upc in append_upcs:
            # 根据append_upcs增加增量样本
            training_filenames.append(train_image.source.path)
            validation_filenames.append(train_image.source.path)
        else:
            old_training_filenames_to_upc[train_image.source.path] = train_image.upc
            old_training_filenames.append(train_image.source.path)


    random.shuffle(old_training_filenames)

    # 存量样本数量上限暂时用1倍策略
    upc_cnt_thresh = int(len(training_filenames)/len(append_upcs))
    upc_to_cnt = {}
    for upc in f_upcs:
        upc_to_cnt[upc] = 0
    # 增加存量样本
    for training_filename in old_training_filenames:
        upc = old_training_filenames_to_upc[training_filename]
        if upc_to_cnt[upc] < upc_cnt_thresh:
            training_filenames.append(training_filename)
            upc_to_cnt[upc] += 1
        elif upc_to_cnt[upc] < upc_cnt_thresh*2:
            validation_filenames.append(training_filename)
            upc_to_cnt[upc] += 1

    return upcs, training_filenames, validation_filenames

def prepare_train(train_action):
    output_dir = train_action.train_path
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    if train_action.action == 'TA':
        upcs, training_filenames, validation_filenames = prepare_train_TA(train_action)
    elif train_action.action == 'TF':
        upcs, training_filenames, validation_filenames = prepare_train_TF(train_action)
    elif train_action.action == 'TC':
        upcs, training_filenames, validation_filenames = prepare_train_TC(train_action)
    else:
        raise ValueError('error parameter')

    names_to_labels = dict(zip(upcs, range(len(upcs))))
    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(training_filenames)
    if validation_filenames is None:
        validation_filenames = training_filenames[int(0.7 * len(training_filenames)):]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, names_to_labels,
                     output_dir)
    _convert_dataset('validation', validation_filenames, names_to_labels,
                     output_dir)

    # Second, write the labels file:
    labels_to_names = dict(zip(range(len(upcs)), upcs))
    dataset_utils.write_label_file(labels_to_names, output_dir)

    logger.info('Finished converting the goods dataset!')
    return labels_to_names, training_filenames, validation_filenames