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

import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from PIL import Image as im
import logging
import io
import cv2
import math
import numpy as np

import tensorflow as tf

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 100

# Seed for repeatability.
_RANDOM_SEED = 0

logger = logging.getLogger("django")


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
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_tfrecord_filename(output_dir, split_name):
    output_filename = 'goods_recogonize_%s.tfrecord' % (
        split_name)
    return os.path.join(output_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, output_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      output_dir: The directory where the converted tfrecord are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(len(class_names_to_ids))))

    output_filename = _get_tfrecord_filename(
        output_dir, split_name)
    writer = tf.python_io.TFRecordWriter(output_filename)
    for shard_id in range(len(class_names_to_ids)):
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
        for i in range(start_ndx, end_ndx):
            logger.info('\r>> Converting image %d/%d shard %d' % (
                i + 1, len(filenames), shard_id))

            # Read the filename:
            with tf.gfile.GFile(filenames[i], 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = im.open(encoded_jpg_io)
            width, height = image.size

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                encoded_jpg, b'jpg', height, width, class_id)
            writer.write(example.SerializeToString())
    writer.close()
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

def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,
                          borderValue=(int(src[1][1][0]), int(src[1][1][1]), int(src[1][1][2])))


def create_step2_goods(data_dir, dataset_dir, step1_model_path):
    graph_step1 = tf.Graph()
    with graph_step1.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(step1_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
    session_step1 = tf.Session(graph=graph_step1, config=config)

    # Definite input and output Tensors for detection_graph
    image_tensor_step1 = graph_step1.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = graph_step1.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = graph_step1.get_tensor_by_name('detection_scores:0')

    """返回所有图片文件路径"""
    dirlist = os.listdir(data_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirlist)):
        class_dir = os.path.join(data_dir, dirlist[i])
        if os.path.isdir(class_dir):
            logger.info('solve class:{}'.format(dirlist[i]))
            output_class_dir = os.path.join(dataset_dir, dirlist[i])
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)
            else:
                # TODO 需要判断目录里面的文件是否需要更新
                continue

            # output_tmp_dir = os.path.join(output_class_dir, 'tmp')
            # if not tf.gfile.Exists(output_tmp_dir):
            #     tf.gfile.MakeDirs(output_tmp_dir)

            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                example, ext = os.path.splitext(image_path)
                xml_path = example + '.xml'
                if ext == ".jpg" and os.path.isfile(xml_path):
                    logger.info('solve image:{}'.format(image_path))
                    image = im.open(image_path)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    # TODO 还未支持多类型
                    index = 0
                    for box in root.iter('bndbox'):
                        index = index + 1
                        # 改变xml中的坐标值
                        xmin = int(box.find('xmin').text)
                        ymin = int(box.find('ymin').text)
                        xmax = int(box.find('xmax').text)
                        ymax = int(box.find('ymax').text)
                        newimage = image.crop((xmin, ymin, xmax, ymax))
                        # 生成新的图片
                        output_image_path = os.path.join(output_class_dir,
                                                         "{}_{}.jpg".format(os.path.split(example)[1], index))
                        newimage.save(output_image_path, 'JPEG')
                        logger.info('save image:{}'.format(output_image_path))

                        # img = cv2.imread(image_path)

                        # augment small sample
                        if len(filelist) < 3 * 8:
                            augment_ratio = 5
                        elif len(filelist) < 3 * 20:
                            augment_ratio = 4
                        elif len(filelist) < 3 * 30:
                            augment_ratio = 3
                        elif len(filelist) < 3 * 40:
                            augment_ratio = 2
                        else:
                            augment_ratio = 1
                        # 使图像旋转
                        for k in range(6 * augment_ratio - 1):
                            angle = 60 / augment_ratio + k * 60 / augment_ratio
                            radian = angle * math.pi / 180
                            tf_img = tf.contrib.image.rotate(image, radian)
                            with tf.Session(config=config) as sess:
                                sess.run(tf.global_variables_initializer())
                                augment_image = sess.run(tf_img)
                            # rotated_img = rotate_image(img, angle)
                            # 写入图像
                            # tmp_image_path = os.path.join(output_tmp_dir,
                            #                               "{}_{}_{}.jpg".format(os.path.split(example)[1], index, k))
                            # cv2.imwrite(tmp_image_path, rotated_img)
                            logger.info("image:{} rotate {}.".format(output_image_path, angle))

                            output_image_path_augment = os.path.join(output_class_dir, "{}_{}_augment{}.jpg".format(
                                os.path.split(example)[1], index, angle))
                            # augment_image = im.open(tmp_image_path)
                            (im_width, im_height) = augment_image.size
                            image_np = np.array(augment_image.getdata()).reshape(
                                (im_height, im_width, 3)).astype(np.uint8)
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            # Actual detection.
                            (boxes, scores) = session_step1.run(
                                [detection_boxes, detection_scores],
                                feed_dict={image_tensor_step1: image_np_expanded})
                            # data solving
                            boxes = np.squeeze(boxes)
                            # classes = np.squeeze(classes).astype(np.int32)
                            scores_step1 = np.squeeze(scores)
                            for l in range(boxes.shape[0]):
                                if scores_step1[l] > 0.5:
                                    ymin, xmin, ymax, xmax = boxes[l]
                                    ymin = int(ymin * im_height)
                                    xmin = int(xmin * im_width)
                                    ymax = int(ymax * im_height)
                                    xmax = int(xmax * im_width)

                                    augment_newimage = augment_image.crop((xmin, ymin, xmax, ymax))
                                    augment_newimage.save(output_image_path_augment, 'JPEG')
                                    logger.info("save image:{}.".format(output_image_path_augment))
                                    break
    session_step1.close()


def prepare_train(data_dir, train_dir, train_name, step1_model_path):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """

    output_dir = os.path.join(train_dir, train_name)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    _remove_tfrecord_ifexists(output_dir)

    dataset_dir = os.path.join(output_dir, 'step2')
    # _clean_up_temporary_files(dataset_dir)
    create_step2_goods(data_dir, dataset_dir, step1_model_path)

    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     output_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     output_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, output_dir)

    logger.info('Finished converting the goods dataset!')
    return class_names_to_ids, training_filenames, validation_filenames
