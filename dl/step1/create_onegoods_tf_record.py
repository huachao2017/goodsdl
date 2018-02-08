# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford good dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/goods/

Example usage:
    ./create_good_tf_record --data_dir=/home/user/good \
        --output_dir=/home/user/good/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf
import shutil

from object_detection.utils import dataset_util

logger = logging.getLogger("dataset")

def get_class_name_from_filename(file_name):
    """Gets the class name from a file.

    Args:
      file_name: The file name to get the class name from.
                 ie. "american_pit_bull_terrier_105.jpg"

    Returns:
      A string of the class name.
    """
    return file_name.split('_')[0]
    # match = re.match(r'([A-Za-z0-9_]+)(_[0-9]+\.jpg)', file_name, re.I)
    # return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       example,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = example
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = '1'
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example

def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}


def create_tf_record(output_filename,
                     label_map_dict,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = example + '.xml'

        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = recursive_parse_xml_to_dict(xml)['annotation']

        # TODO 必须为jpg后缀的图片
        tf_example = dict_to_tf_example(data, label_map_dict, example + '.jpg')
        writer.write(tf_example.SerializeToString())

    writer.close()

def create_label_map_file(output_filename,
                          label_map_dict):
    with open(output_filename, 'w') as output:
        for key in label_map_dict:
            output.write('\nitem {\n')
            output.write("  id: {}\n".format(label_map_dict[key]))
            output.write("  name: '{}'\n".format(key))
            output.write("}\n")

def update_config_file(train_dir,
                       train_name,
                       num_classes,
                       num_steps=200000,
                       is_fineture=False):
    file_path, _ = os.path.split(os.path.realpath(__file__))
    config_template_file_path = os.path.join(file_path, 'faster_rcnn_nas_goods.config.template')
    output_filename = os.path.join(train_dir, train_name, 'faster_rcnn_nas_goods.config')
    with open(config_template_file_path, 'r') as file:
        data = file.read()
        #     p = re.compile(r'num_classes: \d+')
        output = re.sub('num_classes: \d+', 'num_classes: '+str(num_classes), data)
        output = re.sub('# num_steps: \d+', 'num_steps: '+str(num_steps), output)
        if is_fineture:
            # FIXME need restore some new checkpoint
            output = re.sub('fine_tune_checkpoint: ""', 'fine_tune_checkpoint:"'+train_dir+'/model.ckpt"', output)
        output = re.sub('PATH_TO_BE_CONFIGURED_TRAIN', os.path.join(train_dir, train_name), output)
    with open(output_filename, 'w') as file:
        file.write(output)

def read_examples_list_and_label_map_and_classnames(path):
    """返回所有图片文件路径"""
    dirlist = os.listdir(path)  # 列出文件夹下所有的目录与文件
    examples = []
    class_names = []
    for i in range(0, len(dirlist)):
        class_dir = os.path.join(path, dirlist[i])
        if os.path.isdir(class_dir):
            class_names.append(dirlist[i])
            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                example, ext = os.path.splitext(image_path)
                if ext == ".jpg" and os.path.isfile(example + '.xml'):
                    examples.append(example)
    return examples, {'1':1},sorted(class_names)

def prepare_train(data_dir, train_dir, train_name, is_fineture=False):
    logging.info('Reading from one good dataset.')
    examples_list, label_map_dict, class_names = read_examples_list_and_label_map_and_classnames(data_dir)
    logging.info(label_map_dict)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    output_dir = os.path.join(train_dir, train_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    train_output_path = os.path.join(output_dir, 'goods_train.record')
    val_output_path = os.path.join(output_dir, 'goods_val.record')
    label_map_file_path = os.path.join(output_dir, 'goods_label_map.pbtxt')
    create_tf_record(train_output_path, label_map_dict, train_examples)
    create_tf_record(val_output_path, label_map_dict, val_examples)

    create_label_map_file(label_map_file_path, label_map_dict)
    # class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    from datasets import dataset_utils
    dataset_utils.write_label_file(labels_to_class_names, output_dir)

    # 设定每张照片训练20次
    update_config_file(train_dir, train_name, len(label_map_dict), num_steps=num_examples*100, is_fineture=is_fineture)
    return label_map_dict

def prepare_train(data_dir, train_dir, train_name, is_fineture=False):
    logging.info('Reading from one good dataset.')
    examples_list, label_map_dict, class_names = read_examples_list_and_label_map_and_classnames(data_dir)
    logging.info(label_map_dict)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    output_dir = os.path.join(train_dir, train_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    train_output_path = os.path.join(output_dir, 'goods_train.record')
    val_output_path = os.path.join(output_dir, 'goods_val.record')
    label_map_file_path = os.path.join(output_dir, 'goods_label_map.pbtxt')
    create_tf_record(train_output_path, label_map_dict, train_examples)
    create_tf_record(val_output_path, label_map_dict, val_examples)

    create_label_map_file(label_map_file_path, label_map_dict)
    # class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    from datasets import dataset_utils
    dataset_utils.write_label_file(labels_to_class_names, output_dir)

    # 设定每张照片训练20次
    update_config_file(train_dir, train_name, len(label_map_dict), num_steps=num_examples*100, is_fineture=is_fineture)
    return label_map_dict

def prepare_rawdata_update_train(data_dir, train_dir, train_name):
    logging.info('Reading from one good dataset.')
    examples_list, label_map_dict, class_names = read_examples_list_and_label_map_and_classnames(data_dir)
    logging.info(label_map_dict)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    train_examples = examples_list
    logging.info('%d training examples.',
                 len(train_examples))

    output_dir = os.path.join(train_dir, train_name)

    train_output_path = os.path.join(output_dir, 'goods_train.record')
    import datetime
    now = datetime.datetime.now()
    shutil.move(train_output_path, train_output_path+'.'+ str(now.time()))
    create_tf_record(train_output_path, label_map_dict, train_examples)

    # class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # Finally, write the labels file:

    # 设定每张照片训练100次
    config_file_path = os.path.join(output_dir, 'faster_rcnn_nas_goods.config')
    checkpoint = tf.train.latest_checkpoint(output_dir)
    with open(config_file_path, 'r') as file:
        data = file.read()
        #     p = re.compile(r'num_classes: \d+')
        output = re.sub('# num_steps: \d+', 'num_steps: '+str(106300+len(train_examples)*100), data)

    shutil.move(config_file_path, config_file_path+'.'+ str(now.time()))
    with open(config_file_path, 'w') as file:
        file.write(output)
