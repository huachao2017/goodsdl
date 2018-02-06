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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf

from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class TfExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self):
    """Constructor sets keys_to_features and items_to_handlers.

    """
    self.keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, 1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, 1),
        # Object classes.
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
    }
    self.items_to_handlers = {
        fields.InputDataFields.image: slim_example_decoder.Image(
            image_key='image/encoded', format_key='image/format', channels=3),
        # Object classes.
    }
    # primarily after the recent tf.contrib.slim changes make into a release
    # supported by cloudml.
    label_handler = slim_example_decoder.Tensor('image/object/class/label')
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_classes] = label_handler

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
    return tensor_dict
