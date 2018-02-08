# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import logging

import tensorflow as tf
from nets import nets_factory

from dl.step2 import dataset as step2_ds
from dl.step2 import eval2_util
from dl.step2.utils import classify_evaluation

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'example_num', '100', 'example num')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the tf_dataset files are stored.')

tf.app.flags.DEFINE_string(
    'source_dataset_dir', None, 'The directory where the source dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS



def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = step2_ds.get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir, FLAGS.example_num)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        def create_input_dict_fn():
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                shuffle=False,
                common_queue_capacity=2 * FLAGS.batch_size,
                common_queue_min=FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            return image, label

        tensor_dict = eval2_util.extract_prediction_tensors(
            network_fn,
            create_input_dict_fn=create_input_dict_fn)

        def _process_batch(tensor_dict, sess, batch_index, counters):
            """Evaluates tensors in tensor_dict, visualizing the first K examples.

            This function calls sess.run on tensor_dict, evaluating the original_image
            tensor only on the first K examples and visualizing detections overlaid
            on this original_image.

            Args:
              tensor_dict: a dictionary of tensors
              sess: tensorflow session
              batch_index: the index of the batch amongst all batches in the run.
              counters: a dictionary holding 'success' and 'skipped' fields which can
                be updated to keep track of number of successful and failed runs,
                respectively.  If these fields are not updated, then the success/skipped
                counter values shown at the end of evaluation will be incorrect.

            Returns:
              result_dict: a dictionary of numpy arrays
            """
            try:
                result_dict = sess.run(tensor_dict)
                counters['success'] += 1
            except tf.errors.InvalidArgumentError:
                tf.logging.info('Skipping image')
                counters['skipped'] += 1
                return {}
            if not set([
                'original_image', 'detection_scores', 'label'
            ]).issubset(set(result_dict.keys())):
                raise ValueError('result_dict does not contain all expected keys.')
            global_step = tf.train.global_step(sess, tf.train.get_global_step())
            if batch_index < 10000: # TODO
                tag = 'image-{}-{}'.format(result_dict['label'], batch_index)
                eval2_util.visualize_detection_results(
                    result_dict,
                    tag,
                    global_step,
                    labels_to_names=dataset.labels_to_names,
                    summary_dir=FLAGS.eval_dir,
                    source_dataset_dir=FLAGS.source_dataset_dir)
            return result_dict

        variables_to_restore = tf.global_variables()
        global_step = tf.train.get_or_create_global_step()
        variables_to_restore.append(global_step)
        saver = tf.train.Saver(variables_to_restore)

        def _restore_latest_checkpoint(sess):
            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, latest_checkpoint)

        metrics = eval2_util.repeated_checkpoint_run(
            tensor_dict=tensor_dict,
            summary_dir=FLAGS.eval_dir,
            evaluators=[classify_evaluation.PascalClassifyEvaluator(
                labels_to_names=dataset.labels_to_names)],
            batch_processor=_process_batch,
            checkpoint_dirs=[FLAGS.checkpoint_path],
            variables_to_restore=None,
            restore_fn=_restore_latest_checkpoint,
            num_batches=dataset.num_samples,
            eval_interval_secs=600,
            max_number_of_evaluations=dataset.num_samples,
            master=FLAGS.master,
            save_graph=True,
            save_graph_dir=(FLAGS.eval_dir))

if __name__ == '__main__':
    tf.app.run()
