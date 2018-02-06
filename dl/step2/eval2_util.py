# author @huac
"""Common functions for repeatedly evaluating a checkpoint."""
import logging
import time
import os

import numpy as np
import tensorflow as tf
from object_detection.core import standard_fields as fields
from preprocessing import inception_preprocessing

from dl.step2.utils import tf_example_decoder
from dl.step2.utils import visualization_utils as vis_utils

slim = tf.contrib.slim
parallel_reader = tf.contrib.slim.parallel_reader


def extract_prediction_tensors(network_fn,
                                create_input_dict_fn,
                                ignore_groundtruth=False):
  """Restores the model in a tensorflow session.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    tensor_dict: A tensor dictionary with evaluations.
  """
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = input_dict[fields.InputDataFields.image]
  eval_image_size = network_fn.default_image_size

  preprocessed_image = inception_preprocessing.preprocess_for_eval(original_image, eval_image_size, eval_image_size, central_fraction=None)
  preprocessed_images = tf.expand_dims(preprocessed_image, 0)
  logits, _ = network_fn(preprocessed_images)
  probabilities = tf.nn.softmax(logits)

  input_data_fields = fields.InputDataFields()
  output_dict = {
      input_data_fields.original_image: original_image,
  }

  detection_fields = fields.DetectionResultFields
  output_dict[detection_fields.detection_scores] = probabilities

  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
    }
    output_dict.update(groundtruth)

  return output_dict

def create_input_dict(input_path):
  """Builds a tensor dictionary based on the InputReader config.

  Args:
    input_path: A input_reader_pb2.InputReader object.

  Returns:
    A tensor dict based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  _, string_tensor = parallel_reader.parallel_read(
      input_path,  # Convert `RepeatedScalarContainer` to list.
      reader_class=tf.TFRecordReader,
      num_epochs=None,
      num_readers=8,
      shuffle=True,
      dtypes=[tf.string, tf.string],
      capacity=2000,
      min_after_dequeue=1000)

  decoder = tf_example_decoder.TfExampleDecoder()
  return decoder.decode(string_tensor)

def prefetch(tensor_dict, capacity):
  """Creates a prefetch queue for tensors.

  Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a
  dequeue op that evaluates to a tensor_dict. This function is useful in
  prefetching preprocessed tensors so that the data is readily available for
  consumers.

  Example input pipeline when you don't need batching:
  ----------------------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)
  tensor_dict = prefetch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  ----------------------------------------------------

  For input pipelines with batching, refer to core/batcher.py

  Args:
    tensor_dict: a dictionary of tensors to prefetch.
    capacity: the size of the prefetch queue.

  Returns:
    a FIFO prefetcher queue
  """
  names = list(tensor_dict.keys())
  dtypes = [t.dtype for t in tensor_dict.values()]
  shapes = [t.get_shape() for t in tensor_dict.values()]
  prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes,
                                       shapes=shapes,
                                       names=names,
                                       name='prefetch_queue')
  enqueue_op = prefetch_queue.enqueue(tensor_dict)
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))
  tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name,
                                                      capacity),
                    tf.to_float(prefetch_queue.size()) * (1. / capacity))
  return prefetch_queue

def write_metrics(metrics, global_step, summary_dir):
  """Write metrics to a summary directory.

  Args:
    metrics: A dictionary containing metric names and values.
    global_step: Global step at which the metrics are computed.
    summary_dir: Directory to write tensorflow summaries to.
  """
  logging.info('Writing metrics to tf summary.')
  summary_writer = tf.summary.FileWriter(summary_dir)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    logging.info('%s: %f', key, metrics[key])
  summary_writer.close()
  logging.info('Metrics written to tf summary.')


def visualize_detection_results(result_dict,
                                tag,
                                global_step,
                                labels_to_names,
                                summary_dir='',
                                export_dir='',
                                max_num_predictions=20):
  """Visualizes detection results and writes visualizations to image summaries.

  This function visualizes an image with its detected bounding boxes and writes
  to image summaries which can be viewed on tensorboard.  It optionally also
  writes images to a directory. In the case of missing entry in the label map,
  unknown class name in the visualization is shown as "N/A".

  Args:
    result_dict: a dictionary holding groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'original_image': a numpy array representing the image with shape
          [1, height, width, 3]
        'detection_scores': a numpy array of shape [1,N]
        'groundtruth_classes': int value
      Detections are assumed to be provided in decreasing order of score and for
      display, and we assume that scores are probabilities between 0 and 1.
    tag: tensorboard tag (string) to associate with image.
    global_step: global step at which the visualization are generated.
    labels_to_names: a dict
    summary_dir: the output directory to which the image summaries are written.
    export_dir: the output directory to which images are written.  If this is
      empty (default), then images are not exported.
    max_num_predictions: maximum number of detections to visualize
  Raises:
    ValueError: if result_dict does not contain the expected keys (i.e.,
      'original_image', 'detection_scores',
      'groundtruth_classes')
  """
  if not set([
      'original_image', 'detection_scores', 'groundtruth_classes'
  ]).issubset(set(result_dict.keys())):
    raise ValueError('result_dict does not contain all expected keys.')
  logging.info('Creating detection visualizations.')

  image = result_dict['original_image']
  detection_scores = result_dict['detection_scores']
  groundtruth_class_label = result_dict['groundtruth_classes'][0]

  # Plot groundtruth underneath detections
  vis_utils.visualize_groundtruth_and_labels_on_image_array(
      image,
      detection_scores,
      groundtruth_class_label,
      labels_to_names)

  if export_dir:
    export_path = os.path.join(export_dir, 'export-{}.png'.format(tag))
    vis_utils.save_image_array_as_png(image, export_path)

  summary = tf.Summary(value=[
      tf.Summary.Value(
          tag=tag,
          image=tf.Summary.Image(
              encoded_image_string=vis_utils.encode_image_array_as_png_str(
                  image)))
  ])
  summary_writer = tf.summary.FileWriter(summary_dir)
  summary_writer.add_summary(summary, global_step)
  summary_writer.close()

  logging.info('Detection visualizations written to summary with tag %s.', tag)


def _run_checkpoint_once(tensor_dict,
                         evaluators=None,
                         batch_processor=None,
                         checkpoint_dirs=None,
                         variables_to_restore=None,
                         restore_fn=None,
                         num_batches=1,
                         master='',
                         save_graph=False,
                         save_graph_dir=''):
  """Evaluates metrics defined in evaluators.

  This function loads the latest checkpoint in checkpoint_dirs and evaluates
  all metrics defined in evaluators. The metrics are processed in batch by the
  batch_processor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    evaluators: a list of object of type DetectionEvaluator to be used for
      evaluation. Note that the metric names produced by different evaluators
      must be unique.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used --
        a DetectionModel
      will be instantiated directly. Not used if restore_fn is set.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: None, or a function that takes a tf.Session object and correctly
      restores all necessary variables from the correct checkpoint file. If
      None, attempts to restore from the first directory in checkpoint_dirs.
    num_batches: the number of batches to use for evaluation.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.
    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph
      is True this must be non-empty.

  Returns:
    global_step: the count of global steps.
    all_evaluator_metrics: A dictionary containing metric names and values.

  Raises:
    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least
      one element.
    ValueError: if save_graph is True and save_graph_dir is not defined.
  """
  if save_graph and not save_graph_dir:
    raise ValueError('`save_graph_dir` must be defined.')
  sess = tf.Session(master, graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  if restore_fn:
    restore_fn(sess)
  else:
    if not checkpoint_dirs:
      raise ValueError('`checkpoint_dirs` must have at least one entry.')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_file)

  if save_graph:
    tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')

  counters = {'skipped': 0, 'success': 0}
  with tf.contrib.slim.queues.QueueRunners(sess):
    try:
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          logging.info('Running eval ops batch %d/%d', batch + 1, num_batches)
        if not batch_processor:
          try:
            result_dict = sess.run(tensor_dict)
            counters['success'] += 1
          except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            result_dict = {}
        else:
          result_dict = batch_processor(tensor_dict, sess, batch, counters)
        for evaluator in evaluators:
          evaluator.add_single_detected_image_info(detections_dict=result_dict)
      logging.info('Running eval batches done.')
    except tf.errors.OutOfRangeError:
      logging.info('Done evaluating -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      logging.info('# success: %d', counters['success'])
      logging.info('# skipped: %d', counters['skipped'])
      all_evaluator_metrics = {}
      for evaluator in evaluators:
        metrics = evaluator.evaluate()
        evaluator.clear()
        if any(key in all_evaluator_metrics for key in metrics):
          raise ValueError('Metric names between evaluators must not collide.')
        all_evaluator_metrics.update(metrics)
      global_step = tf.train.global_step(sess, tf.train.get_global_step())
  sess.close()
  return (global_step, all_evaluator_metrics)


# TODO: Add tests.
def repeated_checkpoint_run(tensor_dict,
                            summary_dir,
                            evaluators,
                            batch_processor=None,
                            checkpoint_dirs=None,
                            variables_to_restore=None,
                            restore_fn=None,
                            num_batches=1,
                            eval_interval_secs=120,
                            max_number_of_evaluations=None,
                            master='',
                            save_graph=False,
                            save_graph_dir=''):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    summary_dir: a directory to write metrics summaries.
    evaluators: a list of object of type DetectionEvaluator to be used for
      evaluation. Note that the metric names produced by different evaluators
      must be unique.
    batch_processor: a function taking three arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
    checkpoint_dirs: list of directories to load into a DetectionModel or an
      EnsembleModel if restore_fn isn't set. Also used to determine when to run
      next evaluation. Must have at least one element.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: a function that takes a tf.Session object and correctly restores
      all necessary variables from the correct checkpoint file.
    num_batches: the number of batches to use for evaluation.
    eval_interval_secs: the number of seconds between each evaluation run.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as None the evaluation continues indefinitely.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is saved as a pbtxt file.
    save_graph_dir: where to save on disk the Tensorflow graph. If store_graph
      is True this must be non-empty.

  Returns:
    metrics: A dictionary containing metric names and values in the latest
      evaluation.

  Raises:
    ValueError: if max_num_of_evaluations is not None or a positive number.
    ValueError: if checkpoint_dirs doesn't have at least one element.
  """
  if max_number_of_evaluations and max_number_of_evaluations <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if not checkpoint_dirs:
    raise ValueError('`checkpoint_dirs` must have at least one entry.')

  last_evaluated_model_path = None
  number_of_evaluations = 0
  while True:
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime(
        '%Y-%m-%d-%H:%M:%S', time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      logging.info('No model found in %s. Will try again in %d seconds',
                   checkpoint_dirs[0], eval_interval_secs)
    elif model_path == last_evaluated_model_path:
      logging.info('Found already evaluated checkpoint. Will try again in %d '
                   'seconds', eval_interval_secs)
    else:
      last_evaluated_model_path = model_path
      global_step, metrics = _run_checkpoint_once(tensor_dict, evaluators,
                                                  batch_processor,
                                                  checkpoint_dirs,
                                                  variables_to_restore,
                                                  restore_fn, num_batches,
                                                  master, save_graph,
                                                  save_graph_dir)
      write_metrics(metrics, global_step, summary_dir)
    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)

  return metrics
