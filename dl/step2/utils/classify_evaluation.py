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
"""object_classify_evaluation module.

ObjectClassifyEvaluation is a class which manages ground truth information of a
object detection dataset, and computes frequently used detection metrics such as
Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import numpy as np

from object_detection.core import standard_fields


class ClassifyEvaluator(object):
  """Interface for object classify evalution classes.

  Example usage of the Evaluator:
  ------------------------------
  evaluator = ClassifyEvaluator(categories)

  # Classifys and groundtruth for image 1.
  evaluator.add_single_detected_image_info(...)

  # Classifys and groundtruth for image 2.
  evaluator.add_single_detected_image_info(...)

  metrics_dict = evaluator.evaluate()
  """
  __metaclass__ = ABCMeta

  def __init__(self, labels_to_names):
    """Constructor.

    Args:
      labels_to_names: A dicts
    """
    self._labels_to_names = labels_to_names

  @abstractmethod
  def add_single_detected_image_info(self, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      detections_dict: A dictionary of detection numpy arrays required
        for evaluation.
    """
    pass

  @abstractmethod
  def evaluate(self):
    """Evaluates detections and returns a dictionary of metrics."""
    pass

  @abstractmethod
  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    pass


class ObjectClassifyEvaluator(ClassifyEvaluator):
  """A class to evaluate detections."""

  def __init__(self,
               labels_to_names,
               metric_prefix=None):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      evaluate_corlocs: (optional) boolean which determines if corloc scores
        are to be returned or not.
      metric_prefix: (optional) string prefix for metric name; if None, no
        prefix is used.
      use_weighted_mean_ap: (optional) boolean which determines if the mean
        average precision is computed directly from the scores and tp_fp_labels
        of all classes.
    """
    super(ObjectClassifyEvaluator, self).__init__(labels_to_names)
    self._num_classes = len(labels_to_names)
    self._num_example = 0
    self._label_id_offset = 0
    self._evaluation = ObjectClassifyEvaluation(
        self._num_classes,
        label_id_offset=self._label_id_offset)
    self._metric_prefix = (metric_prefix + '/') if metric_prefix else ''

  def add_single_detected_image_info(self, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      detections_dict: A dictionary containing -
        standard_fields.DetectionResultFields.detection_scores: float32 numpy
          array of shape [num_class] containing detection scores for the boxes.
    """
    self._num_example += 1
    detection_scores = detections_dict[standard_fields.DetectionResultFields.detection_scores]
    detection_class_label = np.argpartition(-detection_scores[0],1)[0]
    groundtruth_class_label = detections_dict[standard_fields.InputDataFields.groundtruth_classes][0]
    print(detections_dict[standard_fields.InputDataFields.groundtruth_classes])
    print('{}:detection:{},groundtruth:{}'.format(self._num_example, detection_class_label, groundtruth_class_label))
    self._evaluation.add_single_detected_image_info(
      detection_class_label,
      groundtruth_class_label)

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A dictionary of metrics with the following fields -

      1. summary_metrics:
        'Precision/mAP@<matching_iou_threshold>IOU': mean average precision at
        the specified IOU threshold.

      2. per_category_ap: category specific results with keys of the form
        'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'.
    """
    (per_class_ap, mean_ap, _, _) = (
        self._evaluation.evaluate())
    pascal_metrics = {
        self._metric_prefix +
        'Precision/mAP':
            mean_ap
    }
    for idx in range(per_class_ap.size):
      if idx + self._label_id_offset in self._labels_to_names:
        display_name = (
            self._metric_prefix + 'PerformanceByCategory/AP/{}'.format(
                self._labels_to_names[idx + self._label_id_offset]))
        pascal_metrics[display_name] = per_class_ap[idx]

    return pascal_metrics

  def clear(self):
    self._num_example = 0
    """Clears the state to prepare for a fresh evaluation."""
    self._evaluation = ObjectClassifyEvaluation(
        self._num_classes,
        label_id_offset=self._label_id_offset)


class PascalClassifyEvaluator(ObjectClassifyEvaluator):
  """A class to evaluate detections using PASCAL metrics."""

  def __init__(self, labels_to_names):
    super(PascalClassifyEvaluator, self).__init__(
        labels_to_names,
        metric_prefix='PASCAL')


ObjectClassifyEvalMetrics = collections.namedtuple(
    'ObjectClassifyEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls'
    ])


class ObjectClassifyEvaluation(object):
  """Internal implementation of Pascal object detection metrics."""

  def __init__(self,
               num_groundtruth_classes,
               label_id_offset=0):
    self.num_class = num_groundtruth_classes
    self.label_id_offset = label_id_offset

    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.average_precision_per_class = np.empty(self.num_class, dtype=float)
    self.average_precision_per_class.fill(np.nan)
    self.precisions_per_class = []


  def clear_detections(self):
    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.average_precision_per_class = np.zeros(self.num_class, dtype=float)
    self.precisions_per_class = []


  def add_single_detected_image_info(self, detected_class_label, groundtruth_class_label):
    """Adds detections for a single image to be used for evaluation.

    Args:
      detected_class_label: integer .
      groundtruth_class_label: integer .

    Raises:
      ValueError: if the number of boxes, scores and class labels differ in
        length.
    """

    self.tp_fp_labels_per_class[groundtruth_class_label].append(1 if groundtruth_class_label == detected_class_label else 0)

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A named tuple with the following fields -
        average_precision: float numpy array of average precision for
            each class.
        mean_ap: mean average precision of all classes, float scalar
        precisions: List of precisions, each precision is a float numpy
            array
    """

    for class_index in range(self.num_class):
      tp_fp_labels = np.array(self.tp_fp_labels_per_class[class_index])
      true_positive_labels = tp_fp_labels
      false_positive_labels = 1 - true_positive_labels
      cum_true_positives = np.sum(true_positive_labels)
      cum_false_positives = np.sum(false_positive_labels)
      precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives)
      self.average_precision_per_class[class_index] = precision

    mean_ap = np.nanmean(self.average_precision_per_class)
    return ObjectClassifyEvalMetrics(
      self.average_precision_per_class, mean_ap, self.average_precision_per_class, 0)
