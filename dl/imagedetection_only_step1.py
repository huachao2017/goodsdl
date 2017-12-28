import tensorflow as tf
from tensorflow.contrib import slim
from nets import inception
from preprocessing import inception_preprocessing
import os
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from .step2 import dataset
from object_detection.utils import visualization_utils as vis_util
import logging
import time
from goods.models import ProblemGoods

logger = logging.getLogger("django")


class ImageDetectorFactory_os1:
    _detector = {}

    @staticmethod
    def get_static_detector(exportid):
        if exportid not in ImageDetectorFactory_os1._detector:
            ImageDetectorFactory_os1._detector[exportid] = ImageDetector_os1(exportid)
        return ImageDetectorFactory_os1._detector[exportid]


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              scores_step1,
                                              instance_masks=None,
                                              keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=20,
                                              step1_min_score_thresh=.5,
                                              line_thickness=4):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      scores_step1: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      step1_min_score_thresh: step1 minimum score threshold for a box to be visualized
      line_thickness: integer (default: 4) controlling line width of the boxes.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = vis_util.collections.defaultdict(list)
    box_to_color_map = vis_util.collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = vis_util.collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores_step1 is None or scores_step1[i] > step1_min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores_step1 is None:
                box_to_color_map[box] = 'black'
            else:
                display_str = '{}%'.format(int(100 * scores_step1[i]),)
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = 'DarkOrange'

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            vis_util.draw_mask_on_image_array(
                image,
                box_to_instance_masks_map[box],
                color=color
            )
        vis_util.draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            vis_util.draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates)

    return image

class ImageDetector_os1:
    def __init__(self, exportid):
        self.graph_step1 = None
        self.session_step1 = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.model_dir = os.path.join(self.file_path, 'model', str(exportid))
        self.step1_model_path = os.path.join(self.model_dir, 'frozen_inference_graph.pb')
        self.step1_label_path = os.path.join(self.model_dir, 'goods_label_map.pbtxt')
        self.counter = 0
        self.detection_scores = None

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if self.detection_scores is None:
            logger.info('begin loading step1 model: {}'.format(self.step1_model_path))
            self.graph_step1 = tf.Graph()
            with self.graph_step1.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.step1_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            logger.info('end loading step1 graph...')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
            self.session_step1 = tf.Session(graph=self.graph_step1, config=config)

            # Definite input and output Tensors for detection_graph
            self.image_tensor_step1 = self.graph_step1.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.graph_step1.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.graph_step1.get_tensor_by_name('detection_scores:0')
            # self.detection_classes = self.graph_step1.get_tensor_by_name('detection_classes:0')

            logger.info('end loading model...')
            # semaphore.release()

    def detect(self, image_instance, step1_min_score_thresh=.5):
        if self.detection_scores is None:
            self.load()
            if self.detection_scores is None:
                logger.warning('loading model failed')
                return None

        image_path = image_instance.source.path
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores) = self.session_step1.run(
            [self.detection_boxes, self.detection_scores],
            feed_dict={self.image_tensor_step1: image_np_expanded})

        # data solving
        boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes).astype(np.int32)
        scores_step1 = np.squeeze(scores)

        ret = []
        for i in range(boxes.shape[0]):
            if scores_step1[i] > step1_min_score_thresh:
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)
                ret.append({'class': 1,
                            'score': scores_step1[i],
                            'score2': 0.0,
                            'upc': '',
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                            })

        # visualization
        if len(ret) > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                scores_step1,
                use_normalized_coordinates=True,
                step1_min_score_thresh=step1_min_score_thresh,
                line_thickness=2)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        return ret
