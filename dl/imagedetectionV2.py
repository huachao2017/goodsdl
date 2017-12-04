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

logger = logging.getLogger("django")


class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(type):
        if type not in ImageDetectorFactory._detector:
            ImageDetectorFactory._detector[type] = ImageDetector(type)
        return ImageDetectorFactory._detector[type]


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              classes,
                                              scores_step1,
                                              scores_step2,
                                              labels_to_names,
                                              instance_masks=None,
                                              keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=20,
                                              min_score_thresh=.5,
                                              agnostic_mode=False,
                                              line_thickness=4):
    """Overlay labeled boxes on an image with formatted scores and label names.
  
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.
  
    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores_step1: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      labels_to_names: a dict containing category dictionaries (each holding
        category name) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
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
        if scores_step1 is None or scores_step1[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores_step1 is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in labels_to_names.keys():
                        class_name = labels_to_names[classes[i]]
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%, {}%'.format(
                        class_name,
                        int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]),
                    )
                else:
                    display_str = 'score: {}%, {}%'.format(int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = vis_util.STANDARD_COLORS[
                        classes[i] % len(vis_util.STANDARD_COLORS)]

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


class ImageDetector:
    def __init__(self, type):
        self.graph_step1 = None
        self.session_step1 = None
        self.graph_step2 = None
        self.session_step2 = None
        self.labels_to_names = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.checkpoints_dir = os.path.join(self.file_path, 'model', str(type))
        self.step1_model_path = os.path.join(self.checkpoints_dir, 'frozen_inference_graph.pb')
        self.step1_label_path = os.path.join(self.checkpoints_dir, 'goods_label_map.pbtxt')

    def load(self):
        if self.labels_to_names is None:
            logger.info('begin loading step1 model: {}'.format(self.step1_model_path))
            self.graph_step1 = tf.Graph()
            with self.graph_step1.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.step1_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

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

            step2_checkpoint = tf.train.latest_checkpoint(self.checkpoints_dir)
            logger.info('begin loading step2 model: {}'.format(step2_checkpoint))
            dataset_step2 = dataset.get_split('train', self.checkpoints_dir, )
            image_size = inception.inception_resnet_v2.default_image_size
            self.graph_step2 = tf.Graph()
            with self.graph_step2.as_default():
                image_path = tf.placeholder(dtype=tf.string, name='input_tensor')
                image_string = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
                processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                                                                           is_training=False)
                processed_images = tf.expand_dims(processed_image, 0)

                # Create the model, use the default arg scope to configure the batch norm parameters.
                with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                    logits, _ = inception.inception_resnet_v2(processed_images,
                                                              num_classes=len(dataset_step2.labels_to_names),
                                                              is_training=False)
                probabilities = tf.nn.softmax(logits, name='detection_classes')

                init_fn = slim.assign_from_checkpoint_fn(
                    step2_checkpoint,
                    slim.get_model_variables('InceptionResnetV2'))

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.session_step2 = tf.Session(config=config)
                init_fn(self.session_step2)

            self.image_tensor_step2 = self.graph_step2.get_tensor_by_name('input_tensor:0')
            self.np_image_step2 = self.graph_step2.get_tensor_by_name('image_tensor:0')
            self.detection_classes = self.graph_step2.get_tensor_by_name('detection_classes:0')

            # label_map = label_map_util.load_labelmap(self.step1_label_path)
            # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000,
            #                                                             use_display_name=True)
            self.labels_to_names = dataset_step2.labels_to_names
            logger.info('end loading model')
            # semaphore.release()

    def detect(self, image_path, min_score_thresh=.5):
        if self.labels_to_names is None:
            self.load()
            if self.labels_to_names is None:
                logger.warning('loading model failed')
                return None
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
        classes = []
        scores_step2 = []
        for i in range(boxes.shape[0]):
            classes.append(-1)
            scores_step2.append(-1)
            if scores_step1[i] > min_score_thresh:
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)

                newimage = image.crop((xmin, ymin, xmax, ymax))
                # 生成新的图片
                newimage_split = os.path.split(image_path)
                new_image_path = os.path.join(newimage_split[0], "{}_{}".format(i, newimage_split[1]))
                newimage.save(new_image_path, 'JPEG')

                probabilities = self.session_step2.run(
                    [self.detection_classes], feed_dict={self.image_tensor_step2: new_image_path})
                probabilities = probabilities[0]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

                # print(classes[i])
                # print(self.class_to_name_dic)
                ret.append({'class': sorted_inds[0],
                            'score': scores_step1[i],
                            'score2': probabilities[sorted_inds[0]],
                            'upc': self.labels_to_names[sorted_inds[0]],
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                            })
                classes[i] = sorted_inds[0]
                scores_step2[i] = probabilities[sorted_inds[0]]

        # visualization
        if boxes.shape[0] > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                classes,
                scores_step1,
                scores_step2,
                self.labels_to_names,
                use_normalized_coordinates=True,
                min_score_thresh=min_score_thresh,
                line_thickness=4)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        return ret
