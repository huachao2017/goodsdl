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
from edge.contour_detect import find_contour

logger = logging.getLogger("detect")


class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(export2id):
        key = export2id
        if key not in ImageDetectorFactory._detector:
            ImageDetectorFactory._detector[key] = ImageDetector(export2id)
        return ImageDetectorFactory._detector[key]


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
                                              step1_min_score_thresh=.5,
                                              step2_min_score_thresh=.5,
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
      step1_min_score_thresh: step1 minimum score threshold for a box to be visualized
      step2_min_score_thresh: step2 minimum score threshold for a box to be visualized
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
        if scores_step1 is None or scores_step1[i] > step1_min_score_thresh:
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
                    display_str = '{}'.format(class_name)
                    box_to_display_str_map[box].append(display_str)
                    display_str = '{}%, {}%'.format(
                        int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]),
                    )
                    box_to_display_str_map[box].append(display_str)
                else:
                    display_str = 'score: {}%, {}%'.format(int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]))
                    box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    if scores_step2[i] < step2_min_score_thresh:
                        box_to_color_map[box] = 'Red'
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


def get_step2_labels_to_names(labels_filepath):
    with tf.gfile.Open(labels_filepath, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_names[int(line[:index])] = line[index + 1:]

    return labels_to_names

class ImageDetector:
    def __init__(self, export2id):
        self.graph_step2 = None
        self.session_step2 = None
        self.labels_to_names = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.step2_model_dir = os.path.join(self.file_path, 'model', str(export2id))
        self.counter = 0

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if self.labels_to_names is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            step2_checkpoint = tf.train.latest_checkpoint(self.step2_model_dir)
            logger.info('begin loading step2 model: {}'.format(step2_checkpoint))
            step2_labels_to_names = get_step2_labels_to_names(os.path.join(self.step2_model_dir, 'labels.txt'))
            image_size = inception.inception_resnet_v2.default_image_size

            self.pre_graph_step2 = tf.Graph()
            with self.pre_graph_step2.as_default():
                image_path = tf.placeholder(dtype=tf.string, name='input_image')
                image_string = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
                image = tf.squeeze(image, [0])
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0, name='output_image')

            self.pre_sess_step2 = tf.Session(graph=self.pre_graph_step2, config=config)

            self.graph_step2 = tf.Graph()
            with self.graph_step2.as_default():
                images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3], name='input_tensor')

                # Create the model, use the default arg scope to configure the batch norm parameters.
                with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                    logits, _ = inception.inception_resnet_v2(images,
                                                              num_classes=len(step2_labels_to_names),
                                                              is_training=False)
                probabilities = tf.nn.softmax(logits, name='detection_classes')

                init_fn = slim.assign_from_checkpoint_fn(
                    step2_checkpoint,
                    slim.get_model_variables('InceptionResnetV2'))

                logger.info('end loading step2 graph...')

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.session_step2 = tf.Session(config=config)
                init_fn(self.session_step2)

            self.input_image_tensor_step2 = self.pre_graph_step2.get_tensor_by_name('input_image:0')
            self.output_image_tensor_step2 = self.pre_graph_step2.get_tensor_by_name('output_image:0')
            self.input_images_tensor_step2 = self.graph_step2.get_tensor_by_name('input_tensor:0')
            # self.np_image_step2 = self.graph_step2.get_tensor_by_name('image_tensor:0')
            self.detection_classes = self.graph_step2.get_tensor_by_name('detection_classes:0')

            # label_map = label_map_util.load_labelmap(self.step1_label_path)
            # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000,
            #                                                             use_display_name=True)
            self.labels_to_names = step2_labels_to_names
            logger.info('end loading model...')
            # semaphore.release()

    def detect(self, image_instance, step1_min_score_thresh=.5, step2_min_score_thresh=.5):
        if self.labels_to_names is None:
            self.load()
            if self.labels_to_names is None:
                logger.warning('loading model failed')
                return None

        # import time
        # time0 = time.time()

        image_path = image_instance.source.path
        image = Image.open(image_path)
        # FIXME 需要标定
        cv_image, boxes, scores_step1 = find_contour(image_path,area=(69,86,901,516))
        im_width = cv_image.shape[1]
        im_height = cv_image.shape[0]
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

        # cv 坐标需要处理成tf
        tmp = boxes[:, 0].copy()
        boxes[:, 0] = boxes[:, 1]
        boxes[:, 1] = tmp
        tmp = boxes[:, 2].copy()
        boxes[:, 2] = boxes[:, 3]
        boxes[:, 3] = tmp

        # if image_instance.deviceid == '275':
        #     time1 = time.time() - time0
        #     time0 = time.time()

        step2_images = []
        # sub_time_param = ''
        for i in range(boxes.shape[0]):
            if scores_step1[i] > step1_min_score_thresh:
                # sub_time0 = time.time()
                ymin, xmin, ymax, xmax = boxes[i]
                # ymin = int(ymin * im_height)
                # xmin = int(xmin * im_width)
                # ymax = int(ymax * im_height)
                # xmax = int(xmax * im_width)

                newimage = image.crop((xmin, ymin, xmax, ymax))
                # 生成新的图片
                newimage_split = os.path.split(image_path)
                new_image_path = os.path.join(newimage_split[0], "{}_{}".format(i, newimage_split[1]))
                newimage.save(new_image_path, 'JPEG')
                # if image_instance.deviceid == '275':
                #     sub_time_param = sub_time_param + '%.2f,' %(time.time()-sub_time0)
                #     sub_time0 = time.time()
                step2_images.append(self.pre_sess_step2.run(self.output_image_tensor_step2, feed_dict={self.input_image_tensor_step2: new_image_path}))
                # if image_instance.deviceid == '275':
                #     sub_time_param = sub_time_param + '%.2f,' %(time.time()-sub_time0)

        # if image_instance.deviceid == '275':
        #     time2 = time.time() - time0
        #     time0 = time.time()
        #     PreStep2TimeLog.objects.create(image_id=image_instance.pk,
        #                            param=sub_time_param,
        #                            total=time2)

        if len(step2_images) <= 0:
            return None
        # 统一识别，用于加速
        step2_images_nps = np.array(step2_images)
        probabilities = self.session_step2.run(
            self.detection_classes, feed_dict={self.input_images_tensor_step2: step2_images_nps})

        # if image_instance.deviceid == '275':
        #     time3 = time.time() - time0
        #     TimeLog.objects.create(image_id=image_instance.pk,
        #                            time1=time1,
        #                            time2=time2,
        #                            time3=time3,
        #                            total=time1+time2+time3)

        ret = []
        classes = []
        scores_step2 = []
        index = -1
        for i in range(boxes.shape[0]):
            classes.append(-1)
            scores_step2.append(-1)
            if scores_step1[i] > step1_min_score_thresh:
                index = index + 1
                ymin, xmin, ymax, xmax = boxes[i]
                # ymin = int(ymin * im_height)
                # xmin = int(xmin * im_width)
                # ymax = int(ymax * im_height)
                # xmax = int(xmax * im_width)
                # newimage = np.array(newimage, dtype=np.float32)
                # logger.info(newimage.shape)
                # probabilities = self.session_step2.run(
                #     self.detection_classes, feed_dict={self.image_tensor_step2: newimage})
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities[index]), key=lambda x: x[1])]

                # print(classes[i])
                # print(self.class_to_name_dic)
                classes[i] = sorted_inds[0]
                scores_step2[i] = probabilities[index][sorted_inds[0]]
                # for j in range(5):
                #     index = sorted_inds[j]
                #     logger.info('[%s] Probability %0.2f%% => [%s]' % (self.labels_to_names[index], probabilities[index] * 100, index))

                class_type = sorted_inds[0]
                upc = self.labels_to_names[sorted_inds[0]]
                if probabilities[index][sorted_inds[0]] < step2_min_score_thresh:
                    # 识别度不够
                    class_type = -1
                    upc = ''
                elif probabilities[index][sorted_inds[0]]-probabilities[index][sorted_inds[1]] < 0.1:
                    # 两个类型有混淆
                    class_type = -1
                    upc = ''

                if class_type == -1:
                    # add to database
                    ProblemGoods.objects.create(image_id=image_instance.pk,
                                                index=i,
                                                class_type_0=sorted_inds[0],
                                                class_type_1=sorted_inds[1],
                                                class_type_2=sorted_inds[2],
                                                class_type_3=sorted_inds[3],
                                                class_type_4=sorted_inds[4],
                                                score_0=probabilities[index][sorted_inds[0]],
                                                score_1=probabilities[index][sorted_inds[1]],
                                                score_2=probabilities[index][sorted_inds[2]],
                                                score_3=probabilities[index][sorted_inds[3]],
                                                score_4=probabilities[index][sorted_inds[4]],
                                                )
                ret.append({'class': class_type,
                            'score': scores_step1[i],
                            'score2': scores_step2[i],
                            'upc': upc,
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                            })

        # visualization
        if len(ret) > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                classes,
                scores_step1,
                scores_step2,
                self.labels_to_names,
                use_normalized_coordinates=False,
                step2_min_score_thresh=step2_min_score_thresh,
                line_thickness=2)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        return ret
