# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
from goods.freezer.keras_yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from goods.freezer.keras_yolo3.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from set_config import config
from goods.freezer.keras_yolo3.yolo3 import yolo_freezer
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from goods.freezer.keras_yolo3.good_proxy import proxy
import logging
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

logger = logging.getLogger("detect")
gpu_num = config.yolov3_params['gpu_num']
label_path = config.yolov3_params['label_path']
min_score=config.yolov3_params['score']
(diff_switch,diff_iou) = config.yolov3_params['diff_switch_iou']
(single_switch,single_iou,single_min_score) = config.yolov3_params['single_switch_iou_minscore']
class YOLO(object):
    _defaults = {
        "model_path": config.yolov3_params['good_model_path'],
        "anchors_path": config.yolov3_params['anchors_path'],
        "classes_path": config.yolov3_params['classes_path'],
        "score" : config.yolov3_params['score'],
        "iou" : config.yolov3_params['iou'],
        "model_image_size" : config.yolov3_params['model_image_size'],
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        print(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        logger.info('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def predict_img(self,image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                #K.learning_phase(): 0
            })
        logger.info('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        p_class = []
        p_prob = []
        p_box = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            p_class.append(predicted_class)
            xmin = int(out_boxes[i][1]) if int(out_boxes[i][1]) > 0 else 0
            ymin = int(out_boxes[i][0]) if int(out_boxes[i][0]) > 0 else 0
            xmax = int(out_boxes[i][3]) if int(out_boxes[i][3]) > 0 else 0
            ymax = int(out_boxes[i][2]) if int(out_boxes[i][2]) > 0 else 0
            box = (xmin,ymin,xmax,ymax)
            p_box.append(box)
            score = out_scores[i]
            p_prob.append(score)
        return p_class,p_prob,p_box

    def close_session(self):
        self.sess.close()

def detect(yolov3,image_path):
    yolo_v3 = None
    if yolov3 is None:
        yolo_v3 = yolo_freezer.YOLO()
        logger.info("yolov3 freezer model reload")
    else:
        yolo_v3 = yolov3
    import time
    time0 = time.time()
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    (im_width, im_height) = image.size
    image_np = np.array(image).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    time1 = time.time()
    # Actual detection.
    (classes,scores,boxes) = yolo_v3.predict_img(image)
    if diff_switch:
        classes, scores, boxes = proxy.diff_fiter(diff_iou, classes,scores,boxes)

    if single_switch:
        classes, scores, boxes = proxy.single_filter(single_iou, single_min_score, classes,scores,boxes)

    # data solving
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    time2 = time.time()
    output_image_path = ''
    draw_boxes = []
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
        draw_boxes.append(
            [float(ymin) / im_height, float(xmin) / im_width, float(ymax) / im_height, float(xmax) / im_width])
    if boxes.shape[0] > 0:
        image_dir = os.path.dirname(image_path)
        output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(draw_boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            yolo_v3.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=None,
            min_score_thresh=min_score,
            line_thickness=4)
        output_image = Image.fromarray(image_np)
        output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
        output_image.save(output_image_path)

    ret = []
    # have_classes = {}
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
        ret.append({'class': classes[i],
                    'score': scores[i],
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                    })
    time3 = time.time()
    logger.info('detect_freezer: %d, %.2f, %.2f, %.2f, %.2f' % (
    len(ret), time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    return ret, time1 - time0, output_image_path


