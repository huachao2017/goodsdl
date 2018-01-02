import tensorflow as tf
from tensorflow.contrib import slim
from nets import inception
from preprocessing import inception_preprocessing
import os
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from .step2 import dataset
import logging
import time
from goods.models import ProblemGoods

logger = logging.getLogger("classify")


class ImageClassifyFactory:
    _detector = None

    @staticmethod
    def get_static_detector(type):
        if not ImageClassifyFactory._detector:
            ImageClassifyFactory._detector = ImageClassify(type)
        return ImageClassifyFactory._detector

def get_labels_to_names(labels_filepath):
    with tf.gfile.Open(labels_filepath, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_names[int(line[:index])] = line[index + 1:]

    return labels_to_names

class ImageClassify:
    def __init__(self, type):
        self.graph_step2 = None
        self.session_step2 = None
        self.labels_to_names = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.checkpoints_dir = os.path.join(self.file_path, 'model', str(type))
        self.counter = 0

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if self.labels_to_names is None:
            step2_checkpoint = tf.train.latest_checkpoint(self.checkpoints_dir)
            logger.info('begin loading step2 model: {}'.format(step2_checkpoint))
            image_size = inception.inception_resnet_v2.default_image_size
            labels_to_names = get_labels_to_names(os.path.join(self.checkpoints_dir, 'labels.txt'))
            self.graph_step2 = tf.Graph()
            with self.graph_step2.as_default():
                image_path = tf.placeholder(dtype=tf.string, name='input_tensor')
                image_string = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
                processed_image = inception_preprocessing.preprocess_for_eval(image, image_size, image_size, central_fraction=None)
                processed_images = tf.expand_dims(processed_image, 0)

                # Create the model, use the default arg scope to configure the batch norm parameters.
                with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                    logits, _ = inception.inception_resnet_v2(processed_images,
                                                              num_classes=len(labels_to_names),
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

            self.image_tensor_step2 = self.graph_step2.get_tensor_by_name('input_tensor:0')
            self.np_image_step2 = self.graph_step2.get_tensor_by_name('image_tensor:0')
            self.detection_classes = self.graph_step2.get_tensor_by_name('detection_classes:0')

            # label_map = label_map_util.load_labelmap(self.step1_label_path)
            # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000,
            #                                                             use_display_name=True)
            self.labels_to_names = labels_to_names
            logger.info('end loading model...')
            # semaphore.release()

    def detect(self, image_path, min_score_thresh=.5):
        if self.labels_to_names is None:
            self.load()
            if self.labels_to_names is None:
                logger.warning('loading model failed')
                return None

        if not os.path.isfile(image_path):
            logger.warning('file not exist')
            return None

        ret = []
        probabilities = self.session_step2.run(
            self.detection_classes, feed_dict={self.image_tensor_step2: image_path})
        probabilities = probabilities[0]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        for i in range(5):
            class_type = sorted_inds[i]
            upc = self.labels_to_names[sorted_inds[i]]
            if probabilities[sorted_inds[i]] > min_score_thresh:
                ret.append({'class': class_type,
                        'score': probabilities[sorted_inds[i]],
                        'upc': upc,
                        })

        return ret
