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
from goods.models import ProblemGoods, TimeLog, PreStep2TimeLog

logger = logging.getLogger("detect")


class ImageDetectorFactory_os2:
    _detector = {}

    @staticmethod
    def get_static_detector(exportid):
        if exportid not in ImageDetectorFactory_os2._detector:
            ImageDetectorFactory_os2._detector[exportid] = ImageDetector_os2(exportid)
        return ImageDetectorFactory_os2._detector[exportid]

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

class ImageDetector_os2:
    def __init__(self, export1id, export2id):
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
            logger.info('end loading step2 model...')
            # semaphore.release()

    def detect(self, image_instance):
        if self.labels_to_names is None:
            self.load()
            if self.labels_to_names is None:
                logger.warning('loading model failed')
                return None

        import time
        time0 = time.time()

        image_path = image_instance.source.path

        image_np = self.pre_sess_step2.run(self.output_image_tensor_step2, feed_dict={self.input_image_tensor_step2: image_path})
        image_np_expanded = np.expand_dims(image_np, axis=0)

        time1 = time.time()

        probabilities = self.session_step2.run(
            self.detection_classes, feed_dict={self.input_images_tensor_step2: image_np_expanded})

        time2 = time.time()

        index = 0
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities[index]), key=lambda x: x[1])]
        ret = []
        for i in range(5):
            class_type = sorted_inds[i]
            upc = self.labels_to_names[sorted_inds[i]]
            scores_step2 = probabilities[index][sorted_inds[i]]
            ret.append(
                {'class': class_type,
               'score': scores_step2,
               'upc': upc,
               })

        time3 = time.time()
        logger.info('detect_only_step2: %s, %.2f, %.2f, %.2f' %(image_instance.deviceid, time3-time0, time1-time0, time2-time1))
        return ret
