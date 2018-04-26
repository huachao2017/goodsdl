import tensorflow as tf
import os
import numpy as np
import logging
from dl.step2 import cluster
from dl.util import get_labels_to_names
from nets import nets_factory
from dl import common

logger = logging.getLogger("detect")

class Step2CNN:
    def __init__(self, step2_model_dir, step2_model_name):
        self.model_dir = step2_model_dir
        self.model_name = step2_model_name
        self._pre_graph = None
        self._pre_session = None
        self._graph = None
        self._session = None
        self.labels_to_names = None
        self._isload = False

    def load(self, config):
        checkpoint = tf.train.latest_checkpoint(self.model_dir)
        logger.info('begin loading step2 model: {}'.format(checkpoint))
        self.labels_to_names = get_labels_to_names(os.path.join(self.model_dir, 'labels.txt'))
        ####################
        # Select step2 model #
        ####################
        network_fn = nets_factory.get_network_fn(
            self.model_name,
            num_classes=len(self.labels_to_names),
            is_training=False)
        image_size = network_fn.default_image_size

        self._pre_graph = tf.Graph()
        with self._pre_graph.as_default():
            image_path = tf.placeholder(dtype=tf.string, name='input_image')
            image_string = tf.read_file(image_path)
            image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0, name='output_image')

        self._pre_session = tf.Session(graph=self._pre_graph, config=config)

        self._input_image_tensor = self._pre_graph.get_tensor_by_name('input_image:0')
        self._output_image_tensor = self._pre_graph.get_tensor_by_name('output_image:0')

        self._graph = tf.Graph()
        with self._graph.as_default():
            images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3], name='input_tensor')

            # Create the model, use the default arg scope to configure the batch norm parameters.
            logits, _ = network_fn(images)
            probabilities = tf.nn.softmax(logits, name='detection_classes')

            variables_to_restore = tf.global_variables()
            saver = tf.train.Saver(variables_to_restore)

            logger.info('end loading step2 graph...')

            self._session = tf.Session(config=config)
            saver.restore(self._session, checkpoint)

        self._input_images_tensor = self._graph.get_tensor_by_name('input_tensor:0')
        self._detection_classes = self._graph.get_tensor_by_name('detection_classes:0')


        cluster_setting = cluster.ClusterSettings(os.path.join(self.model_dir, common.CLUSTER_FILE_NAME))
        self.cluster_upc_to_traintype = cluster_setting.get_main_class_name_to_traintype()

        logger.info('end loading model...')
        self._isload = True

    def is_load(self):
        return self._isload

    def pre_detect(self, image_path):
        return self._pre_session.run(self._output_image_tensor,
                                     feed_dict={self._input_image_tensor: image_path})

    def detect(self, images):
        # 统一识别，用于加速
        images_nps = np.array(images)
        probabilities = self._session.run(
            self._detection_classes, feed_dict={self._input_images_tensor: images_nps})
        upcs = []
        scores = []
        logger.info(self.labels_to_names)
        for i in range(len(probabilities)):
            type_to_probability = probabilities[i]
            sorted_inds = [j[0] for j in sorted(enumerate(-type_to_probability), key=lambda x: x[1])]

            logger.info(sorted_inds[0])
            upcs.append(self.labels_to_names(sorted_inds[0]))
            scores.append(type_to_probability[sorted_inds[0]])
        return upcs, scores
