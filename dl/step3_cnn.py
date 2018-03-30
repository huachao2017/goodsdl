import tensorflow as tf
import os
import numpy as np
from dl.util import get_labels_to_names
import logging
from nets import nets_factory

logger = logging.getLogger("detect")

class Step3CNN:
    def __init__(self, model_dir,export3_arr):
        self.model_dir = model_dir
        self.export3_arr = sorted(export3_arr, key=lambda x:(x.update_time,), reverse=True)

        self.labels_to_names_dic = {}
        self._session_dic = {}
        self._input_image_tensor_dic = {}
        self._detection_classes_dic = {}

        self._load_dic = {}


    def load(self, config, traintype,export3):
        labels_to_names = None
        session = None
        input_image_tensor = None
        detection_classes = None
        if traintype not in self._session_dic:
            if traintype in self._load_dic and self._load_dic[traintype]:
                return None,None,None,None
            self._load_dic[traintype] = True

            traintype_modeldir = os.path.join(self.model_dir, str(export3.pk))
            checkpoint = tf.train.latest_checkpoint(traintype_modeldir)
            logger.info('begin loading step3 model: {}-{}'.format(traintype, checkpoint))
            labels_to_names = get_labels_to_names(os.path.join(traintype_modeldir, 'labels.txt'))

            network_fn = nets_factory.get_network_fn(
                export3.model_name,
                num_classes=len(labels_to_names),
                is_training=False)
            image_size = network_fn.default_image_size

            _graph = tf.Graph()
            with _graph.as_default():
                input_image_path = tf.placeholder(dtype=tf.string, name='input_image')
                image_string = tf.read_file(input_image_path)
                image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.expand_dims(image, 0)
                images = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
                images = tf.subtract(images, 0.5)
                images = tf.multiply(images, 2.0)
                logits, _ = network_fn(images)
                probabilities = tf.nn.softmax(logits, name='detection_classes')
                variables_to_restore = tf.global_variables()
                saver = tf.train.Saver(variables_to_restore)
                session = tf.Session(config=config)
                saver.restore(session, checkpoint)
                logger.info('end loading step3 graph...')

            input_image_tensor = _graph.get_tensor_by_name('input_image:0')
            detection_classes = _graph.get_tensor_by_name('detection_classes:0')

            self.labels_to_names_dic[traintype] = labels_to_names
            self._session_dic[traintype] = session
            self._input_image_tensor_dic[traintype] = input_image_tensor
            self._detection_classes_dic[traintype] = detection_classes
            self._load_dic[traintype] = False
        else:
            labels_to_names = self.labels_to_names_dic[traintype]
            session = self._session_dic[traintype]
            input_image_tensor = self._input_image_tensor_dic[traintype]
            detection_classes = self._detection_classes_dic[traintype]

        return labels_to_names, session, input_image_tensor, detection_classes

    def detect(self, config, image_path, traintype):
        cur_export3 = None
        for export3 in self.export3_arr:
            if export3.train_action.traintype == traintype:
                cur_export3 = export3
                break

        if cur_export3 is None:
            return None,None

        labels_to_names, session, input_image_tensor, detection_classes = self.load(config,traintype,cur_export3)

        if session is None:
            return None,None

        probabilities = session.run(
            detection_classes, feed_dict={input_image_tensor: image_path})
        return probabilities, labels_to_names

