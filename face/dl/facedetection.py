import tensorflow as tf
import numpy as np
import os
import logging
logger = logging.getLogger("face")
from django.conf import settings
from face.dl import facenet
from scipy import misc

class FaceDetectorFactory:
    _detector = None

    @staticmethod
    def get_static_detector():
        if FaceDetectorFactory._detector is None:
            FaceDetectorFactory._detector = FaceDetector()
        return FaceDetectorFactory._detector


class FaceDetector:
    def __init__(self):
        self.detection_graph = None
        self.session = None
        self.phase_train_placeholder = None
        self.image_size = 160

        self.model_path = os.path.join(settings.BASE_DIR, 'face', 'model', '20180402-114759.pb')
        self.counter = 0

    def load(self):
        if self.counter <= 0:
            self.counter = self.counter + 1
            if self.phase_train_placeholder is None:
                logger.info('begin loading face model: {}'.format(self.model_path))
                self.detection_graph = tf.Graph()
                with self.detection_graph.as_default():
                    od_graph_def = tf.GraphDef()
                    with tf.gfile.GFile(self.model_path, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
                self.session = tf.Session(graph=self.detection_graph, config=config)

                self.images_placeholder = self.detection_graph.get_tensor_by_name("input:0")
                self.embeddings = self.detection_graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.detection_graph.get_tensor_by_name("phase_train:0")

                logger.info('end loading face model')
         # semaphore.release()

    def detect(self, image_path):
        if self.phase_train_placeholder is None:
            self.load()
            if self.phase_train_placeholder is None:
                logger.warning('loading model failed')
                return None
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        aligned = misc.imresize(img, (self.image_size, self.image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        np_image = np.expand_dims(prewhitened, 0)
        feed_dict = {self.images_placeholder: np_image, self.phase_train_placeholder: False}
        embs = self.session.run(self.embeddings, feed_dict=feed_dict)
        index = embs[0]

        return index
