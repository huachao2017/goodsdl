import tensorflow as tf
import os
import numpy as np
import logging
import time
from arm.dl.cnn import CNN

logger = logging.getLogger("django")


class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(train_model):

        if train_model.pk not in ImageDetectorFactory._detector:
            ImageDetectorFactory._detector[train_model.pk] = ImageDetector(train_model.model_path)
        return ImageDetectorFactory._detector[train_model.pk]

class ImageDetector:
    def __init__(self, model_path):
        self.cnn = CNN(model_path)

        self.counter = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if not self.cnn.is_load():
            self.cnn.load(self.config)


    def detect(self, image_path):
        if not self.cnn.is_load():
            self.load()
            if not self.cnn.is_load():
                logger.warning('loading model failed')
                return None

        import time
        time0 = time.time()

        image_np = self.cnn.pre_detect(image_path)

        time1 = time.time()

        upcs, scores = self.cnn.detect(image_np)

        time2 = time.time()
        # logger.info('detect: %.2f, %.2f, %.2f' %(time2-time0, time1-time0, time2-time1))
        return upcs, scores
