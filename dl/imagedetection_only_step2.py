import tensorflow as tf
import os
import numpy as np
import logging
import time
from dl.step2_cnn import Step2CNN

logger = logging.getLogger("detect")
step2_model_names = ['inception_resnet_v2','nasnet_large']


class ImageDetectorFactory_os2:
    _detector = {}

    @staticmethod
    def get_static_detector(exportid, model_name='nasnet_large'):
        if model_name not in step2_model_names:
            return None
        # model_name : 'nasnet_large','inception_resnet_v2'

        if exportid not in ImageDetectorFactory_os2._detector:
            ImageDetectorFactory_os2._detector[exportid] = ImageDetector_os2(exportid, model_name)
        return ImageDetectorFactory_os2._detector[exportid]

class ImageDetector_os2:
    def __init__(self, export2id, model_name):
        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step2_cnn = Step2CNN(os.path.join(file_path, 'model', str(export2id)),model_name)

        self.counter = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if not self.step2_cnn.is_load():
            self.step2_cnn.load(self.config)


    def detect(self, image_instance):
        if not self.step2_cnn.is_load():
            self.load()
            if not self.step2_cnn.is_load():
                logger.warning('loading model failed')
                return None

        import time
        time0 = time.time()

        image_path = image_instance.source.path

        image_np = self.step2_cnn.pre_detect(image_path)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        time1 = time.time()

        probabilities = self.step2_cnn.detect(image_np_expanded)

        time2 = time.time()

        index = 0
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities[index]), key=lambda x: x[1])]
        ret = []
        for i in range(5):
            class_type = sorted_inds[i]
            upc = self.step2_cnn.labels_to_names[sorted_inds[i]]
            scores_step2 = probabilities[index][sorted_inds[i]]
            ret.append(
                {'class': class_type,
               'score': scores_step2,
               'upc': upc,
               })

        time3 = time.time()
        logger.info('detect_only_step2: %s, %.2f, %.2f, %.2f' %(image_instance.deviceid, time3-time0, time1-time0, time2-time1))
        return ret, time3-time0
