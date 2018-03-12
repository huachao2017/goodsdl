import tensorflow as tf
import os
import numpy as np
import logging
import time
from dl.step2_cnn import Step2CNN
from dl.step3_cnn import Step3CNN

logger = logging.getLogger("detect")
step2_model_names = ['inception_resnet_v2','nasnet_large']

class ImageDetectorFactory_os3:
    _detector = {}

    @staticmethod
    def get_static_detector(export2id, export3_arr, step2_model_name='nasnet_large'):
        if step2_model_name not in step2_model_names:
            return None
        # model_name : 'nasnet_large','inception_resnet_v2'

        if export2id not in ImageDetectorFactory_os3._detector:
            ImageDetectorFactory_os3._detector[export2id] = ImageDetectorFactory_os3(export2id, export3_arr, step2_model_name)
        return ImageDetectorFactory_os3._detector[export2id]

class ImageDetector_os3:
    def __init__(self, export2id, export3_arr, step2_model_name):
        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step2_cnn = Step2CNN(os.path.join(file_path, 'model', str(export2id)),step2_model_name)
        self.step3_cnn = Step3CNN(os.path.join(file_path, 'model'), export3_arr)

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

        step2_probabilities = self.step2_cnn.detect(image_np_expanded)

        time2 = time.time()

        index = 0
        step2_sorted_inds = [i[0] for i in sorted(enumerate(-step2_probabilities[index]), key=lambda x: x[1])]
        step2_class_type = step2_sorted_inds[0]
        step2_upc = self.step2_cnn.labels_to_names[step2_sorted_inds[0]]
        step2_score = step2_probabilities[index][step2_sorted_inds[0]]
        ret = {'step2_class': step2_class_type,
               'step2_score': step2_score,
               'step2_upc': step2_upc,
               'step3_class': -1,
               'step3_score': -1,
               'step3_upc': ''
               }
        if step2_upc in self.step2_cnn.cluster_upc_to_traintype:
            probabilities, labels_to_names = self.step3_cnn.detect(self.config,image_path,self.step2_cnn.cluster_upc_to_traintype[step2_upc])
            if labels_to_names is not None:
                sorted_inds = [j[0] for j in sorted(enumerate(-probabilities[0]), key=lambda x: x[1])]
                step3_class_type = sorted_inds[0]
                step3_score = probabilities[sorted_inds[0]]

                step3_upc = labels_to_names[step3_class_type]

                ret['step3_class'] = step3_class_type
                ret['step3_score'] = step3_score
                ret['step3_upc'] = step3_upc
                # TODO add step3_upc 判断action

        time3 = time.time()
        logger.info('detect_only_step3: %s, %.2f, %.2f, %.2f' %(image_instance.deviceid, time3-time0, time1-time0, time2-time1))
        return ret, time3-time0
