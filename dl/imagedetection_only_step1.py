import tensorflow as tf
import os
from PIL import Image
import numpy as np
import logging
import time
from dl.step1_cnn import Step1CNN
from dl.util import visualize_boxes_and_labels_on_image_array_V1
from tradition.edge.table_contour import TableContour

logger = logging.getLogger("detect")


class ImageDetectorFactory_os1:
    _detector = {}

    @staticmethod
    def get_static_detector(exportid):
        if exportid not in ImageDetectorFactory_os1._detector:
            ImageDetectorFactory_os1._detector[exportid] = ImageDetector_os1(exportid)
        return ImageDetectorFactory_os1._detector[exportid]

class ImageDetector_os1:
    def __init__(self, exportid):
        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step1_cnn = Step1CNN(os.path.join(file_path, 'model', str(exportid)))

        self.counter = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if not self.step1_cnn.is_load():
            self.step1_cnn.load(self.config)

    def detect(self, image_path, step1_min_score_thresh=.5, table_check=True):
        if not self.step1_cnn.is_load():
            self.load()
            if not self.step1_cnn.is_load():
                logger.warning('loading model failed')
                return None

        import time
        time0 = time.time()

        # image_path = image_instance.source.path
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image)
        # Actual detection.
        (boxes, scores) = self.step1_cnn.detect(image_np)

        # data solving
        boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes).astype(np.int32)
        scores_step1 = np.squeeze(scores)

        table_contour = None
        if table_check:
            table_contour = TableContour(image_path, debug_type=1)

        ret = []
        logger.info('detect number:{}'.format(len(boxes.shape[0])))
        for i in range(boxes.shape[0]):
            if scores_step1[i] > step1_min_score_thresh:
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)

                if table_contour is not None and not table_contour.check_box(xmin, ymin, xmax-xmin, ymax-ymin, i):
                    scores_step1[i] = 0.1
                else:
                    ret.append({'class': 1,
                                'score': scores_step1[i],
                                'score2': 0.0,
                                'upc': '',
                                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                                })

        # visualization
        output_image_path = None
        if len(ret) > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array_V1(
                image_np,
                boxes,
                scores_step1,
                use_normalized_coordinates=True,
                step1_min_score_thresh=step1_min_score_thresh,
                line_thickness=2,
                show_error_boxes=False,
                max_boxes_to_draw=None,
            )
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        time1 = time.time()
        return ret, time1-time0,output_image_path
