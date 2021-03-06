import tensorflow as tf
import os
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import threading
from dl import common
from goods.models import ExportAction
import logging
logger = logging.getLogger("detect")

class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(deviceid):
        if deviceid not in ImageDetectorFactory._detector:
            if deviceid == '10':
                # FIXME hack for first train
                ImageDetectorFactory._detector[deviceid] = ImageDetector(10)
            else:
                exportalls = ExportAction.objects.filter(train_action__action='TA').filter(
                    checkpoint_prefix__gt=0).order_by(
                    '-update_time')[:1]
                if len(exportalls) == 0:
                    logger.error('not found detection model!')
                    return None
                ImageDetectorFactory._detector[deviceid] = ImageDetector(exportalls[0].pk)
        return ImageDetectorFactory._detector[deviceid]


class ImageDetector:
    def __init__(self, exportallid):
        self.detection_graph = None
        self.session = None
        self.category_index = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.model_path = os.path.join(self.file_path, 'model', str(exportallid), 'frozen_inference_graph.pb')
        self.label_path = os.path.join(self.file_path, 'model', str(exportallid), 'goods_label_map.pbtxt')
        self.counter = 0

    def load(self):
        if self.counter <= 0:
            self.counter = self.counter + 1
            if self.category_index is None:
                logger.info('begin loading old model: {}'.format(self.model_path))
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

                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

                # num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                label_map = label_map_util.load_labelmap(self.label_path)
                categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000,
                                                                            use_display_name=True)
                self.category_index = label_map_util.create_category_index(categories)
                logger.info('end loading old model')
         # semaphore.release()

    def detect(self,image_instance,step1_min_score_thresh=.5,step2_min_score_thresh=.5):
        if self.category_index is None:
            self.load()
            if self.category_index is None:
                logger.warning('loading model failed')
                return None
        import time
        time0 = time.time()
        image_path = image_instance.source.path
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_np_expanded})

        # data solving
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        if boxes.shape[0] > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                min_score_thresh=step1_min_score_thresh,
                line_thickness=4)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        ret = []
        # have_classes = {}
        for i in range(boxes.shape[0]):
            action = 0
            if scores is not None and scores[i] < step1_min_score_thresh:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(ymin*im_height)
            xmin = int(xmin*im_width)
            ymax = int(ymax*im_height)
            xmax = int(xmax*im_width)
            ret.append({'class':common.MATCH_TYPE_DEEPLEARNING,
                        'score':scores[i],
                        'action':action,
                        'upc':self.category_index[classes[i]]['name'],
                        'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax
                        })
        time1 = time.time()
        logger.info('detect_all: %s, %d, %.2f' %(image_instance.deviceid, len(ret), time1-time0))
        return ret, time1-time0
