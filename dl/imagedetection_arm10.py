import tensorflow as tf
import os
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import math
import logging
logger = logging.getLogger("django")

class ImageDetectorFactory:
    _detector = None

    @staticmethod
    def get_static_detector():
        if ImageDetectorFactory._detector is None:
            ImageDetectorFactory._detector = ImageDetector(10)
        return ImageDetectorFactory._detector


class ImageDetector:
    def __init__(self, model_id):
        self.detection_graph = None
        self.session = None
        self.category_index = None
        self.file_path, _ = os.path.split(os.path.realpath(__file__))

        self.model_path = os.path.join(self.file_path, 'model', str(model_id), 'frozen_inference_graph.pb')
        self.label_path = os.path.join(self.file_path, 'model', str(model_id), 'goods_label_map.pbtxt')
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

    def detect(self,image_path, edge_boxes):
        if self.category_index is None:
            self.load()
            if self.category_index is None:
                logger.warning('loading model failed')
                return None
        import time
        time0 = time.time()
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
                min_score_thresh=0.5,
                line_thickness=4)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width), int(im_height)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        types = []
        # have_classes = {}
        for i in range(len(edge_boxes)):
            xmin1 = edge_boxes[i][0]
            ymin1 = edge_boxes[i][1]
            xmax1 = edge_boxes[i][2]
            ymax1 = edge_boxes[i][3]

            index = 0
            for j in range(boxes.shape[0]):
                ymin2, xmin2, ymax2, xmax2 = boxes[j]
                ymin2 = int(ymin2*im_height)
                xmin2 = int(xmin2*im_width)
                ymax2 = int(ymax2*im_height)
                xmax2 = int(xmax2*im_width)

                xmin=max(xmin1,xmin2)
                ymin=max(ymin1,ymin2)
                xmax=min(xmax1,xmax2)
                ymax=min(ymax1,ymax2)

                if xmin < xmax and ymin < ymax:
                    area = (xmax-xmin)*(ymax-ymin)
                    area1 = (xmax1-xmin1)*(ymax1-ymin1)
                    if area/area1 > 0.5:
                        index = j
                        break
                index += 1

            if index >= len(boxes.shape[0]):
                types.append(0)
            else:
                types.append(classes[index])
        time1 = time.time()
        logger.info('detect_all: %d, %.2f' %(len(types), time1-time0))
        return types
