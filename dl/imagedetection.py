import tensorflow as tf
import os
from PIL import Image
import numpy as np
from object_detection.utils import label_map_util

class ImageDetectorFactory:
    _detector = None

    @staticmethod
    def get_static_detector():
        if ImageDetectorFactory._detector is None:
            ImageDetectorFactory._detector = ImageDetector()
        return ImageDetectorFactory._detector


class ImageDetector:
    def __init__(self):
        self.detection_graph = None
        self.session = None
        self.model_path, _ = os.path.split(os.path.realpath(__file__))
        self.model_path = os.path.join(self.model_path, 'model/frozen_inference_graph.pb')
        self.label_path = os.path.join(self.model_path, 'model/goods_label_map.pbtxt')

    def load(self):
        #print('loading ImageDetector')
        # self.class_to_name_dic = {1:'香海鱼豆腐',
        #                           2:'亲嘴豆皮',
        #                           3:'多力多玉米片',
        #                           4:'焙朗早餐饼',
        #                           5:'康师傅妙芙（巧克力味）',
        #                           6:'法丽兹曲奇（抹茶慕斯）',
        #                           7:'百奇（巧克力味）',
        #                           8:'三元纯牛奶',
        #                           9:'雪碧（瓶装）',
        #                           10:'可口可乐（听装）',
        #                           }
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
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10000,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def detect(self,image_path,min_score_thresh=.5):
        if self.detection_graph is None:
            self.load()
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

        ret = []
        # have_classes = {}
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                # if classes[i] in have_classes:
                #     continue
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = int(ymin*im_height)
                xmin = int(xmin*im_width)
                ymax = int(ymax*im_height)
                xmax = int(xmax*im_width)
                #print(classes[i])
                #print(self.class_to_name_dic)
                ret.append({'class':classes[i],
                            'score':scores[i],
                            'upc':self.category_index[classes[i]]['name'],
                            'box':{'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}})
                # have_classes[classes[i]] = True
        return ret
