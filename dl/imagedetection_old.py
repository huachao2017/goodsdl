import tensorflow as tf
import os
from PIL import Image
import numpy as np
from object_detection.utils import visualization_utils as vis_util

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
        self.model_path = os.path.join(self.model_path, 'model/frozen_inference_graph_old.pb')

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
        self.category_index = {1: {'id':1, 'name':1, 're_name':'香海鱼豆腐','price':6.8},
                               2: {'id':2, 'name':2, 're_name': '亲嘴豆皮', 'price': 5.8},
                               3: {'id':3, 'name':3, 're_name': '多力多玉米片', 'price': 6.6},
                               4: {'id':4, 'name':4, 're_name': '焙朗早餐饼', 'price': 14.6},
                               5: {'id':5, 'name':5, 're_name': '康师傅妙芙（巧克力味）', 'price': 16.8},
                               6: {'id':6, 'name':6, 're_name': '法丽兹曲奇（抹茶慕斯）', 'price': 10.6},
                               7: {'id':7, 'name':7, 're_name': '百奇（巧克力味）', 'price': 12.8},
                               8: {'id':8, 'name':8, 're_name': '三元纯牛奶', 'price': 1.8},
                               9: {'id':9, 'name':9, 're_name': '雪碧（瓶装）', 'price': 3.0},
                               10: {'id':10, 'name':10, 're_name': '可口可乐（听装）', 'price': 2.5},
                               }
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 占用GPU20%的显存
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
                min_score_thresh=.1,
                line_thickness=4)
            output_image = Image.fromarray(image_np)
            output_image.thumbnail((int(im_width*0.5), int(im_height*0.5)), Image.ANTIALIAS)
            output_image.save(output_image_path)

        ret = []
        # have_classes = {}
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                # FIXME 不允许有同类型的物品出现
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
                            'name':self.category_index[classes[i]]['re_name'],
                            'price':self.category_index[classes[i]]['price'],
                            'box':{'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}})
                # have_classes[classes[i]] = True
        return ret
