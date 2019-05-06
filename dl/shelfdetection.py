import tensorflow as tf
import os
from PIL import Image
import numpy as np
import logging
import time
from dl.step1_cnn import Step1CNN
from dl.util import visualize_boxes_and_labels_on_image_array_for_shelf
from dl.shelftradition_match import ShelfTraditionMatch

from sklearn.cluster import KMeans
import traceback
logger = logging.getLogger("django")


class ShelfDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(exportid,shopid):
        if exportid not in ShelfDetectorFactory._detector:
            ShelfDetectorFactory._detector[exportid] = ShelfDetector(exportid,shopid)
        return ShelfDetectorFactory._detector[exportid]

class ShelfDetector:
    def __init__(self, exportid, shopid):
        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step1_cnn = Step1CNN(os.path.join(file_path, 'model', str(exportid)))

        self.counter = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.tradition_match = ShelfTraditionMatch(shopid)


    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if not self.step1_cnn.is_load():
            self.step1_cnn.load(self.config)

        if not self.tradition_match.is_load():
            self.tradition_match.load()

    def detect(self, image_path, step1_min_score_thresh=.5, totol_level = 6):
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

        ret = []
        # logger.info('detect number:{}'.format(boxes.shape[0]))
        for i in range(boxes.shape[0]):
            if scores_step1[i] > step1_min_score_thresh:
                ymin, xmin, ymax, xmax = boxes[i]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)

                newimage = image.crop((xmin, ymin, xmax, ymax))

                # 生成新的图片 TODO 需要处理性能问题
                newimage_split = os.path.split(image_path)
                single_image_dir = os.path.join(newimage_split[0], 'single')
                if not tf.gfile.Exists(single_image_dir):
                    tf.gfile.MakeDirs(single_image_dir)
                new_image_path = os.path.join(single_image_dir, "{}_{}".format(i, newimage_split[1]))
                # newimage.save(new_image_path, 'JPEG')
                #
                # upc_match, score_match = self.tradition_match.detect_one_with_path(new_image_path)

                upc_match, score_match = self.tradition_match.detect_one_with_cv2array(new_image_path, newimage)
                if score_match < 0.5:
                    upc_match = ''
                    score_match = 0
                ret.append({'score': scores_step1[i],
                            'level': -1,
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                            'upc': upc_match,
                            'score2': score_match,
                            })

        if len(ret) > 0:
            self.caculate_level(ret,totol_level)

        # visualization
        output_image_path = None
        if len(ret) > 0:
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array_for_shelf(
                image_np,
                boxes,
                ret,
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

    def caculate_level(self,boxes,n_clusters = 6):
        """
        通过连续聚类计算框所属的层级
        """
        try:
            data = []
            for one_box in boxes:
                data.append((one_box['ymin'],one_box['ymax']))
            X = np.array(data)
            logger.info('calulate level: {},{}'.format(n_clusters,X.shape))
            estimator = KMeans(n_clusters=int(n_clusters))
            estimator.fit(X)
            label_pred = estimator.labels_  # 获取聚类标签
            label_to_mean = {}

            for i in range(n_clusters):
                one_X = X[label_pred == i]
                label_to_mean[i] = np.sum(one_X)/one_X.shape[0]

            # 根据平均值排序
            sorted_list = sorted(label_to_mean.items(),key=lambda item:item[1])
            t = np.array(sorted_list, dtype=int)
            t = t[:,0]
            sorted_label = {}
            for i in range(t.shape[0]):
                sorted_label[t[i]] = i

            for i in range(len(boxes)):
                box_label = label_pred[i]
                boxes[i]['level'] = sorted_label[box_label]
        except Exception as e:
            logger.error('caculate level error:{}'.format(e))
            traceback.print_exc()