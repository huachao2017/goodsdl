import tensorflow as tf
import os
from PIL import Image
import numpy as np
import logging
import time
from goods.models import ProblemGoods
from dl.step1_cnn import Step1CNN
from dl.step2_cnn import Step2CNN
from dl.step3_cnn import Step3CNN
from dl.util import visualize_boxes_and_labels_on_image_array_V2

logger = logging.getLogger("detect")

step2_model_names = ['inception_resnet_v2','nasnet_large']


class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(export1id,export2id, export3_arr=None, step2_model_name='nasnet_large'):
        if step2_model_name not in step2_model_names:
            return None
        # step2_model_name : 'nasnet_large','inception_resnet_v2'

        key = '{}_{}'.format(str(export1id),str(export2id))
        if key not in ImageDetectorFactory._detector:
            ImageDetectorFactory._detector[key] = ImageDetector(export1id,export2id,export3_arr,step2_model_name)
        return ImageDetectorFactory._detector[key]

class ImageDetector:
    def __init__(self, export1id, export2id, export3_arr, step2_model_name):
        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step1_cnn = Step1CNN(os.path.join(file_path, 'model', str(export1id)))
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
            self.step1_cnn.load(self.config)
            self.step2_cnn.load(self.config)

    def detect(self, image_instance, step1_min_score_thresh=.5, step2_min_score_thresh=.5):
        if not self.step2_cnn.is_load():
            self.load()
            if not self.step2_cnn.is_load():
                logger.warning('loading model failed')
                return None, .0

        import time
        time0 = time.time()

        image_path = image_instance.source.path
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.array(image)

        (boxes, scores) = self.step1_cnn.detect(image_np)

        # data solving
        boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        time1 = time.time()

        boxes[:,0] = boxes[:,0]*image.size[1]
        boxes[:,1] = boxes[:,1]*image.size[0]
        boxes[:,2] = boxes[:,2]*image.size[1]
        boxes[:,3] = boxes[:,3]*image.size[0]
        boxes = np.array(boxes,dtype=np.int)

        boxes_step1 = []
        scores_step1 = []
        step2_images = []
        step2_image_paths = []
        for i in range(boxes.shape[0]):
            if scores[i] > step1_min_score_thresh:
                # sub_time0 = time.time()
                ymin, xmin, ymax, xmax = boxes[i]
                boxes_step1.append([ymin, xmin, ymax, xmax])
                scores_step1.append(scores[i])

                newimage = image.crop((xmin, ymin, xmax, ymax))
                # 生成新的图片
                newimage_split = os.path.split(image_path)
                new_image_path = os.path.join(newimage_split[0], "{}_{}".format(i, newimage_split[1]))
                step2_image_paths.append(new_image_path)
                newimage.save(new_image_path, 'JPEG')
                step2_images.append(self.step2_cnn.pre_detect(new_image_path))

        if len(step2_images) <= 0:
            time2 = time.time()
            logger.info('detectV3: %s, 0, %.2f, %.2f, %.2f' % (image_instance.deviceid, time2 - time0, time1 - time0, time2 - time1))
            return [], time2-time0

        probabilities = self.step2_cnn.detect(step2_images)

        labels_step2 = []
        scores_step2 = []
        for i in range(len(boxes_step1)):
            type_to_probability = probabilities[i]
            sorted_inds = [j[0] for j in sorted(enumerate(-type_to_probability), key=lambda x: x[1])]

            labels_step2.append(sorted_inds[0])
            scores_step2.append(type_to_probability[sorted_inds[0]])
            if scores_step2[i] < step2_min_score_thresh:
                # add to database
                self.log_problem_goods(i, image_instance, type_to_probability, sorted_inds)

        time2 = time.time()

        ret = self.do_addition_logic_work(boxes_step1, scores_step1, labels_step2, scores_step2, step2_image_paths, image_instance, image_np, step2_min_score_thresh)

        time3 = time.time()
        logger.info('detectV3: %s, %d, %.2f, %.2f, %.2f, %.2f' %(image_instance.deviceid, len(ret), time3-time0, time1-time0, time2-time1, time3-time2))
        return ret, time3-time0

    def do_addition_logic_work(self, boxes_step1, scores_step1, labels_step2, scores_step2, step2_image_paths, image_instance, image_np, step2_min_score_thresh):
        ret = []
        for i in range(len(boxes_step1)):
            ymin, xmin, ymax, xmax = boxes_step1[i]

            class_type = labels_step2[i]
            action = 0
            upc = self.step2_cnn.labels_to_names[class_type]
            if scores_step2[i] < step2_min_score_thresh:
                # 识别度不够
                class_type = -1
                upc = ''

            if upc == 'bottled-drink-stand' or upc == 'ziptop-drink-stand':
                # 立姿水需要躺倒平放
                class_type = -1
                upc = ''
                action = 2
            elif upc in self.step2_cnn.cluster_upc_to_traintype:
                pass
                # TODO need test
                # probabilities, labels_to_names = self.step3_cnn.detect(self.config,step2_image_paths[i],self.step2_cnn.cluster_upc_to_traintype[upc])
                # if labels_to_names is not None:
                #     sorted_inds = [j[0] for j in sorted(enumerate(-probabilities[0]), key=lambda x: x[1])]
                #     step3_class_type = sorted_inds[0]
                #     step3_score = probabilities[sorted_inds[0]]
                #
                #     if step3_score > step2_min_score_thresh:
                #         upc = labels_to_names[step3_class_type]
                #         TODO add upc 判断action

            ret.append({'class': class_type,
                        'score': scores_step1[i],
                        'score2': scores_step2[i],
                        'action': action,
                        'upc': upc,
                        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                        })

        # visualization
        if len(ret) > 0:
            # TODO need add step3
            image_path = image_instance.source.path
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array_V2(
                image_np,
                np.array(boxes_step1),
                np.array(labels_step2),
                scores_step1,
                scores_step2,
                self.step2_cnn.labels_to_names,  # TODO need fix
                use_normalized_coordinates=False,
                step2_min_score_thresh=step2_min_score_thresh,
                line_thickness=2)
            output_image = Image.fromarray(image_np)
            output_image.save(output_image_path)
        return ret

    def log_problem_goods(self, i, image_instance, type_to_probability, sorted_inds):
        ProblemGoods.objects.create(image_id=image_instance.pk,
                                    index=i,
                                    class_type_0=sorted_inds[0],
                                    class_type_1=sorted_inds[1],
                                    class_type_2=sorted_inds[2],
                                    class_type_3=sorted_inds[3],
                                    class_type_4=sorted_inds[4],
                                    score_0=type_to_probability[sorted_inds[0]],
                                    score_1=type_to_probability[sorted_inds[1]],
                                    score_2=type_to_probability[sorted_inds[2]],
                                    score_3=type_to_probability[sorted_inds[3]],
                                    score_4=type_to_probability[sorted_inds[4]],
                                    )
