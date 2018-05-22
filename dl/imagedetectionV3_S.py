import tensorflow as tf
import os
from PIL import Image
import numpy as np
import logging
import time
from goods.models import ProblemGoods
from dl.step1_cnn import Step1CNN
from dl.step2S_cnn import Step2SCNN
from dl.tradition_match import TraditionMatch
from dl.util import visualize_boxes_and_labels_on_image_array_V2
from dl import common
from goods.models import ExportAction

logger = logging.getLogger("detect")

step2_model_names = ['inception_resnet_v2','nasnet_large']


class ImageDetectorFactory:
    _detector = {}

    @staticmethod
    def get_static_detector(deviceid):

        key = deviceid
        if key not in ImageDetectorFactory._detector:
            export1s = ExportAction.objects.filter(train_action__action='T1').filter(
                checkpoint_prefix__gt=0).order_by(
                '-update_time')[:1]
            export2Ss = ExportAction.objects.filter(train_action__action='T2S').filter(
                train_action__serial='').filter(checkpoint_prefix__gt=0).order_by(
                '-update_time')[:1]

            if len(export1s) == 0:
                logger.error('not found detection model!')
                return None
            else:
                if len(export2Ss) == 0:
                    export2Sid = 0
                    step2_model_name = ''
                else:
                    export2Sid = export2Ss[0].pk
                    step2_model_name = export2Ss[0].model_name

            ImageDetectorFactory._detector[key] = ImageDetector(deviceid, export1s[0].pk, export2Sid, step2_model_name)
        return ImageDetectorFactory._detector[key]

class ImageDetector:
    def __init__(self, deviceid, export1id, export2Sid, step2_model_name):
        self.deviceid = deviceid

        file_path, _ = os.path.split(os.path.realpath(__file__))
        self.step1_cnn = Step1CNN(os.path.join(file_path, 'model', str(export1id)))
        if export2Sid > 0:
            self.step2S_cnn = Step2SCNN(os.path.join(file_path, 'model', str(export2Sid)), step2_model_name)
        else:
            self.step2S_cnn = None
        self.tradition_match = TraditionMatch(deviceid, step=common.STEP2S_PREFIX)

        self.counter = 0
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True


    def load(self):
        if self.counter > 0:
            logger.info('waiting model to load (3s) ...')
            time.sleep(3)
            return
        self.counter = self.counter + 1
        if not self.tradition_match.is_load():
            self.step1_cnn.load(self.config)
            if self.step2S_cnn is not None:
                self.step2S_cnn.load(self.config)
            self.tradition_match.load()
            self.counter = 0

    def add_baseline_image(self, image_path, upc):
        if not self.tradition_match.is_load():
            self.load()
        self.tradition_match.add_baseline_image(image_path, upc)


    def removeall_baseline_image(self):
        if not self.tradition_match.is_load():
            self.tradition_match.removeall_baseline_image()

    def detect(self, image_instance, step1_min_score_thresh=.5, step2_min_score_thresh=.5):
        if not self.tradition_match.is_load():
            self.load()
            if not self.tradition_match.is_load():
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
                single_image_dir = os.path.join(newimage_split[0], 'single')
                if not tf.gfile.Exists(single_image_dir):
                    tf.gfile.MakeDirs(single_image_dir)
                new_image_path = os.path.join(single_image_dir, "{}_{}".format(i, newimage_split[1]))
                step2_image_paths.append(new_image_path)
                newimage.save(new_image_path, 'JPEG')
                if self.step2S_cnn is not None:
                    step2_images.append(self.step2S_cnn.pre_detect(new_image_path))

        if len(step2_image_paths) <= 0:
            time2 = time.time()
            logger.info('detectV3_S: %s, 0, %.2f, %.2f, %.2f' % (image_instance.deviceid, time2 - time0, time1 - time0, time2 - time1))
            return [], time2-time0

        # upcs_match, scores_match = self.tradition_match.detect(step2_image_paths)
        # time2 = time.time()

        if self.step2S_cnn is not None:
            upcs_step2, scores_step2  = self.step2S_cnn.detect(step2_images)

            time2 = time.time()

            types_step2 = []
            for i in range(len(step2_image_paths)):
                within_upcs = [upcs_step2[i]]
                score_verify = self.tradition_match.verify_score(step2_image_paths[i], within_upcs)
                logger.info('step2 tridition verify: %s, %.2f, %.2f' % (upcs_step2[i], score_verify,scores_step2[i]))
                if score_verify > 0.5:
                    types_step2.append(common.MATCH_TYPE_BOTH)
                else:
                    time3_0 = time.time()
                    upc_match, score_match = self.tradition_match.detect_one(step2_image_paths[i])
                    time3_1 = time.time()
                    logger.info('step2 tridition match: %.2f,%s, %.2f' % (time3_1-time3_0, upc_match, score_match))
                    if score_match > 0.6: # TODO
                        upcs_step2[i] = upc_match
                        scores_step2[i] = score_match
                        types_step2.append(common.MATCH_TYPE_TRADITION)
                    elif score_verify > 0.3: # TODO
                        types_step2.append(common.MATCH_TYPE_DEEPLEARNING)
                    else:
                        # 瓶装水类放弃传统识别
                        if upcs_step2[i] in ['6921168509256','6954767425979','6954767434674'] or upcs_step2[i] == 'ziptop-drink-stand':
                            types_step2.append(common.MATCH_TYPE_DEEPLEARNING)
                        else:
                            types_step2.append(common.MATCH_TYPE_UNKNOWN) # TODO 暂时做悲观处理
        else:
            upcs_step2, scores_step2  = self.tradition_match.detect(step2_image_paths)
            time2 = time.time()

            types_step2 = []
            for i in range(len(step2_image_paths)):
                if scores_step2[i] > 0.6:
                    types_step2.append(common.MATCH_TYPE_TRADITION)
                else:
                    types_step2.append(common.MATCH_TYPE_UNKNOWN)

        ret = self.do_addition_logic_work(boxes_step1, scores_step1, upcs_step2, scores_step2, types_step2, image_instance, image_np, step2_min_score_thresh)

        time3 = time.time()
        logger.info('detectV3_S: %s, %d, %.2f, %.2f, %.2f, %.2f' %(image_instance.deviceid, len(ret), time3-time0, time1-time0, time2-time1, time3-time2))
        return ret, time3-time0

    def do_addition_logic_work(self, boxes_step1, scores_step1, upcs_step2, scores_step2, match_types_step2, image_instance, image_np, step2_min_score_thresh):
        ret = []
        for i in range(len(boxes_step1)):
            ymin, xmin, ymax, xmax = boxes_step1[i]

            score2 = scores_step2[i]
            action = 0
            upc = upcs_step2[i]
            match_type = match_types_step2[i]
            if match_type == common.MATCH_TYPE_UNKNOWN:
                # 识别度不够
                upc = ''
            elif match_type == common.MATCH_TYPE_DEEPLEARNING and (upc == 'bottled-drink-stand' or upc == 'ziptop-drink-stand'):
                # 立姿水需要躺倒平放
                upc = ''
                action = 2

            ret.append({'class': match_type,
                        'score': scores_step1[i],
                        'score2': score2,
                        'action': action,
                        'upc': upc,
                        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                        })

        # visualization
        if len(ret) > 0:
            image_path = image_instance.source.path
            image_dir = os.path.dirname(image_path)
            output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
            visualize_boxes_and_labels_on_image_array_V2(
                image_np,
                np.array(boxes_step1),
                np.array(upcs_step2),
                scores_step1,
                scores_step2,
                use_normalized_coordinates=False,
                step2_min_score_thresh=step2_min_score_thresh,
                line_thickness=2)
            output_image = Image.fromarray(image_np)
            output_image.save(output_image_path)
        return ret