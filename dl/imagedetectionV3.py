import tensorflow as tf
import os
from PIL import Image
import numpy as np
from object_detection.utils import visualization_utils as vis_util
import logging
import time
from goods.models import ProblemGoods
from nets import nets_factory
from dl.step2 import cluster

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


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              classes,
                                              scores_step1,
                                              scores_step2,
                                              labels_to_names,
                                              instance_masks=None,
                                              keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=20,
                                              step1_min_score_thresh=.0,
                                              step2_min_score_thresh=.5,
                                              agnostic_mode=False,
                                              line_thickness=4):
    """Overlay labeled boxes on an image with formatted scores and label names.
  
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.
  
    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores_step1: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      labels_to_names: a dict containing category dictionaries (each holding
        category name) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      step1_min_score_thresh: step1 minimum score threshold for a box to be visualized
      step2_min_score_thresh: step2 minimum score a bounding box's color
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
  
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = vis_util.collections.defaultdict(list)
    box_to_color_map = vis_util.collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = vis_util.collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores_step1 is None or scores_step1[i] > step1_min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores_step1 is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in labels_to_names.keys():
                        class_name = labels_to_names[classes[i]]
                    else:
                        class_name = 'N/A'
                    display_str = '{}'.format(class_name)
                    box_to_display_str_map[box].append(display_str)
                    display_str = '{}%, {}%'.format(
                        int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]),
                    )
                    box_to_display_str_map[box].append(display_str)
                else:
                    display_str = 'score: {}%, {}%'.format(int(100 * scores_step1[i]),
                        int(100 * scores_step2[i]))
                    box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    if scores_step2[i] < step2_min_score_thresh:
                        box_to_color_map[box] = 'Red'
                    else:
                        box_to_color_map[box] = vis_util.STANDARD_COLORS[
                            classes[i] % len(vis_util.STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            vis_util.draw_mask_on_image_array(
                image,
                box_to_instance_masks_map[box],
                color=color
            )
        vis_util.draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            vis_util.draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates)

    return image


def get_labels_to_names(labels_filepath):
    with tf.gfile.Open(labels_filepath, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_names[int(line[:index])] = line[index + 1:]

    return labels_to_names

class Step1CNN:
    def __init__(self, step1_model_dir):
        self.model_dir = step1_model_dir
        self.model_path = os.path.join(self.model_dir, 'frozen_inference_graph.pb')
        self._graph = None
        self._session = None

    def load(self, config):
        logger.info('begin loading step1 model: {}'.format(self.model_path))
        self._graph = tf.Graph()
        with self._graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        logger.info('end loading step1 graph...')

        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
        self._session = tf.Session(graph=self._graph, config=config)

        # Definite input and output Tensors for detection_graph
        self._image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self._detection_boxes = self._graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self._detection_scores = self._graph.get_tensor_by_name('detection_scores:0')

    def detect(self,image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores) = self._session.run(
            [self._detection_boxes, self._detection_scores],
            feed_dict={self._image_tensor: image_np_expanded})

        return (boxes, scores)


class Step2CNN:
    def __init__(self, step2_model_dir, step2_model_name):
        self.model_dir = step2_model_dir
        self.model_name = step2_model_name
        self._pre_graph = None
        self._pre_session = None
        self._graph = None
        self._session = None
        self.labels_to_names = None
        self._isload = False

    def load(self, config):
        checkpoint = tf.train.latest_checkpoint(self.model_dir)
        logger.info('begin loading step2 model: {}'.format(checkpoint))
        self.labels_to_names = get_labels_to_names(os.path.join(self.model_dir, 'labels.txt'))
        ####################
        # Select step2 model #
        ####################
        network_fn = nets_factory.get_network_fn(
            self.model_name,
            num_classes=len(self.labels_to_names),
            is_training=False)
        image_size = network_fn.default_image_size

        self._pre_graph = tf.Graph()
        with self._pre_graph.as_default():
            image_path = tf.placeholder(dtype=tf.string, name='input_image')
            image_string = tf.read_file(image_path)
            image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0, name='output_image')

        self._pre_session = tf.Session(graph=self._pre_graph, config=config)

        self._input_image_tensor = self._pre_graph.get_tensor_by_name('input_image:0')
        self._output_image_tensor = self._pre_graph.get_tensor_by_name('output_image:0')

        self._graph = tf.Graph()
        with self._graph.as_default():
            images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3], name='input_tensor')

            # Create the model, use the default arg scope to configure the batch norm parameters.
            logits, _ = network_fn(images)
            probabilities = tf.nn.softmax(logits, name='detection_classes')

            variables_to_restore = tf.global_variables()
            saver = tf.train.Saver(variables_to_restore)

            logger.info('end loading step2 graph...')

            self._session = tf.Session(config=config)
            saver.restore(self._session, checkpoint)

        self._input_images_tensor = self._graph.get_tensor_by_name('input_tensor:0')
        self._detection_classes = self._graph.get_tensor_by_name('detection_classes:0')


        cluster_setting = cluster.ClusterSettings(os.path.join(self.model_dir, 'cluster.txt'))
        self.cluster_upc_to_traintype = cluster_setting.get_main_class_name_to_traintype()

        logger.info('end loading model...')
        self._isload = True

    def is_load(self):
        return self._isload

    def pre_detect(self, image_path):
        return self._pre_session.run(self._output_image_tensor,
                                     feed_dict={self._input_image_tensor: image_path})

    def detect(self, images):
        # 统一识别，用于加速
        images_nps = np.array(images)
        probabilities = self._session.run(
            self._detection_classes, feed_dict={self._input_images_tensor: images_nps})
        return probabilities

class Step3CNN:
    def __init__(self, model_dir,export3_arr):
        self.model_dir = model_dir
        self.export3_arr = sorted(export3_arr, key=lambda x:(x.update_time,), reverse=True)

        self.labels_to_names_dic = {}
        self._session_dic = {}
        self._input_image_tensor_dic = {}
        self._detection_classes_dic = {}

        self._load_dic = {}


    def load(self, config, traintype,export3):
        labels_to_names = None
        session = None
        input_image_tensor = None
        detection_classes = None
        if traintype not in self._session_dic:
            if traintype in self._load_dic and self._load_dic[traintype]:
                return None,None,None,None
            self._load_dic[traintype] = True

            traintype_modeldir = os.path.join(self.model_dir, str(export3.pk))
            checkpoint = tf.train.latest_checkpoint(traintype_modeldir)
            logger.info('begin loading step3 model: {}-{}'.format(traintype, checkpoint))
            labels_to_names = get_labels_to_names(os.path.join(traintype_modeldir, 'labels.txt'))

            network_fn = nets_factory.get_network_fn(
                export3.model_name,
                num_classes=len(labels_to_names),
                is_training=False)
            image_size = network_fn.default_image_size

            _graph = tf.Graph()
            with _graph.as_default():
                input_image_path = tf.placeholder(dtype=tf.string, name='input_image')
                image_string = tf.read_file(input_image_path)
                image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.expand_dims(image, 0)
                images = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
                images = tf.subtract(images, 0.5)
                images = tf.multiply(images, 2.0)
                logits, _ = network_fn(images)
                probabilities = tf.nn.softmax(logits, name='detection_classes')
                variables_to_restore = tf.global_variables()
                saver = tf.train.Saver(variables_to_restore)
                session = tf.Session(config=config)
                saver.restore(session, checkpoint)
                logger.info('end loading step3 graph...')

            input_image_tensor = _graph.get_tensor_by_name('input_image:0')
            detection_classes = _graph.get_tensor_by_name('detection_classes:0')

            self.labels_to_names_dic[traintype] = labels_to_names
            self._session_dic[traintype] = session
            self._input_image_tensor_dic[traintype] = input_image_tensor
            self._detection_classes_dic[traintype] = detection_classes
            self._load_dic[traintype] = False
        else:
            labels_to_names = self.labels_to_names_dic[traintype]
            session = self._session_dic[traintype]
            input_image_tensor = self._input_image_tensor_dic[traintype]
            detection_classes = self._detection_classes_dic[traintype]

        return labels_to_names, session, input_image_tensor, detection_classes

    def detect(self, config, image_path, traintype):
        cur_export3 = None
        for export3 in self.export3_arr:
            if export3.train_action.traintype == traintype:
                cur_export3 = export3
                break

        if export3 is None:
            return None,None

        labels_to_names, session, input_image_tensor, detection_classes = self.load(config,traintype,cur_export3)

        if session is None:
            return None,None

        probabilities = session.run(
            detection_classes, feed_dict={input_image_tensor: image_path})
        return probabilities, labels_to_names


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
                return None

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
            return None, .0

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
            visualize_boxes_and_labels_on_image_array(
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
