import os
import logging
import numpy as np
import cv2
import math
import time
from PIL import Image as im
import xml.etree.ElementTree as ET
from dl import common
from matcher.matcher import Matcher
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import ExportAction

import tensorflow as tf

# from datasets import dataset_utils

def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,
                          borderValue=(160, 160, 160)) # 桌面样本背景色


def solves_one_image(image_path,
                     class_name,
                     output_class_dir,
                     session_step1,
                     image_tensor_step1,
                     detection_boxes,
                     detection_scores,
                     augment_total,
                     augment_total_error
                     ):
    img = cv2.imread(image_path)

    augment_size = 8
    if class_name == 'ziptop-drink-stand' or class_name == 'bottled-drink-stand':
        augment_size = 1

    matcher = None
    for i in range(augment_size):
        angle = int(i * 360 / augment_size)
        output_image_path_augment = os.path.join(output_class_dir, "{}_augment{}.jpg".format(
            os.path.split(os.path.splitext(image_path)[0])[1], angle))
        if tf.gfile.Exists(output_image_path_augment):
            # 文件存在不再重新生成，从而支持增量生成
            break
        logging.info("image:{} rotate {}.".format(output_image_path_augment, angle))
        if angle > 0:
            rotated_img = rotate_image(img, angle)
        else:
            rotated_img = img

        im_height = rotated_img.shape[0]
        im_width = rotated_img.shape[1]
        image_np = np.asarray(rotated_img).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        # logging.info("begin detect...")
        (boxes, scores) = session_step1.run(
            [detection_boxes, detection_scores],
            feed_dict={image_tensor_step1: image_np_expanded})
        # logging.info("end detect...")
        # data solving
        boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes).astype(np.int32)
        scores_step1 = np.squeeze(scores)
        augment_total += 1

        new_boxes = []
        for j in range(boxes.shape[0]):
            if scores_step1[j] > 0.7:
                new_boxes.append(boxes[j])
        if len(new_boxes) <= 0:
            augment_total_error += 1
            logging.error("not detected error! image:{} ,rotate:{}.".format(image_path, angle))
            if angle == 0:
                break
        elif len(new_boxes) == 1:
            ymin, xmin, ymax, xmax = new_boxes[0]
            ymin = int(ymin * im_height)
            xmin = int(xmin * im_width)
            ymax = int(ymax * im_height)
            xmax = int(xmax * im_width)

            augment_newimage = rotated_img[ymin:ymax, xmin:xmax]
            cv2.imwrite(output_image_path_augment, augment_newimage)
            if angle == 0:
                matcher = Matcher()
                matcher.add_baseline_image(output_image_path_augment)
            else:
                if not matcher.is_find_match(output_image_path_augment, solve_center=True):
                    os.remove(output_image_path_augment)
                    augment_total_error += 1
                    logging.error("match error! image:{} ,rotate:{}.".format(image_path, angle))

        else:
            index = -1
            area = 0
            filter_area = im_height * im_width * .9
            for j in range(len(new_boxes)):
                # 取最大面积的识别物体
                ymin, xmin, ymax, xmax = new_boxes[j]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)
                if area < (ymax - ymin) * (xmax - xmin) and filter_area > (ymax - ymin) * (xmax - xmin):
                    area = (ymax - ymin) * (xmax - xmin)
                    index = j

            if index >= 0:
                ymin, xmin, ymax, xmax = new_boxes[index]
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)
                # augment_newimage = augment_image.crop((xmin, ymin, xmax, ymax))
                augment_newimage = rotated_img[ymin:ymax, xmin:xmax]
                cv2.imwrite(output_image_path_augment, augment_newimage)
                if angle == 0:
                    matcher = Matcher()
                    matcher.add_baseline_image(output_image_path_augment)
                else:
                    if not matcher.is_find_match(output_image_path_augment, solve_center=True):
                        os.remove(output_image_path_augment)
                        augment_total_error += 1
                        logging.error("match error! image:{} ,rotate:{}.".format(image_path, angle))
            else:
                augment_total_error += 1
                logging.error("not detected error! image:{} ,rotate:{}.".format(output_image_path_augment, angle))
                if angle == 0:
                    break


def create_step2_goods_V2(data_dir, output_dir, step1_model_path, dir_day_hour=None):
    graph_step1 = tf.Graph()
    with graph_step1.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(step1_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
    session_step1 = tf.Session(graph=graph_step1, config=config)

    # Definite input and output Tensors for detection_graph
    image_tensor_step1 = graph_step1.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = graph_step1.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = graph_step1.get_tensor_by_name('detection_scores:0')

    # class_names = get_class_names(os.path.join(os.path.dirname(step1_model_path), dataset_utils.LABELS_FILENAME))
    """返回所有图片文件路径"""

    augment_total = 0
    augment_total_error = 0
    dirlist = os.listdir(data_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirlist)):
        # 根据step1的classname确定进入step2的类别
        # 不再需要，step1的检测应该可以泛化
        # if dirlist[i] not in class_names:
        #     continue

        class_name = dirlist[i]
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            if dir_day_hour is not None:
                cur_dir_day_hour = time.strftime('%d%H', time.localtime(os.path.getmtime(class_dir)))
                if cur_dir_day_hour != dir_day_hour:
                    logging.info('skip class:{}({})'.format(class_name, cur_dir_day_hour))
                    continue
            logging.info('solve class:{}'.format(class_name))
            output_class_dir = os.path.join(output_dir, class_name)
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)

            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                prefix = filelist[j].split('_')[0]
                example, ext = os.path.splitext(image_path)
                if ext == ".jpg" and prefix != 'visual':
                    logging.info('solve image:{}'.format(image_path))
                    solves_one_image(
                        image_path,
                        class_name,
                        output_class_dir,
                        session_step1,
                        image_tensor_step1,
                        detection_boxes,
                        detection_scores,
                        augment_total,
                        augment_total_error
                    )

    logging.info("augment complete: {}/{}".format(augment_total_error, augment_total))
    session_step1.close()

tf.app.flags.DEFINE_string(
    'day_hour', None,
    'day and hour')
tf.app.flags.DEFINE_string(
    'source_dir_serial', '',
    'source dir serial')
tf.app.flags.DEFINE_string(
    'dest_dir_serial', '',
    'dest dir serial')
tf.app.flags.DEFINE_string(
    'device', "0",
    'device id')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # if not FLAGS.day_hour:
    #     raise ValueError('You must supply day and hour --day_hour')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device
    logger = logging.getLogger()
    logger.setLevel('INFO')
    dataset_dir = '/home/src/goodsdl/media/dataset'
    source_dir = os.path.join(dataset_dir, 'data_new{}'.format(FLAGS.source_dir_serial if FLAGS.source_dir_serial=='' else '_'+FLAGS.source_dir_serial))
    step2_dir = os.path.join(dataset_dir, common.STEP2_PREFIX if FLAGS.dest_dir_serial=='' else common.STEP2_PREFIX+'_'+FLAGS.dest_dir_serial)
    export1s = ExportAction.objects.filter(train_action__action='T1').order_by('-update_time')[:1]
    step1_model_path = os.path.join('/home/src/goodsdl/dl/model', str(export1s[0].pk), 'frozen_inference_graph.pb')

    create_step2_goods_V2(source_dir, step2_dir, step1_model_path, dir_day_hour=FLAGS.day_hour)

if __name__ == '__main__':
    tf.app.run()
