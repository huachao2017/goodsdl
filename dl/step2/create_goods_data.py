import os
import logging
import numpy as np
import cv2
import math
import time
from PIL import Image as im
import xml.etree.ElementTree as ET

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


# def get_class_names(labels_filepath):
#     with tf.gfile.Open(labels_filepath, 'rb') as f:
#         lines = f.read().decode()
#     lines = lines.split('\n')
#     lines = filter(None, lines)
#
#     class_names = []
#     for line in lines:
#         index = line.index(':')
#         class_names.append(line[index + 1:])
#
#     return class_names


def create_step2_goods(data_dir, dataset_dir, step1_model_path, dir_day_hour=None):
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
        class_dir = os.path.join(data_dir, dirlist[i])
        if os.path.isdir(class_dir):
            if dir_day_hour is not None:
                cur_dir_day_hour = time.strftime('%d%H', time.localtime(os.path.getctime(class_dir)))
                if cur_dir_day_hour != dir_day_hour:
                    continue
            logging.info('solve class:{}'.format(dirlist[i]))
            output_class_dir = os.path.join(dataset_dir, dirlist[i])
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)

            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                example, ext = os.path.splitext(image_path)
                xml_path = example + '.xml'
                if ext == ".jpg" and os.path.isfile(xml_path):
                    logging.info('solve image:{}'.format(image_path))
                    image = im.open(image_path)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    # TODO 还未支持多类型
                    index = 0
                    for box in root.iter('bndbox'):
                        index = index + 1
                        # 改变xml中的坐标值
                        xmin = int(box.find('xmin').text)
                        ymin = int(box.find('ymin').text)
                        xmax = int(box.find('xmax').text)
                        ymax = int(box.find('ymax').text)
                        newimage = image.crop((xmin, ymin, xmax, ymax))
                        # 生成新的图片
                        output_image_path = os.path.join(output_class_dir,
                                                         "{}_{}.jpg".format(os.path.split(example)[1], index))
                        if not tf.gfile.Exists(output_image_path):
                            newimage.save(output_image_path, 'JPEG')
                            # logging.info('save image:{}'.format(output_image_path))

                        img = cv2.imread(image_path)

                        # augment small sample
                        # if len(filelist) < 3 * 6:
                        #     augment_ratio = 5
                        # elif len(filelist) < 3 * 8:
                        #     augment_ratio = 4
                        # elif len(filelist) < 3 * 10:
                        #     augment_ratio = 3
                        # elif len(filelist) < 3 * 15:
                        #     augment_ratio = 2
                        # else:
                        #     augment_ratio = 1
                        # 使图像旋转
                        for k in range(7):
                            angle = 45 + k * 45
                            output_image_path_augment = os.path.join(output_class_dir, "{}_{}_augment{}.jpg".format(
                                os.path.split(example)[1], index, angle))
                            if tf.gfile.Exists(output_image_path_augment):
                                # 文件存在不再重新生成，从而支持增量生成
                                continue
                            # logging.info("image:{} rotate {}.".format(output_image_path, angle))
                            rotated_img = rotate_image(img, angle)
                            # logging.info("rotate image...")
                            # 写入图像
                            # tmp_image_path = os.path.join(output_tmp_dir,
                            #                               "{}_{}_{}.jpg".format(os.path.split(example)[1], index, k))
                            # cv2.imwrite(tmp_image_path, rotated_img)

                            # augment_image = im.open(tmp_image_path)
                            # (im_width, im_height) = augment_image.size
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
                            if boxes.shape[0] <= 0:
                                augment_total += 1
                                augment_total_error += 1
                                logging.error("image:{} ,rotate:{}, thresh:{}, count:{}/{}.".format(
                                    output_image_path, angle, str(scores_step1[l]), augment_total_error, augment_total))

                            for l in range(boxes.shape[0]):
                                augment_total += 1
                                if scores_step1[l] < 0.8:
                                    augment_total_error += 1
                                    logging.error("image:{} ,rotate:{}, thresh:{}, count:{}/{}.".format(
                                        output_image_path, angle, str(scores_step1[l]), augment_total_error, augment_total))
                                else:
                                    ymin, xmin, ymax, xmax = boxes[l]
                                    ymin = int(ymin * im_height)
                                    xmin = int(xmin * im_width)
                                    ymax = int(ymax * im_height)
                                    xmax = int(xmax * im_width)

                                    # if ymax-ymin > im_height - 5 and xmax-xmin > im_width - 5:
                                    #     # 如果没有识别准确，不采用次旋转样本
                                    #     logging.warning('detect failed:{}'.format(output_image_path_augment))
                                    #     break

                                    # augment_newimage = augment_image.crop((xmin, ymin, xmax, ymax))
                                    augment_newimage = rotated_img[ymin:ymax, xmin:xmax]
                                    # augment_newimage.save(output_image_path_augment, 'JPEG')
                                    cv2.imwrite(output_image_path_augment, augment_newimage)
                                    # logging.info("save image...")
                                break
    logging.info("augment complete: {}/{}".format(augment_total_error, augment_total))
    session_step1.close()


def create_step2_goods_V2(data_dir, dataset_dir, step1_model_path, dir_day_hour=None):
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
        class_dir = os.path.join(data_dir, dirlist[i])
        if os.path.isdir(class_dir):
            if dir_day_hour is not None:
                cur_dir_day_hour = time.strftime('%d%H', time.localtime(os.path.getmtime(class_dir)))
                logging.info('solve class:{}_{}'.format(dirlist[i],cur_dir_day_hour))
                if cur_dir_day_hour != dir_day_hour:
                    continue
            logging.info('solve class:{}'.format(dirlist[i]))
            output_class_dir = os.path.join(dataset_dir, dirlist[i])
            if not tf.gfile.Exists(output_class_dir):
                tf.gfile.MakeDirs(output_class_dir)

            filelist = os.listdir(class_dir)
            for j in range(0, len(filelist)):
                image_path = os.path.join(class_dir, filelist[j])
                prefix = filelist[j].split('_')[0]
                example, ext = os.path.splitext(image_path)
                if ext == ".jpg" and prefix != 'visual':
                    logging.info('solve image:{}'.format(image_path))
                    img = cv2.imread(image_path)

                    augment_size = 8
                    if dirlist[i] == 'ziptop-drink-stand' or dirlist[i] == 'bottled-drink-stand':
                        augment_size = 1
                    for k in range(augment_size):
                        angle = k * 45
                        output_image_path_augment = os.path.join(output_class_dir, "{}_augment{}.jpg".format(
                            os.path.split(example)[1], angle))
                        if tf.gfile.Exists(output_image_path_augment):
                            # 文件存在不再重新生成，从而支持增量生成
                            continue
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
                        if boxes.shape[0] <= 0:
                            augment_total += 1
                            augment_total_error += 1
                            logging.error("image:{} ,rotate:{}, count:{}/{}.".format(
                                output_image_path_augment, angle, augment_total_error, augment_total))

                        new_boxes = []
                        for l in range(boxes.shape[0]):
                            if scores_step1[l] > 0.7:
                                new_boxes.append(boxes[l])
                        augment_total += 1
                        if len(new_boxes) <= 0:
                            augment_total_error += 1
                            logging.error("image:{} ,rotate:{}, count:{}/{}.".format(
                                output_image_path_augment, angle, augment_total_error,
                                augment_total))
                        elif len(new_boxes) == 1:
                            ymin, xmin, ymax, xmax = new_boxes[0]
                            ymin = int(ymin * im_height)
                            xmin = int(xmin * im_width)
                            ymax = int(ymax * im_height)
                            xmax = int(xmax * im_width)

                            # if ymax-ymin > im_height - 5 and xmax-xmin > im_width - 5:
                            #     # 如果没有识别准确，不采用次旋转样本
                            #     logging.warning('detect failed:{}'.format(output_image_path_augment))
                            #     break

                            # augment_newimage = augment_image.crop((xmin, ymin, xmax, ymax))
                            augment_newimage = rotated_img[ymin:ymax, xmin:xmax]
                            # augment_newimage.save(output_image_path_augment, 'JPEG')
                            cv2.imwrite(output_image_path_augment, augment_newimage)
                            # logging.info("save image...")
                        else:
                            index = -1
                            area = 0
                            filter_area = im_height*im_width*.9
                            for l in range(len(new_boxes)):
                                # 取最大面积的识别物体
                                ymin, xmin, ymax, xmax = new_boxes[l]
                                ymin = int(ymin * im_height)
                                xmin = int(xmin * im_width)
                                ymax = int(ymax * im_height)
                                xmax = int(xmax * im_width)
                                if area < (ymax-ymin)*(xmax-xmin) and filter_area > (ymax-ymin)*(xmax-xmin):
                                   area = (ymax-ymin)*(xmax-xmin)
                                   index = l

                            if index >= 0:
                                ymin, xmin, ymax, xmax = new_boxes[index]
                                ymin = int(ymin * im_height)
                                xmin = int(xmin * im_width)
                                ymax = int(ymax * im_height)
                                xmax = int(xmax * im_width)
                                # augment_newimage = augment_image.crop((xmin, ymin, xmax, ymax))
                                augment_newimage = rotated_img[ymin:ymax, xmin:xmax]
                                # augment_newimage.save(output_image_path_augment, 'JPEG')
                                cv2.imwrite(output_image_path_augment, augment_newimage)
                                # logging.info("save image...")
                            else:
                                augment_total_error += 1
                                logging.error("image:{} ,rotate:{}, count:{}/{}.".format(
                                    output_image_path_augment, angle, augment_total_error,
                                    augment_total))

    logging.info("augment complete: {}/{}".format(augment_total_error, augment_total))
    session_step1.close()

def prepare_data(source_dir,dest_dir,step1_model_path):
    """Runs the data augument.

    Args:
      source_dir: The source directory where the step1 dataset is stored.
      dest_dir: step2 dataset will be stored.
    """

    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MakeDirs(dest_dir)

    # _clean_up_temporary_files(dataset_dir)
    create_step2_goods(source_dir, dest_dir, step1_model_path)

tf.app.flags.DEFINE_string(
    'day_hour', None,
    'day and hour')
tf.app.flags.DEFINE_string(
    'device', "0",
    'device id')
FLAGS = tf.app.flags.FLAGS

def main(_):
    if not FLAGS.day_hour:
        raise ValueError('You must supply day and hour --day_hour')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device
    logger = logging.getLogger()
    logger.setLevel('INFO')
    dataset_dir = '/home/src/goodsdl/media/dataset'
    source_dir = os.path.join(dataset_dir, 'data_new')
    step2_dir = os.path.join(dataset_dir, 'step2')
    step1_model_path = os.path.join('/home/src/goodsdl/dl/model/58/','frozen_inference_graph.pb')

    create_step2_goods_V2(source_dir, step2_dir, step1_model_path, dir_day_hour=FLAGS.day_hour)

if __name__ == '__main__':
    tf.app.run()
