import datetime
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image as im
from django.conf import settings
from django.db.models import Q
from object_detection.utils import visualization_utils as vis_util
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from dl import imagedetection, imagedetectionV2
from dl.step1 import create_onegoods_tf_record, export_inference_graph as e1
from dl.step2 import convert_goods
from .models import Image, Goods
from .serializers import *
import tensorflow as tf

logger = logging.getLogger("django")

class Test(APIView):
    def get(self,request):
        return Response({'Test':True})

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

# 下线旧功能
# class ImageOldViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
#     queryset = Image.objects.order_by('-id')
#     serializer_class = ImageSerializer
#
#     def create(self, request, *args, **kwargs):
#         logger.info('begin old create:')
#         # 兼容没有那么字段的请求
#         if 'deviceid' not in request.data :
#             request.data['deviceid'] = get_client_ip(request)
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         detector = imagedetection_old.ImageDetectorFactory.get_static_detector()
#         logger.info('begin old detect:{}'.format(serializer.instance.source.path))
#         ret = detector.detect(serializer.instance.source.path, min_score_thresh=.8)
#         if len(ret) <= 0:
#             logger.info('end old detect:0')
#             # 删除无用图片
#             os.remove(serializer.instance.source.path)
#             Image.objects.get(pk=serializer.instance.pk).delete()
#         else:
#             logger.info('end old detect:{}'.format(str(len(ret))))
#             for goods in ret:
#                 Goods.objects.create(image_id=serializer.instance.pk,
#                                      class_type=goods['class'],
#                                      score=goods['score'],
#                                      upc=goods['name'],
#                                      xmin=goods['box']['xmin'],
#                                      ymin=goods['box']['ymin'],
#                                      xmax=goods['box']['xmax'],
#                                      ymax=goods['box']['ymax'],
#                                      )
#         logger.info('end old create')
#         #return Response({'Test':True})
#         return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin create:')
        # 兼容没有那么字段的请求
        if 'deviceid' not in request.data :
            request.data['deviceid'] = get_client_ip(request)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # 暂时性分解Detect，需要一个处理type编码
        if serializer.instance.deviceid == '109' or serializer.instance.deviceid == '0':
            detector = imagedetectionV2.ImageDetectorFactory.get_static_detector('0')
            step1_min_score_thresh = .7
            step2_min_score_thresh = .7
            logger.info('begin detect:{},{}'.format(serializer.instance.deviceid, serializer.instance.source.path))
            ret = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh, step2_min_score_thresh=step2_min_score_thresh)
        else:
            detector = imagedetection.ImageDetectorFactory.get_static_detector('10')
            min_score_thresh = .5
            logger.info('begin detect:{},{}'.format(serializer.instance.deviceid, serializer.instance.source.path))
            ret = detector.detect(serializer.instance.source.path, min_score_thresh = min_score_thresh)


        if ret is None or len(ret) <= 0:
            logger.info('end detect:0')
            # 删除无用图片
            os.remove(serializer.instance.source.path)
            Image.objects.get(pk=serializer.instance.pk).delete()
        else:
            logger.info('end detect:{},{}'.format(serializer.instance.deviceid, str(len(ret))))
            ret_reborn = []
            index = 0
            class_index_dict = {}
            for goods in ret:
                # 兼容上一个版本
                if 'score2' not in goods:
                    goods['score2'] = .0
                Goods.objects.create(image_id=serializer.instance.pk,
                                     class_type=goods['class'],
                                     score1=goods['score'],
                                     score2=goods['score2'],
                                     upc=goods['upc'],
                                     xmin=goods['xmin'],
                                     ymin=goods['ymin'],
                                     xmax=goods['xmax'],
                                     ymax=goods['ymax'],
                                     )
                if goods['class'] in class_index_dict:
                    ret_reborn[class_index_dict[goods['class']]]['box'].append({
                        'score': goods['score'],
                        'score2': goods['score2'],
                        'xmin': goods['xmin'],
                        'ymin': goods['ymin'],
                        'xmax': goods['xmax'],
                        'ymax': goods['ymax'],
                    })
                else:
                    box = []
                    box.append({
                        'score': goods['score'],
                        'score2': goods['score2'],
                        'xmin': goods['xmin'],
                        'ymin': goods['ymin'],
                        'xmax': goods['xmax'],
                        'ymax': goods['ymax'],
                    })
                    ret_reborn.append({
                        'class':goods['class'],
                        'upc':goods['upc'],
                        'box':box
                    })
                    class_index_dict[goods['class']] = index
                    index = index + 1
            ret = ret_reborn
        logger.info('end create')
        #return Response({'Test':True})
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class ProblemGoodsViewSet(DefaultMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    queryset = ProblemGoods.objects.order_by('-id')
    serializer_class = ProblemGoodsSerializer

def get_client_ip(request):
    try:
      real_ip = request.META['HTTP_X_FORWARDED_FOR']
      regip = real_ip.split(",")[0]
    except:
      try:
        regip = request.META['REMOTE_ADDR']
      except:
        regip = ""
    return regip

class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        # 兼容没有那么字段的请求
        # if 'name' not in request.data :
        #     request.data['name'] = request.data['upc']
        if 'deviceid' not in request.data :
            request.data['deviceid'] = get_client_ip(request)
        if 'traintype' not in request.data :
            request.data['traintype'] = 0

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create image xml:{}'.format(serializer.instance.source.path))

        image_path = serializer.instance.source.path
        image = im.open(image_path)
        xml_path, _ = os.path.split(os.path.realpath(__file__))
        xml_path = os.path.join(xml_path, 'template.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root.find('size').find('width').text = str(image.size[0])
        root.find('size').find('height').text = str(image.size[1])
        root.find('object').find('name').text = str(serializer.instance.upc)
        for box in root.iter('bndbox'):
            # 改变xml中的坐标值
            box.find('xmin').text = str(serializer.instance.xmin)
            box.find('ymin').text = str(serializer.instance.ymin)
            box.find('xmax').text = str(serializer.instance.xmax)
            box.find('ymax').text = str(serializer.instance.ymax)
        # 写入新的xml
        a, b = os.path.splitext(serializer.instance.source.path)
        tree.write(a + ".xml")

        # 画带box的图片
        image_dir = os.path.dirname(image_path)
        output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
        image = im.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        image_pil = im.fromarray(np.uint8(image_np)).convert('RGB')

        vis_util.draw_bounding_box_on_image(image_pil,
                                            serializer.instance.ymin, serializer.instance.xmin, serializer.instance.ymax, serializer.instance.xmax,
                                            color='DarkOrange',
                                            display_str_list=(serializer.instance.name,), use_normalized_coordinates=False)

        np.copyto(image_np, np.array(image_pil))
        output_image = im.fromarray(image_np)
        output_image.thumbnail((int(im_width * 0.5), int(im_height * 0.5)), im.ANTIALIAS)
        output_image.save(output_image_path)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        a, b = os.path.splitext(instance.source.path)
        dir = os.path.dirname(instance.source.path)
        if os.path.isfile(instance.source.path):
            os.remove(instance.source.path)
        if os.path.isfile(a + ".xml"):
            os.remove(a + ".xml")

        havefile = False
        for i in os.listdir(dir):
            havefile = True
            break
        if not havefile:
            shutil.rmtree(dir)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


step2_model_name = 'inception_resnet_v2'
class ActionLogViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, viewsets.GenericViewSet):
    queryset = ActionLog.objects.order_by('-id')
    serializer_class = ActionLogSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create action:{}'.format(serializer.instance.action))

        if serializer.instance.action == 'T1':
            # 杀死原来的train
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            # 训练准备
            if serializer.instance.traintype == 0:
                data_dir = os.path.join(settings.MEDIA_ROOT, 'data')
            else:
                data_dir = os.path.join(settings.MEDIA_ROOT, str(serializer.instance.traintype))
            label_map_dict = create_onegoods_tf_record.prepare_train(data_dir, settings.TRAIN_ROOT, str(serializer.instance.pk))
            serializer.instance.param = str(label_map_dict)
            serializer.instance.save()

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))

            # 训练
            # --num_clones=2 同步修改config中的batch_size=2
            command = 'nohup python3 {}/step1/train.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={}  > /root/train1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
            )
            logger.info(command)
            subprocess.call(command, shell=True)
            # 评估
            command = 'nohup python3 {}/step1/eval.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --checkpoint_dir={} --eval_dir={}  > /root/eval1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                os.path.join(train_logs_dir, 'eval_log'),
            )
            logger.info(command)
            subprocess.call(command, shell=True)
        elif serializer.instance.action == 'T2':
            # 杀死原来的train
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            # 训练准备
            if serializer.instance.traintype == 0:
                data_dir = os.path.join(settings.MEDIA_ROOT, 'data')
            else:
                data_dir = os.path.join(settings.MEDIA_ROOT, str(serializer.instance.traintype))
            class_names_to_ids, training_filenames, validation_filenames = convert_goods.prepare_train(data_dir, settings.TRAIN_ROOT, str(serializer.instance.pk))

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))

            # 训练
            command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --batch_size={}  > /root/train2.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                len(training_filenames),
                step2_model_name,
                32
            )
            logger.info(command)
            subprocess.call(command, shell=True)
            # 评估
            command = 'nohup python3 {}/step2/eval.py --dataset_split_name=validation --dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}  > /root/eval2.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                os.path.join(train_logs_dir, 'eval_log'),
                len(validation_filenames),
                step2_model_name
            )
            logger.info(command)
            subprocess.call(command, shell=True)
        elif serializer.instance.action == 'ST':
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
        elif serializer.instance.action == 'E1':

            lastT1 = ActionLog.objects.filter(action="T1").filter(traintype=serializer.instance.traintype).order_by('-id')[0]
            logger.info('Export Grapy from train1:{}'.format(lastT1.pk))
            trained_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(lastT1.pk))
            prefix = 0
            for i in os.listdir(trained_checkpoint_dir):
                a, b = os.path.splitext(i)
                if b == '.meta':
                    t_prefix = int(a.split('-')[1])
                    if t_prefix > prefix:
                        prefix = t_prefix

            if prefix > 0:
                trained_checkpoint_prefix = os.path.join(trained_checkpoint_dir, 'model.ckpt-{}'.format(prefix))
                # 备份上一个pb
                model_dir = os.path.join(settings.BASE_DIR, 'dl', 'model', str(lastT1.traintype))
                export_file_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
                label_file_path = os.path.join(model_dir, 'goods_label_map.pbtxt')

                if os.path.isfile(export_file_path):
                    now = datetime.datetime.now()
                    postfix = now.strftime('%Y%m%d%H%M%S')
                    os.rename(export_file_path, export_file_path+'.'+postfix)
                    os.rename(label_file_path, label_file_path+'.'+postfix)
                    shutil.rmtree(os.path.join(model_dir, 'saved_model'))
                    serializer.instance.param = 'trainid:{},prefix:{},postfix:{}'.format(lastT1.pk, prefix, postfix)
                    serializer.instance.save()
                # 输出pb
                e1.export(os.path.join(trained_checkpoint_dir, 'faster_rcnn_nas_goods.config'),
                                              trained_checkpoint_prefix,
                                              model_dir,
                                              )
                # copy label
                shutil.copy(os.path.join(settings.TRAIN_ROOT, str(lastT1.pk), 'goods_label_map.pbtxt'), label_file_path)

                # reboot django
                os.utime(os.path.join(settings.BASE_DIR, 'main', 'settings.py'), None)

        elif serializer.instance.action == 'E2':
            lastT2 = ActionLog.objects.filter(action="T2").filter(traintype=serializer.instance.traintype).order_by('-id')[0]
            logger.info('Export Grapy from train2:{}'.format(lastT2.pk))
            trained_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(lastT2.pk))
            checkpoint_model_path = tf.train.latest_checkpoint(trained_checkpoint_dir)
            model_dir = os.path.join(settings.BASE_DIR, 'dl', 'model', str(lastT2.traintype))
            checkpoint_file_path = os.path.join(model_dir, 'checkpoint')

            if os.path.isfile(checkpoint_file_path):
                # 备份上一个checkpoint
                now = datetime.datetime.now()
                postfix = now.strftime('%Y%m%d%H%M%S')
                os.rename(checkpoint_file_path, checkpoint_file_path+'.'+postfix)
                label_file_path = os.path.join(model_dir, 'labels.txt')
                os.rename(label_file_path, label_file_path+'.'+postfix)
                serializer.instance.param = 'trainid:{},postfix:{}'.format(lastT2.pk, postfix)
                serializer.instance.save()
            # 输出pb
            # e2.export(step2_model_name, trained_checkpoint_dir, export_file_path)
            # 重写checkpoint file
            with open(checkpoint_file_path, 'w') as output:
                a = os.path.split(checkpoint_model_path)
                output.write('model_checkpoint_path: {}\n'.format(os.path.join(model_dir, a[1])))
                output.write('all_model_checkpoint_path: {}\n'.format(os.path.join(model_dir, a[1])))

            shutil.copy(checkpoint_model_path + '.data-00000-of-00001', model_dir)
            shutil.copy(checkpoint_model_path + '.index', model_dir)
            shutil.copy(checkpoint_model_path + '.meta', model_dir)

            # copy dataset
            shutil.copy(os.path.join(settings.TRAIN_ROOT, str(lastT2.pk), 'goods_recogonize_train.tfrecord'), model_dir)

            # copy label
            shutil.copy(os.path.join(settings.TRAIN_ROOT, str(lastT2.pk), 'labels.txt'), model_dir)

            # reboot django
            os.utime(os.path.join(settings.BASE_DIR, 'main', 'settings.py'), None)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if serializer.instance.action == 'T1':
            # 杀死原来的train
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))
            # 继续训练 --num_clones=2
            command = 'nohup python3 {}/step1/train.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={}  > /root/train1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
            )
            logger.info(command)
            subprocess.call(command, shell=True)
            # 评估
            command = 'nohup python3 {}/step1/eval.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --checkpoint_dir={} --eval_dir={}  > /root/eval1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                os.path.join(train_logs_dir, 'eval_log'),
            )
            logger.info(command)
            subprocess.call(command, shell=True)

        if serializer.instance.action == 'T2':
            # TODO
            pass
        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

