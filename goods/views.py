from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import *
from rest_framework import status
from dl import imagedetection, imagedetection_old
import logging
import os
import shutil
import datetime
import subprocess
from .models import Image, Goods
import xml.etree.ElementTree as ET
from PIL import Image as im
from dl import create_goods_tf_record
from dl import export_inference_graph
from django.conf import settings
from PIL import Image
import numpy as np
from object_detection.utils import visualization_utils as vis_util

logger = logging.getLogger("django")

class Test(APIView):
    def get(self,request):
        return Response({'Test':True})

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class ImageOldViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin create:')
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        detector = imagedetection_old.ImageDetectorFactory.get_static_detector()
        logger.info('begin detect:{}'.format(serializer.instance.source.path))
        ret = detector.detect(serializer.instance.source.path, min_score_thresh=.8)
        if len(ret) <= 0:
            logger.info('end detect:0')
            # 删除无用图片
            os.remove(serializer.instance.source.path)
            Image.objects.get(pk=serializer.instance.pk).delete()
        else:
            logger.info('end detect:{}'.format(str(len(ret))))
            for goods in ret:
                Goods.objects.create(image_id=serializer.instance.pk,
                                     class_type=goods['class'],
                                     score=goods['score'],
                                     upc=goods['name'],
                                     xmin=goods['box']['xmin'],
                                     ymin=goods['box']['ymin'],
                                     xmax=goods['box']['xmax'],
                                     ymax=goods['box']['ymax'],
                                     )
        logger.info('end create')
        #return Response({'Test':True})
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin create:')
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        detector = imagedetection.ImageDetectorFactory.get_static_detector()
        logger.info('begin detect:{}'.format(serializer.instance.source.path))
        ret = detector.detect(serializer.instance.source.path, min_score_thresh = .5)
        if len(ret) <= 0:
            logger.info('end detect:0')
            # 删除无用图片
            os.remove(serializer.instance.source.path)
            Image.objects.get(pk=serializer.instance.pk).delete()
        else:
            logger.info('end detect:{}'.format(str(len(ret))))
            ret_reborn = []
            index = 0
            class_index_dict = {}
            for goods in ret:
                Goods.objects.create(image_id=serializer.instance.pk,
                                     class_type=goods['class'],
                                     score=goods['score'],
                                     upc=goods['upc'],
                                     xmin=goods['xmin'],
                                     ymin=goods['ymin'],
                                     xmax=goods['xmax'],
                                     ymax=goods['ymax'],
                                     )
            if goods['class'] in class_index_dict:
                ret_reborn[class_index_dict[goods['class']]]['box'].append({
                    'score': goods['score'],
                    'xmin': goods['xmin'],
                    'ymin': goods['ymin'],
                    'xmax': goods['xmax'],
                    'ymax': goods['ymax'],
                })
            else:
                box = []
                box.append({
                    'score': goods['score'],
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

class GoodsViewSet(DefaultMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    queryset = Goods.objects.order_by('id')
    serializer_class = GoodsSerializer

class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
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
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')

        vis_util.draw_bounding_box_on_image(image_pil,
                                            serializer.instance.ymin, serializer.instance.xmin, serializer.instance.ymax, serializer.instance.xmax,
                                            color='DarkOrange',
                                            display_str_list=(serializer.instance.upc,), use_normalized_coordinates=False)

        np.copyto(image_np, np.array(image_pil))
        output_image = Image.fromarray(image_np)
        output_image.thumbnail((int(im_width * 0.5), int(im_height * 0.5)), Image.ANTIALIAS)
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


class ActionLogViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, viewsets.GenericViewSet):
    queryset = ActionLog.objects.order_by('-id')
    serializer_class = ActionLogSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create action:{}'.format(serializer.instance.action))

        if serializer.instance.action == 'BT':
            # 杀死原来的train
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            # 训练准备
            data_dir = os.path.join(settings.MEDIA_ROOT, 'data')
            label_map_dict = create_goods_tf_record.prepare_train(data_dir, settings.TRAIN_ROOT, str(serializer.instance.pk))
            serializer.instance.param = str(label_map_dict)
            serializer.instance.save()

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))

            # 训练
            command = 'nohup python3 {}/train.py --logtostderr --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={}  > train.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
            )
            logger.info(command)
            subprocess.call(command, shell=True)
        elif serializer.instance.action == 'ET':
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
        elif serializer.instance.action == 'EG':

            lastBT = ActionLog.objects.filter(action='BT').order_by('-id')[0]
            trained_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(lastBT.pk))
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
                model_dir = os.path.join(settings.BASE_DIR, 'dl', 'model')
                export_file_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
                label_file_path = os.path.join(model_dir, 'goods_label_map.pbtxt')
                if os.path.isfile(export_file_path):
                    now = datetime.datetime.now()
                    postfix = now.strftime('%Y%m%d%H%M%S')
                    os.rename(export_file_path, export_file_path+'.'+postfix)
                    os.rename(label_file_path, label_file_path+'.'+postfix)
                    serializer.instance.param = 'trainid:{},prefix:{}'.format(lastBT.pk, prefix)
                    serializer.instance.save()
                # 输出pb
                export_inference_graph.export(os.path.join(settings.TRAIN_ROOT, lastBT.pk, 'faster_rcnn_nas_goods.config'),
                                              trained_checkpoint_prefix,
                                              model_dir,
                                              )
                # copy label
                shutil.copy(os.path.join(settings.TRAIN_ROOT, lastBT.pk, 'goods_label_map.pbtxt'), label_file_path)

                # reboot django
                os.utime(os.path.join(settings.BASE_DIR, 'main', 'setting.py'), None)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if serializer.instance.action == 'BT':
            # 杀死原来的train
            os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))
            # 继续训练
            command = 'nohup python3 {}/train.py --logtostderr --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={}  > train.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
            )
            logger.info(command)
            subprocess.call(command, shell=True)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

