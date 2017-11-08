from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import *
from rest_framework import status
from dl.imagedetection import ImageDetectorFactory, ImageDetector
import logging
import os
from .models import Image, Goods
import xml.etree.ElementTree as ET
from PIL import Image as im
from dl import create_goods_tf_record
from django.conf import settings

logger = logging.getLogger("django")

class Test(APIView):
    def get(self,request):
        return Response({'Test':True})

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin create:')
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        detector = ImageDetectorFactory.get_static_detector()
        logger.info('begin detect:{}'.format(serializer.instance.source.path))
        ret = detector.detect(serializer.instance.source.path, min_score_thresh = .8)
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
                                     name=goods['name'],
                                     xmin=goods['box']['xmin'],
                                     ymin=goods['box']['ymin'],
                                     xmax=goods['box']['xmax'],
                                     ymax=goods['box']['ymax'],
                                     )
        logger.info('end create')
        #return Response({'Test':True})
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class GoodsViewSet(DefaultMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    queryset = Goods.objects.order_by('id')
    serializer_class = GoodsSerializer

class TrainImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create image xml:{}'.format(serializer.instance.source.path))

        image = im.open(serializer.instance.source.path)
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

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class ActionLogViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ActionLog.objects.order_by('-id')
    serializer_class = ActionLogSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create action:{}'.format(serializer.instance.action))

        if serializer.instance.action == 'BT':
            # 训练准备
            data_dir = os.path.join(settings.MEDIA_ROOT, 'data')
            output_dir = settings.TRAIN_ROOT
            label_map_dict = create_goods_tf_record.prepare_train(data_dir, output_dir)
            serializer.instance.param = str(label_map_dict)
            serializer.instance.save()
            #TODO 训练
        elif serializer.instance.action == 'ET':
            pass
            #TODO 结束训练
        elif serializer.instance.action == 'EG':
            pass
            #TODO 输出pb

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
