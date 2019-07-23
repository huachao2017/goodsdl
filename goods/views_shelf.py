import datetime
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from PIL import Image as PILImage

import numpy as np
from django.conf import settings
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
import goods.util
import cv2
import math

from dl import shelfdetection
# from dl.old import imagedetection
from .serializers import *
import tensorflow as tf

logger = logging.getLogger("detect")

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class CreateShelfImage(APIView):

    def get(self, request):

        shopid = request.query_params['shopid']
        shelfid = request.query_params['shelfid']
        if 'tlevel' in request.query_params:
            tlevel = int(request.query_params['tlevel'])
        else:
            tlevel = 6
        picurl = request.query_params['picurl']
        logger.info('begin detect:{},{}'.format(shopid, shelfid))
        now = datetime.datetime.now()
        image_name = '{}.jpg'.format(now.strftime('%Y%m%d_%H%M%S'))
        shelf_image = ShelfImage.objects.create(
            shopid = shopid,
            shelfid = shelfid,
            picurl = picurl,
            image_name = image_name,
        )

        ret = []
        export1s = ExportAction.objects.filter(train_action__action='T1').filter(train_action__traintype=2).filter(checkpoint_prefix__gt=0).order_by(
            '-update_time')[:1]

        if len(export1s) > 0:
            detector = shelfdetection.ShelfDetectorFactory.get_static_detector(export1s[0].pk)
            step1_min_score_thresh = .5
            media_dir = settings.MEDIA_ROOT
            # 通过 picurl 获取图片
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', '{}_{}'.format(shopid, shelfid))
            if not tf.gfile.Exists(image_dir):
                tf.gfile.MakeDirs(image_dir)
            image_path = os.path.join(image_dir, image_name)
            urllib.request.urlretrieve(picurl, image_path)
            detect_ret, aiinterval, visual_image_path = detector.detect(image_path, shopid, shelfid, step1_min_score_thresh=step1_min_score_thresh,totol_level = tlevel)

            logger.info('create shelf image: {},{}'.format(len(detect_ret), aiinterval))
            for one_box in detect_ret:
                shelf_goods = ShelfGoods.objects.create(
                    shelf_image_id = shelf_image.pk,
                    shopid=shopid,
                    shelfid=shelfid,
                    xmin = one_box['xmin'],
                    ymin = one_box['ymin'],
                    xmax = one_box['xmax'],
                    ymax = one_box['ymax'],
                    level = one_box['level'],
                    upc = one_box['upc'],
                    score1 = one_box['score'],
                    score2 = one_box['score2']
                )
                ret.append({
                    'id': shelf_goods.pk,
                    'xmin': shelf_goods.xmin,
                    'ymin': shelf_goods.ymin,
                    'xmax': shelf_goods.xmax,
                    'ymax': shelf_goods.ymax,
                    'level': shelf_goods.level,
                    'upc': shelf_goods.upc,
                    'score': shelf_goods.score2,
                })

        logger.info('end detect:{},{}'.format(shopid, shelfid))
        return Response(goods.util.wrap_ret(ret), status=status.HTTP_200_OK)


class RectifyShelfImage(APIView):
    def get(self, request):

        picurl = request.query_params['picurl']
        x1 = int(request.query_params['x1'])
        y1 = int(request.query_params['y1'])
        x2 = int(request.query_params['x2'])
        y2 = int(request.query_params['y2'])
        if x1>x2:
            xt = x1
            yt = y1
            x1 = x2
            y1 = y2
            x2 = xt
            y2 = yt

        x3 = int(request.query_params['x3'])
        y3 = int(request.query_params['y3'])
        x4 = int(request.query_params['x4'])
        y4 = int(request.query_params['y4'])
        if x3>x4:
            xt = x3
            yt = y3
            x3 = x4
            y3 = y4
            x4 = xt
            y4 = yt

        # width = int(request.query_params['width'])
        # height = int(width * (math.sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3))) / math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))
        #
        height = 800
        width = int(height * math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) / math.sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)))


        now = datetime.datetime.now()
        source_image_name = '{}.jpg'.format(now.strftime('%Y%m%d_%H%M%S'))
        media_dir = settings.MEDIA_ROOT
        # 通过 picurl 获取图片
        image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', 'rectify')
        if not tf.gfile.Exists(image_dir):
            tf.gfile.MakeDirs(image_dir)
        source_image_path = os.path.join(image_dir, source_image_name)
        urllib.request.urlretrieve(picurl, source_image_path)

        dest_image_name = 'rectify_{}.jpg'.format(now.strftime('%Y%m%d_%H%M%S'))
        dest_image_path = os.path.join(image_dir, dest_image_name)
        img = cv2.imread(source_image_path)
        rows, cols = img.shape[:2]
        # 原图中书本的四个角点
        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # 变换后分别在左上、右上、左下、右下四个点
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(img, M, (width, height))
        cv2.imwrite(dest_image_path,dst)
        ret = {'returl':os.path.join(settings.MEDIA_URL, settings.DETECT_DIR_NAME, 'shelf', 'rectify',dest_image_name)}

        return Response(goods.util.wrap_ret(ret), status=status.HTTP_200_OK)

class ShelfImageViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = ShelfImage.objects.order_by('-id')
    serializer_class = ShelfImageSerializer


class ShelfGoodsViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                        mixins.UpdateModelMixin,mixins.DestroyModelMixin,
                        viewsets.GenericViewSet):
    queryset = ShelfGoods.objects.order_by('-id')
    serializer_class = ShelfGoodsSerializer

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()

        old_upc = instance.upc

        instance.score2 = 1.0
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        upc = serializer.instance.upc
        if upc != '':
            sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample', '{}'.format(serializer.instance.shopid),'{}'.format(serializer.instance.shelfid))
            if not tf.gfile.Exists(sample_dir):
                tf.gfile.MakeDirs(sample_dir)
            old_sample_path = os.path.join(sample_dir, '{}_{}.jpg'.format(old_upc, serializer.instance.pk))
            if os.path.isfile(old_sample_path):
                # 删除原来的样本
                os.remove(old_sample_path)

            # 添加新样本
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', '{}_{}'.format(serializer.instance.shopid,serializer.instance.shelfid))
            image_path = os.path.join(image_dir, serializer.instance.shelf_image.image_name)
            image = PILImage.open(image_path)
            sample_image = image.crop((serializer.instance.xmin, serializer.instance.ymin, serializer.instance.xmax, serializer.instance.ymax))
            sample_image_path = os.path.join(sample_dir, '{}_{}.jpg'.format(serializer.instance.upc, serializer.instance.pk))
            sample_image.save(sample_image_path, 'JPEG')

        return Response(serializer.data)


    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample',
                                  '{}'.format(instance.shopid),
                                  '{}'.format(instance.shelfid))
        # 删除原来的样本
        old_sample_path = os.path.join(sample_dir, '{}_{}.jpg'.format(instance.upc, instance.pk))
        if os.path.isfile(old_sample_path):
            os.remove(old_sample_path)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
