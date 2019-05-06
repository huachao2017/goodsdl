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

from dl import shelfdetection
# from dl.old import imagedetection
from .serializers import *
import tensorflow as tf

logger = logging.getLogger("django")

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
        now = datetime.datetime.now()
        image_name = '{}.jpg'.format(now.strftime('%Y%m%d_%H%M%S'))
        shelf_image = ShelfImage.objects.create(
            shopid = shopid,
            shelfid = shelfid,
            picurl = picurl,
            image_name = image_name,
        )

        ret = []
        export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by(
            '-update_time')[:1]

        if len(export1s) > 0:
            detector = shelfdetection.ShelfDetectorFactory.get_static_detector(export1s[0].pk)
            step1_min_score_thresh = .5
            media_dir = settings.MEDIA_ROOT
            # 通过 picurl 获取图片
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', '{}_{}'.format(shopid,shelfid))
            if not tf.gfile.Exists(image_dir):
                tf.gfile.MakeDirs(image_dir)
            image_path = os.path.join(image_dir, image_name)
            logger.info(image_path)
            urllib.request.urlretrieve(picurl, image_path)
            detect_ret, aiinterval, visual_image_path = detector.detect(image_path, step1_min_score_thresh=step1_min_score_thresh,totol_level = tlevel)

            for one_box in detect_ret:
                shelf_goods = ShelfGoods.objects.create(
                    shelf_image_id = shelf_image.pk,
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

        instance.score2 = 1.0
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        upc = serializer.instance.upc
        if upc != '':
            sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample', '{}'.format(serializer.instance.shelf_image.shopid),'{}'.format(serializer.instance.shelf_image.shelfid))
            if not tf.gfile.Exists(sample_dir):
                tf.gfile.MakeDirs(sample_dir)
            old_sample_path = os.path.join(sample_dir,'{}.jpg'.format(serializer.instance.pk))
            if os.path.isfile(old_sample_path):
                # 删除原来的样本
                os.remove(old_sample_path)

            # 添加新样本
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', '{}_{}'.format(serializer.instance.shelf_image.shopid,serializer.instance.shelf_image.shelfid))
            image_path = os.path.join(image_dir, serializer.instance.shelf_image.image_name)
            image = PILImage.open(image_path)
            sample_image = image.crop((serializer.instance.xmin, serializer.instance.ymin, serializer.instance.xmax, serializer.instance.ymax))
            sample_image_path = os.path.join(sample_dir, '{}.jpg'.format(serializer.instance.pk))
            sample_image.save(sample_image_path, 'JPEG')

        return Response(serializer.data)


    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample',
                                  '{}'.format(instance.shelf_image.shopid),
                                  '{}'.format(instance.shelf_image.shelfid))
        # 删除原来的样本
        old_sample_path = os.path.join(sample_dir, '{}.jpg'.format(instance.pk))
        if os.path.isfile(old_sample_path):
            os.remove(old_sample_path)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
