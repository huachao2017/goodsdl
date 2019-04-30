import datetime
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request

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
        import tensorflow as tf

        shopid = request.query_params['shopid']
        shelfid = request.query_params['shelfid']
        if 'tlevel' in request.query_params:
            tlevel = request.query_params['tlevel']
        else:
            tlevel = 6
        picurl = request.query_params['picurl']
        shelf_image = ShelfImage.objects.create(
            shopid = shopid,
            shelfid = shelfid,
            picurl = picurl,
        )

        ret = []
        export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by(
            '-update_time')[:1]

        if len(export1s) > 0:
            detector = shelfdetection.ShelfDetectorFactory.get_static_detector(export1s[0].pk)
            step1_min_score_thresh = .5
            media_dir = settings.MEDIA_ROOT
            # 通过 picurl 获取图片
            now = datetime.datetime.now()
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf', '{}_{}'.format(shopid,shelfid))
            if not tf.gfile.Exists(image_dir):
                tf.gfile.MakeDirs(image_dir)
            image_path = os.path.join(image_dir, '{}.jpg'.format(now.strftime('%Y%m%d_%H%M%S')))
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
                   viewsets.GenericViewSet):
    queryset = ShelfGoods.objects.order_by('-id')
    serializer_class = ShelfGoodsSerializer
