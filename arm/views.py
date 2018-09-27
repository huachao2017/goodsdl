import logging
import json
import os
import time
import urllib.request
import numpy as np
import shutil
from PIL import Image

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from arm.serializers import *
from tradition.edge.contour_detect_3d import Contour_3d
from goods2.models import TrainImage
logger = logging.getLogger("django")
from django.conf import settings
from dl import imagedetection_arm10

from goods2 import common as goods2_common
from arm import common

# Create your views here.
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

class ArmImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = ArmImage.objects.order_by('-id')
    serializer_class = ArmImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin detect arm:')
        time0 = time.time()

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        detect = Contour_3d(serializer.instance.rgb_source.path, serializer.instance.depth_source.path, serializer.instance.table_z-10)
        min_rectes, z, boxes = detect.find_contour(False)

        time1 = time.time()
        if len(min_rectes)>0:
            detector = imagedetection_arm10.ImageDetectorFactory.get_static_detector()
            step1_min_score_thresh = .5
            types = detector.detect(serializer.instance.rgb_source.path, boxes)

        ret = []
        index = 0
        for min_rect in min_rectes:
            logger.info('center: %d,%d; w*h:%d,%d; theta:%d; z:%d, boxes: x1:%d, y1:%d, x2:%d, y2:%d, type:%d' % (
            min_rect[0][0], min_rect[0][1], min_rect[1][0], min_rect[1][1], min_rect[2], z[index], boxes[index][0],
            boxes[index][1], boxes[index][2], boxes[index][3], types[index]))
            one = {
                'x': min_rect[0][0],
                'y': min_rect[0][1],
                'z': z[index],
                'angle': min_rect[2],
                'box': {
                    'xmin':boxes[index][0],
                    'ymin': boxes[index][1],
                    'xmax': boxes[index][2],
                    'ymax': boxes[index][3],
                },
                'upc': types[index],
            }
            ret.append(one)
            index += 1
        serializer.instance.result = json.dumps(ret, cls=NumpyEncoder)
        serializer.instance.save()

        time2 = time.time()
        logger.info('end detect arm: %.2f, %.2f, %.2f' % (time2-time0, time1-time0, time2-time1))
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)


class ArmTrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = ArmTrainImage.objects.order_by('-id')
    serializer_class = ArmTrainImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        detect = Contour_3d(serializer.instance.rgb_source.path, serializer.instance.depth_source.path, serializer.instance.table_z-10)
        min_rectes, z, boxes = detect.find_contour(False)

        if len(boxes) == 1:

            train_source = '{}/{}/{}/{}'.format(goods2_common.get_dataset_dir(), serializer.instance.deviceid, serializer.instance.upc,
                                                'arm_' + os.path.basename(serializer.instance.rgb_source.path))
            train_source_dir = '{}/{}/{}'.format(goods2_common.get_dataset_dir(True), serializer.instance.deviceid,
                                                 serializer.instance.upc)

            import tensorflow as tf
            if not tf.gfile.Exists(train_source_dir):
                tf.gfile.MakeDirs(train_source_dir)
            train_source_path = '{}/{}'.format(train_source_dir, 'arm_' + os.path.basename(serializer.instance.rgb_source.path))
            image = Image.open(serializer.instance.rgb_source.path)
            newimage = image.crop((boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
            newimage.save(train_source_path, 'JPEG')
            TrainImage.objects.create(
                deviceid=serializer.instance.deviceid,
                source=train_source,
                upc=serializer.instance.upc,
                source_from=3,
                score=1.0,
            )
        else:
            raise ValueError()

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
