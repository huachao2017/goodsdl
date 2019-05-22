import logging
import json
import os
import time
import urllib.request
import numpy as np
import shutil
from PIL import Image
import tensorflow as tf

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from arm.serializers import *
from tradition.edge.contour_detect_3d import Contour_3d
from tradition.cylinder.cylinder_detect_3d import Cylinder_3d
from goods2.models import TrainImage, TrainAction, TrainModel
logger = logging.getLogger("django")
from arm.dl import imagedetection

from goods2 import common as goods2_common
from arm import common
import datetime


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

        z_deviation = 0 #10 # 60 # 10
        detect = Contour_3d(serializer.instance.rgb_source.path, serializer.instance.depth_source.path, serializer.instance.table_z-z_deviation)
        min_rectes, z, boxes = detect.find_contour(False)

        time1 = time.time()
        detector = None
        image = None
        tmp_dir = None
        if len(min_rectes)>0:
            # 准备阶段
            last_normal_train_qs = TrainAction.objects.filter(state=goods2_common.TRAIN_STATE_COMPLETE).filter(
                deviceid='100000').exclude(action='TC').order_by('-id')
            if len(last_normal_train_qs) > 0:
                last_train = last_normal_train_qs[0]
                last_normal_train_model = \
                TrainModel.objects.filter(train_action_id=last_train.pk).exclude(model_path='').order_by('-id')[0]
                detector = imagedetection.ImageDetectorFactory.get_static_detector(
                    last_normal_train_model)
                image = Image.open(serializer.instance.rgb_source.path)
                tmp_dir = '{}/tmp'.format(os.path.dirname(serializer.instance.rgb_source.path))
                if not tf.gfile.Exists(tmp_dir):
                    tf.gfile.MakeDirs(tmp_dir)
        else:
            deleted_dir = '{}/deleted'.format(os.path.dirname(serializer.instance.rgb_source.path))
            if not tf.gfile.Exists(deleted_dir):
                tf.gfile.MakeDirs(deleted_dir)
            deleted_rgb = '{}/{}'.format(deleted_dir, os.path.basename(serializer.instance.rgb_source.path))
            shutil.move(serializer.instance.rgb_source.path, deleted_rgb)
            deleted_depth = '{}/{}'.format(deleted_dir, os.path.basename(serializer.instance.depth_source.path))
            shutil.move(serializer.instance.depth_source.path, deleted_depth)

        ret = []
        index = 0
        for min_rect in min_rectes:
            # 检测类型
            upcs = [0,]
            scores = [0,]
            if detector is not None:
                oneimage = image.crop((boxes[index][0], boxes[index][1], boxes[index][2], boxes[index][3]))
                one_image_path = os.path.join(tmp_dir,'%d_%d.jpg' % (serializer.instance.pk, index))
                oneimage.save(one_image_path, 'JPEG')
                upcs, scores = detector.detect(one_image_path)
                shutil.move(one_image_path,os.path.join(tmp_dir,'%d_%d_%s_%.2f.jpg' % (serializer.instance.pk, index, upcs[0], scores[0])))
            logger.info('center: %d,%d; w*h:%d,%d; theta:%d; z:%d, boxes: x1:%d, y1:%d, x2:%d, y2:%d, type:%s, score:%.2f' % (
            min_rect[0][0], min_rect[0][1], min_rect[1][0], min_rect[1][1], min_rect[2], z[index], boxes[index][0],
            boxes[index][1], boxes[index][2], boxes[index][3], upcs[0], scores[0]))
            one = {
                'x': min_rect[0][0],
                'y': min_rect[0][1],
                'z': z[index]+z_deviation,
                'w': min_rect[1][0],
                'h': min_rect[1][1],
                'angle': min_rect[2],
                'box': {
                    'xmin':boxes[index][0],
                    'ymin': boxes[index][1],
                    'xmax': boxes[index][2],
                    'ymax': boxes[index][3],
                },
                'upc': upcs[0],
            }
            ret.append(one)
            index += 1
        serializer.instance.result = json.dumps(ret, cls=NumpyEncoder)
        serializer.instance.save()

        time2 = time.time()
        logger.info('end detect arm: %.2f, %.2f, %.2f' % (time2-time0, time1-time0, time2-time1))
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)


class ArmCylinderImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = ArmImage.objects.order_by('-id')
    serializer_class = ArmImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin detect arm cylinder:')
        time0 = time.time()

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        z_deviation = 0  # 10 # 60 # 10
        detect = Cylinder_3d(serializer.instance.rgb_source.path, serializer.instance.depth_source.path,
                             serializer.instance.table_x,serializer.instance.table_y,
                            serializer.instance.table_z - z_deviation)
        alpha, beta, x, y, z = detect.find_cylinder()

        ret = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': alpha,
            'beta': beta
        }

        serializer.instance.result = json.dumps(ret, cls=NumpyEncoder)
        serializer.instance.save()


        logger.info('end detect arm: x=%d, y=%d, z=%d, alpha=%.2f, beta=%.2f' % (x,y,z,alpha,beta))
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

        if len(boxes) >= 1:

            train_source = '{}/{}/{}/{}'.format(goods2_common.get_dataset_dir(), serializer.instance.deviceid, serializer.instance.upc,
                                                'arm_' + os.path.basename(serializer.instance.rgb_source.path))
            train_source_dir = '{}/{}/{}'.format(goods2_common.get_dataset_dir(True), serializer.instance.deviceid,
                                                 serializer.instance.upc)

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
