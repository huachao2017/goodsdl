import logging
import json
import os
import datetime
import urllib.request
import numpy as np
import tensorflow as tf

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from face.serializers import *
from face.dl import facedetection
logger = logging.getLogger("face")
from django.conf import settings

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

class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin detect face:')

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        now = datetime.datetime.now()
        image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'face', now.strftime('%Y%m%d'))
        if not tf.gfile.Exists(image_dir):
            tf.gfile.MakeDirs(image_dir)
        image_path = os.path.join(image_dir, '{}.jpg'.format(now.strftime('%H%M%S')))
        logger.info(image_path)
        urllib.request.urlretrieve(serializer.instance.picurl, image_path)

        detector = facedetection.FaceDetectorFactory.get_static_detector()
        index = detector.detect(image_path)
        serializer.instance.index = json.dumps(index,cls=NumpyEncoder)
        serializer.instance.image_path = image_path
        serializer.instance.save()

        logger.info('end detect face:')
        return Response(serializer.instance.index, status=status.HTTP_201_CREATED, headers=headers)
