import logging
import json
import os
import datetime
import urllib.request
import numpy as np

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from arm.serializers import *
from tradition.edge.contour_detect_3d import find_contour
logger = logging.getLogger("arm")
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

class ArmImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = ArmImage.objects.order_by('-id')
    serializer_class = ArmImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin detect arm:')

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        now = datetime.datetime.now()
        min_rectes, z, boxes = find_contour(serializer.instance.rgb_source.path, serializer.instance.depth_source.path, serializer.instance.table_z)
        ret = []
        index = 0
        for min_rect in min_rectes:
            logger.info('center: %d,%d; w*h:%d,%d; theta:%d; z:%d' % (
            min_rect[0][0], min_rect[0][1], min_rect[1][0], min_rect[1][1], min_rect[2], z[index]))
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
                'upc': "1", # TODO
            }
            ret.append(one)
            index += 1
        serializer.instance.result = json.dumps(ret, cls=NumpyEncoder)
        serializer.instance.save()

        logger.info('end detect arm:')
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)
