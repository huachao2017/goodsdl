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
        ret = []
        serializer.instance.result = json.dumps(ret, cls=NumpyEncoder)
        serializer.instance.save()

        logger.info('end detect arm:')
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)
