import logging
import os
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from django.conf import settings
from . import common
from . import convert_goods

from .serializers import *

logger = logging.getLogger("django")

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class TrainActionViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer


class TrainModelViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer
