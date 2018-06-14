import logging
import os
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

class TrainActionViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create action:{}'.format(serializer.instance.action))

        serializer.instance.train_path = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))
        # 数据准备
        names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_train(serializer.instance, serializer.instance.action)
        # 更新数据
        # 'upcs'
        for upc in names_to_labels:
            train_upc = TrainUpc.objects.get(upc=upc)
            TrainActionUpcs.objects.create(
                train_action_id=serializer.instance.pk,
                train_upc=train_upc,
                upc=upc,
                cnt=train_upc.cnt,
            )
        serializer.instance.train_cnt = len(training_filenames)
        serializer.instance.validation_cnt = len(validation_filenames)
        # 'devcice'
        if serializer.instance.action == 'TC':
            pass

        serializer.instance.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TrainModelViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer
