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
        if serializer.instance.action == 'TA':
            # 数据准备
            names_to_labels, training_filenames, _ = convert_goods.prepare_train_TA(serializer.instance)

            #更新数据
            # 'upcs'
            for upc in names_to_labels:
                TrainActionUpcs.objects.create(
                    train_action_id=serializer.instance.pk,
                    train_upc=TrainUpc.objects.get(upc=upc),
                )

            batch_size = 8 if serializer.instance.model_name == 'nasnet_large' else 64
            # 'max_step'
            serializer.instance.max_step = int(len(training_filenames) * 100 / batch_size)  # 设定最大训练次数，每个样本进入网络100次，测试验证200次出现过拟合
            if serializer.instance.max_step < 20000:
                serializer.instance.max_step = 20000  # 小样本需要增加训练次数
        elif serializer.instance.action == 'TF':
            pass
        elif serializer.instance.action == 'TC':
            # 'max_step', 'upcs', 'devices'
            pass

        serializer.instance.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TrainModelViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer
