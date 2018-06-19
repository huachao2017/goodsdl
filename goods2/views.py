import logging
import shutil

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings

from .serializers import *

logger = logging.getLogger("django")


class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100


class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # TODO detect

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class ImageGroundTruthViewSet(DefaultMixin, mixins.CreateModelMixin,viewsets.GenericViewSet):
    queryset = ImageGroundTruth.objects.order_by('-id')
    serializer_class = ImageGroundTruthSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        images = Image.objects.filter(identify=serializer.instance.identify)
        for image in images:
            image.image_ground_truth=serializer.instance
            image.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TrainImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                        mixins.DestroyModelMixin, viewsets.GenericViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # add or update TrainUpc
        try:
            train_upc = TrainUpc.objects.get(upc=serializer.instance.upc)
            train_upc.cnt += 1
            train_upc.save()
        except TrainUpc.DoesNotExist as e:
            TrainUpc.objects.create(
                upc=serializer.instance.upc,
                cnt=1,
            )

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        train_upc = TrainUpc.objects.get(upc=instance.upc)
        train_upc.cnt -= 1
        train_upc.save()

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


class TrainUpcViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = TrainUpc.objects.order_by('-id')
    serializer_class = TrainUpcSerializer
