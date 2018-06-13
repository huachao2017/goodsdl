import logging
import os
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response

from .serializers import *

logger = logging.getLogger("django")

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        dir = os.path.dirname(instance.source.path)
        if os.path.isfile(instance.source.path):
            os.remove(instance.source.path)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

class TrainActionViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer
