from django.shortcuts import render
from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend

from track.serializers import *
# Create your views here.
class DefaultMixin:
  paginate_by = 25
  paginate_by_param = 'page_size'
  max_paginate_by = 100


class BasePointViewSet(DefaultMixin, viewsets.ModelViewSet):
  queryset = BasePoint.objects.order_by('-id')
  serializer_class = BasePointSerializer


class EntranceAreaViewSet(DefaultMixin, viewsets.ModelViewSet):
  queryset = EntranceArea.objects.order_by('-id')
  serializer_class = EntranceAreaSerializer


class RecognitionViewSet(DefaultMixin, viewsets.ModelViewSet):
  queryset = Recognition.objects.order_by('-id')
  serializer_class = RecognitionSerializer


  def create(self, request, *args, **kwargs):
    serializer = self.get_serializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    self.perform_create(serializer)
    headers = self.get_success_headers(serializer.data)

    # FIXME 识别功能写在这里

    return Response({'sid':0}, status=status.HTTP_201_CREATED, headers=headers)
