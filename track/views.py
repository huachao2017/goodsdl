import math

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

    cur_recognition = serializer.instance
    r_sid = cur_recognition.sid
    if r_sid > 0:
      return Response({'sid': r_sid}, status=status.HTTP_201_CREATED, headers=headers)

    r_sid = 0

    # 查询是否分配入门sid
    entrance_area = EntranceArea.objects.filter(shopid=cur_recognition.shopid).filter(deviceid=cur_recognition.deviceid).order_by('-id')[0]
    if entrance_area is not None:
      if abs(entrance_area.x-cur_recognition.x)<entrance_area.w/2 and abs(entrance_area.y-cur_recognition.y)<entrance_area.h/2:
        # 分配一个新的id，
        # FIXME 需要考虑重入问题
        last_recognition = Recognition.objects.filter(shopid=cur_recognition.shopid).filter(deviceid=cur_recognition.deviceid).filter(r_sid__gt=0).order_by('-id')[0]
        if last_recognition is None:
          r_sid = 1
        else:
          r_sid = last_recognition.r_sid + 1

        cur_recognition.r_sid = r_sid
        cur_recognition.save()
        return Response({'sid': r_sid}, status=status.HTTP_201_CREATED, headers=headers)

    # 查询其他有sid设备2s内的位置
    last_recognition_qs = Recognition.objects.filter(shopid=cur_recognition.shopid).exclude(deviceid=cur_recognition.deviceid).filter(sid__gt=0).order_by('-id')[:10]
    deviceid_to_recognition = {}
    for last_recognition in last_recognition_qs:
      delta = cur_recognition.create_time - last_recognition.create_time
      if delta.seconds > 2:
        break
      if last_recognition.deviceid not in deviceid_to_recognition:
        deviceid_to_recognition[last_recognition.deviceid] = last_recognition

    # 获取BasePoint
    deviceid_to_base_point = {}
    base_point_qs = BasePoint.objects.filter(shopid=cur_recognition.shopid).order_by('-id')[:10]
    for base_point in base_point_qs:
      if base_point.deviceid not in deviceid_to_base_point:
        deviceid_to_base_point[base_point.deviceid] = base_point

    # 获取距离
    deviceid_to_distance = {}
    cur_point_x = cur_recognition.x - deviceid_to_base_point[cur_recognition.deviceid].x
    cur_point_y = cur_recognition.y - deviceid_to_base_point[cur_recognition.deviceid].y
    for deviceid in deviceid_to_recognition:
      point_x = deviceid_to_recognition[deviceid].x - deviceid_to_base_point[deviceid].x
      point_y = deviceid_to_recognition[deviceid].y - deviceid_to_base_point[deviceid].y
      distance = math.sqrt((point_x-cur_point_x)(point_x-cur_point_x)+(point_y-cur_point_y)(point_y-cur_point_y))
      deviceid_to_distance[deviceid] = distance

    # 判断最近距离
    transfer_deviceid = None
    min_distance = 1000
    for deviceid in deviceid_to_distance:
      if deviceid_to_distance[deviceid] < min_distance:
        min_distance = deviceid_to_distance[deviceid]
        transfer_deviceid = deviceid

    if transfer_deviceid is not None:
      r_sid = deviceid_to_recognition[transfer_deviceid].sid

      TransferRecognition.objects.create(
        recognition_id=cur_recognition.pk,
        transfer_deviceid=transfer_deviceid,
        distance=min_distance,
      )
      cur_recognition.r_sid = r_sid
      cur_recognition.save()
      return Response({'sid':r_sid}, status=status.HTTP_201_CREATED, headers=headers)

    return Response({'sid':r_sid}, status=status.HTTP_201_CREATED, headers=headers)
