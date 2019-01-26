from rest_framework import serializers
from track.models import BasePoint, EntranceArea, Recognition, TransferRecognition

class BasePointSerializer(serializers.ModelSerializer):
  class Meta:
    model = BasePoint
    fields = ('pk', 'shopid', 'deviceid', 'x', 'y', 'z')
    read_only_fields = ('create_time',)

class EntranceAreaSerializer(serializers.ModelSerializer):
  class Meta:
    model = EntranceArea
    fields = ('pk', 'shopid', 'deviceid', 'x', 'y', 'w', 'h')
    read_only_fields = ('create_time',)

class RecognitionSerializer(serializers.ModelSerializer):
  class Meta:
    model = Recognition
    fields = ('pk', 'shopid', 'deviceid', 'cid', 'sid', 'x', 'y', 'z')
    read_only_fields = ('r_sid','create_time',)
