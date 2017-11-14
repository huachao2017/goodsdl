from rest_framework import serializers
from .models import Image, Goods, TrainImage, ActionLog

class GoodsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Goods
        fields = ('pk', 'image', 'class_type', 'score', 'upc', 'xmin', 'ymin', 'xmax', 'ymax')

class ImageSerializer(serializers.ModelSerializer):
    image_goods = GoodsSerializer(many=True, read_only=True)
    class Meta:
        model = Image
        fields = ('pk', 'source', 'image_goods')

class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'source', 'upc', 'name', 'xmin', 'ymin', 'xmax', 'ymax', 'create_time')
        read_only_fields = ('create_time',)

class ActionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActionLog
        fields = ('pk', 'action', 'desc', 'param', 'create_time')
        read_only_fields = ( 'param', 'create_time',)

