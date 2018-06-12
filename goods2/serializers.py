from rest_framework import serializers
from .models import Image, TrainImage, TrainUpc, TrainAction, TrainActionUpcs

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'source', 'create_time')
        read_only_fields = ('create_time',)

class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'deviceid', 'source', 'upc', 'source_from', 'create_time')
        read_only_fields = ('create_time',)

class TrainUpcSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainUpc
        fields = ('pk', 'upc', 'cnt')
        read_only_fields = ('create_time', 'update_time',)

class TrainActionUpcsSerializer(serializers.ModelSerializer):
    train_upc = TrainUpcSerializer(many=False, read_only=True)
    class Meta:
        model = TrainActionUpcs
        fields = ('pk', 'train_upc')

class TrainActionSerializer(serializers.ModelSerializer):
    upcs = TrainActionUpcsSerializer(many=True, read_only=True)
    class Meta:
        model = TrainAction
        fields = ('pk', 'train_ip', 'action', 'model_name', 'desc', 'upcs')
        read_only_fields = (  'create_time', 'update_time',)
