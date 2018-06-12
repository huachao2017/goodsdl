from rest_framework import serializers
from .models import Image, TrainImage, TrainAction, TrainActionUpcs

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'source', 'create_time')
        read_only_fields = ('create_time',)

class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'deviceid', 'source', 'upc', 'create_time')
        read_only_fields = ('create_time',)

class TrainActionUpcsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainActionUpcs
        fields = ('pk', 'upc')

class TrainActionSerializer(serializers.ModelSerializer):
    upcs = TrainActionUpcsSerializer(many=True, read_only=True)
    class Meta:
        model = TrainAction
        fields = ('pk', 'ip', 'action', 'model_name', 'desc', 'upcs')
        read_only_fields = (  'create_time', 'update_time',)
