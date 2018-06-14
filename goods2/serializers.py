from rest_framework import serializers
from .models import Image, ImageGroundTruth, ImageResult, TrainImage, TrainUpc, TrainAction, TrainModel, TrainActionUpcs, TrainActionDevices


class ImageResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageResult
        fields = ('pk', 'image', 'upc', 'score')


class ImageGroundTruthSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageGroundTruth
        fields = ('pk', 'upc', 'identify')
        read_only_fields = ('create_time',)


class ImageSerializer(serializers.ModelSerializer):
    image_results = ImageResultSerializer(many=True, read_only=True)
    image_ground_truth = ImageGroundTruthSerializer(many=False, read_only=True)

    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'identify', 'source', 'image_ground_truth', 'image_results')
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

    class Meta:
        model = TrainActionUpcs
        fields = ('pk', 'upc', 'cnt')


class TrainActionDevicesSerializer(serializers.ModelSerializer):

    class Meta:
        model = TrainActionDevices
        fields = ('pk', 'train_deviceid', 'train_action')


class TrainModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainModel
        fields = ('pk', )
        read_only_fields = ('model_path', 'checkpoint_prefix', 'precision', 'create_time', )


class TrainActionSerializer(serializers.ModelSerializer):
    upcs = TrainActionUpcsSerializer(many=True, read_only=True)
    devices = TrainActionDevicesSerializer(many=True, read_only=True)

    class Meta:
        model = TrainAction
        fields = ('pk', 'action', 'model_name', 'f_model', 'desc')
        read_only_fields = ('train_ip', 'train_path', 'create_time', 'update_time', 'train_cnt', 'validation_cnt', 'upcs', 'devices')
