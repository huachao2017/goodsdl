from rest_framework import serializers
from goods2.models import Image, ImageGroundTruth, ImageResult, TrainImage, TrainUpc, TrainAction, TrainModel, TrainActionUpcs, TrainActionDevices, ImageTrainModel, TaskLog


class ImageResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageResult
        fields = ('pk', 'image', 'upc', 'score')


class ImageGroundTruthSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageGroundTruth
        fields = ('pk', 'upc', 'identify', 'create_time')
        read_only_fields = ('create_time',)

class ImageTrainModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageTrainModel
        fields = ('pk', 'train_model', 'image')


class ImageSerializer(serializers.ModelSerializer):
    image_results = ImageResultSerializer(many=True, read_only=True)
    image_ground_truth = ImageGroundTruthSerializer(many=False, read_only=True)
    train_models = ImageTrainModelSerializer(many=False, read_only=True)

    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'identify', 'source', 'image_ground_truth', 'image_results', 'train_models', 'create_time')
        read_only_fields = ('create_time',)


class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'deviceid', 'source', 'upc', 'source_from', 'score', 'source_image', 'create_time', 'update_time')
        read_only_fields = ('source_image', 'create_time', 'update_time')


class TrainUpcSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainUpc
        fields = ('pk', 'upc', 'cnt', 'create_time', 'update_time')
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
        fields = ('pk', 'model_path', 'checkpoint_prefix', 'precision', 'create_time')
        read_only_fields = ('model_path', 'checkpoint_prefix', 'precision', 'create_time', )


class TrainActionSerializer(serializers.ModelSerializer):
    upcs = TrainActionUpcsSerializer(many=True, read_only=True)
    devices = TrainActionDevicesSerializer(many=True, read_only=True)

    class Meta:
        model = TrainAction
        fields = ('pk', 'action', 'f_model', 'desc', 'train_path', 'create_time', 'update_time', 'complete_time', 'train_cnt', 'validation_cnt', 'upcs', 'devices')
        read_only_fields = ('train_path', 'create_time', 'update_time', 'complete_time', 'train_cnt', 'validation_cnt', 'upcs', 'devices')

class TaskLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaskLog
        fields = ('pk', 'name', 'ip', 'message', 'state', 'create_time', 'update_time')
