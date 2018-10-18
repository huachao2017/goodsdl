from rest_framework import serializers
from goods2.models import Deviceid, DeviceidPrecision, DeviceidTrain, Image, ImageGroundTruth, ImageResult, TrainImage, TrainAction, TrainModel, TrainActionUpcs, ImageTrainModel, TaskLog


class DeviceidPrecisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceidPrecision
        fields = ('pk', 'truth_rate', 'precision', 'create_time')
        read_only_fields = ('create_time',)

class DeviceidSerializer(serializers.ModelSerializer):
    device_precisions = DeviceidPrecisionSerializer(many=True, read_only=True)
    class Meta:
        model = Deviceid
        fields = ('pk', 'deviceid', 'state', 'device_precisions', 'create_time', 'update_time', 'commercial_time')
        read_only_fields = ('create_time', 'update_time', 'commercial_time')

class DeviceidTrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceidTrain
        fields = ('pk', 'deviceid','create_time')
        read_only_fields = ('create_time',)

class ImageResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageResult
        fields = ('pk', 'image', 'upc', 'score')


class ImageGroundTruthSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageGroundTruth
        fields = ('pk', 'deviceid', 'upc', 'identify', 'cnt', 'truth_rate', 'precision', 'create_time')
        read_only_fields = ('cnt', 'truth_rate', 'precision', 'create_time',)

class ImageTrainModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageTrainModel
        fields = ('pk', 'train_model', 'image')


class UserImageSerializer(serializers.ModelSerializer):
    image_results = ImageResultSerializer(many=True, read_only=True)
    image_ground_truth = ImageGroundTruthSerializer(many=False, read_only=True)
    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'source', 'upc', 'image_ground_truth', 'image_results', 'create_time')
        read_only_fields = ('create_time',)


class ImageSerializer(serializers.ModelSerializer):
    image_results = ImageResultSerializer(many=True, read_only=True)
    image_ground_truth = ImageGroundTruthSerializer(many=False, read_only=True)
    train_models = ImageTrainModelSerializer(many=True, read_only=True)

    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'identify', 'source', 'image_ground_truth', 'image_results', 'train_models', 'is_train', 'is_hand','create_time')
        read_only_fields = ('create_time',)


class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'deviceid', 'source', 'upc', 'source_from', 'score', 'source_image', 'create_time', 'update_time')
        read_only_fields = ('source_image', 'create_time', 'update_time')


class TrainActionUpcsSerializer(serializers.ModelSerializer):

    class Meta:
        model = TrainActionUpcs
        fields = ('pk', 'upc', 'cnt')


class TrainModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainModel
        fields = ('pk', 'model_path', 'checkpoint_step', 'precision', 'create_time')
        read_only_fields = ('model_path', 'checkpoint_step', 'precision', 'create_time', )


class TrainActionSerializer(serializers.ModelSerializer):
    upcs = TrainActionUpcsSerializer(many=True, read_only=True)

    class Meta:
        model = TrainAction
        fields = ('pk', 'action', 'state', 'deviceid', 'f_model', 'desc', 'train_path', 'create_time', 'update_time', 'ip', 'train_command', 'eval_command', 'complete_time', 'train_cnt', 'validation_cnt', 'upcs')
        read_only_fields = ('train_path', 'create_time', 'update_time', 'ip', 'train_command', 'eval_command', 'complete_time', 'train_cnt', 'validation_cnt', 'upcs')

class TaskLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaskLog
        fields = ('pk', 'name', 'ip', 'message', 'state', 'create_time', 'update_time')
