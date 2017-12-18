from rest_framework import serializers
from .models import Image, ImageClass, Goods, GoodsClass, ProblemGoods, TrainImage, TrainImageClass, TrainAction, ExportAction, StopTrainAction

class GoodsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Goods
        fields = ('pk', 'image', 'class_type', 'score1', 'score2', 'upc')

class ProblemGoodsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProblemGoods
        fields = ('pk', 'image', 'index', 'class_type_0', 'class_type_1', 'class_type_2', 'class_type_3', 'class_type_4', 'score_0', 'score_1', 'score_2', 'score_3', 'score_4')

class ImageSerializer(serializers.ModelSerializer):
    image_problem_goods = ProblemGoodsSerializer(many=True, read_only=True)
    image_goods = GoodsSerializer(many=True, read_only=True)
    class Meta:
        model = Image
        fields = ('pk', 'deviceid', 'source', 'image_goods', 'image_problem_goods', 'create_time')
        read_only_fields = ('create_time',)

class GoodsClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = GoodsClass
        fields = ('pk', 'image_class', 'class_type', 'score', 'upc')

class ImageClassSerializer(serializers.ModelSerializer):
    image_class_goods = GoodsClassSerializer(many=True, read_only=True)
    class Meta:
        model = ImageClass
        fields = ('pk', 'deviceid', 'source', 'image_class_goods', 'create_time')
        read_only_fields = ('create_time',)

class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ('pk', 'deviceid', 'traintype', 'source', 'upc', 'name', 'xmin', 'ymin', 'xmax', 'ymax', 'create_time')
        read_only_fields = ('create_time',)

class TrainImageClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImageClass
        fields = ('pk', 'deviceid', 'traintype', 'source', 'upc', 'name', 'create_time')
        read_only_fields = ('create_time',)

class ExportActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExportAction
        fields = ('pk', 'train_action', 'checkpoint_prefix', 'backup_postfix', 'param', 'create_time')
        read_only_fields = ( 'checkpoint_prefix', 'backup_postfix', 'param', 'create_time',)

class StopTrainActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StopTrainAction
        fields = ('pk', 'train_action', 'param', 'create_time')
        read_only_fields = ( 'param', 'create_time',)

class TrainActionSerializer(serializers.ModelSerializer):
    export_actions = ExportActionSerializer(many=True, read_only=True)
    stop_train_actions = StopTrainActionSerializer(many=True, read_only=True)
    class Meta:
        model = TrainAction
        fields = ('pk', 'action', 'traintype', 'desc', 'param', 'export_actions', 'stop_train_actions', 'create_time')
        read_only_fields = ( 'param', 'create_time',)
