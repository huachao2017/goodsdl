from rest_framework import serializers
from .models import Image, ImageClass, Goods, GoodsClass, ProblemGoods, TrainImage, ActionLog

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

class ActionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActionLog
        fields = ('pk', 'action', 'traintype', 'desc', 'param', 'create_time')
        read_only_fields = ( 'param', 'create_time',)

