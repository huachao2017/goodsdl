from rest_framework import serializers
from .models import Image, ImageClass, Goods, GoodsClass, ProblemGoods, TrainImage, TrainImageOnly, TrainImageClass, TrainAction, ExportAction, StopTrainAction, RfidImageCompareAction, RfidTransaction, TransactionMetrix, RfidGoods, DatasetAction

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
        fields = ('pk', 'deviceid', 'source', 'image_goods', 'image_problem_goods', 'lastinterval', 'aiinterval', 'create_time')
        read_only_fields = ('aiinterval', 'create_time',)

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

class TrainImageOnlySerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImageOnly
        fields = ('pk', 'deviceid', 'traintype', 'source', 'create_time')
        read_only_fields = ('create_time',)

class TrainImageClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImageClass
        fields = ('pk', 'deviceid', 'traintype', 'source', 'upc', 'name', 'create_time')
        read_only_fields = ('create_time',)

class DatasetActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetAction
        fields = ('pk', 'action', 'traintype', 'desc', 'param', 'create_time', 'update_time')
        read_only_fields = ( 'filenum', 'param', 'create_time', 'update_time',)

class ExportActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExportAction
        fields = ('pk', 'train_action', 'model_name', 'checkpoint_prefix', 'param', 'create_time', 'update_time')
        read_only_fields = ( 'checkpoint_prefix', 'param', 'create_time', 'update_time')

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
        fields = ('pk', 'action', 'traintype', 'is_fineture', 'desc', 'param', 'export_actions', 'stop_train_actions', 'create_time', 'update_time')
        read_only_fields = ( 'param', 'create_time', 'update_time',)

class RfidImageCompareActionSerializer(serializers.ModelSerializer):
    # shopCode = serializers.CharField(default='ARBEEMkAABYQ')
    # startTime = serializers.DateField(default='2017-12-21')
    # endTime = serializers.DateField(default='2017-12-22')
    class Meta:
        model = RfidImageCompareAction
        fields = ('pk', 'deviceid', 'startTime', 'endTime')

class RfidTransactionSerializer(serializers.ModelSerializer):
    image = serializers.SlugRelatedField(
        many=False,
        read_only=True,
        slug_field='source'
     )
    class Meta:
        model = RfidTransaction
        fields = ('image', 'transaction_time', 'create_time')

class TransactionMetrixSerializer(serializers.ModelSerializer):
    rfid_transaction = RfidTransactionSerializer(many=False, read_only=True)
    class Meta:
        model = TransactionMetrix
        fields = ('pk', 'rfid_transaction', 'same_upc_num', 'only_rfid_upc_num', 'only_image_upc_num')
        read_only_fields = ( 'rfid_transaction', 'same_upc_num', 'only_rfid_upc_num', 'only_image_upc_num')
