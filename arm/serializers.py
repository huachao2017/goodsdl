from rest_framework import serializers
from arm.models import ArmImage, ArmTrainImage, ArmCylinderImage

class ArmImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = ArmImage
        fields = ('pk', 'rgb_source', 'depth_source', 'table_z')
        read_only_fields = ('result','create_time',)


class ArmCylinderImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArmCylinderImage
        fields = ('pk', 'rgb_source', 'depth_source', 'table_x', 'table_y', 'table_z')
        read_only_fields = ('result', 'create_time',)


class ArmTrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArmTrainImage
        fields = ('pk', 'deviceid', 'rgb_source', 'depth_source', 'table_z', 'upc')
        read_only_fields = ('create_time',)
