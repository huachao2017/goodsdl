from rest_framework import serializers
from arm.models import ArmImage

class ArmImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = ArmImage
        fields = ('pk', 'rgb_source', 'depth_source', 'table_z')
        read_only_fields = ('result','create_time',)
