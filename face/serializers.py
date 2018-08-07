from rest_framework import serializers
from face.models import Image

class ImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = ('pk', 'picurl', 'index', 'create_time')
        read_only_fields = ('index', 'create_time',)
