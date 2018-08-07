from rest_framework import serializers
from face.models import Image

class ImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = ('pk', 'picurl', 'image_path', 'index', 'create_time')
        read_only_fields = ('image_path', 'index', 'create_time',)
