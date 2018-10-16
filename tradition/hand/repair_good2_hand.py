import os
import time
from tradition.hand.hand_detect import HandDetect
import django
from django.conf import settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods2.models import Image

if __name__ == "__main__":

  path_prefix = '\\\\192.168.1.170/Image/'

  goods2_image_qs = Image.objects.order_by('-id')
  for image in goods2_image_qs:
    image_path = path_prefix + str(image.source)
    hand_detect = HandDetect(image_path, debug_type=1)
    ret = hand_detect.detect()
    if ret:
      image.is_hand = True
      image.save()
    print('%s: %s' % (image_path, str(ret)))
