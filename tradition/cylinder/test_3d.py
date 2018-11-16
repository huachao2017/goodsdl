import os
import cv2
import time
import django
from django.conf import settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from arm.models import ArmImage
from tradition.cylinder.cylinder_detect_3d import Cylinder_3d


def _inner_find_one(rgb_path, depth_path, table_x, table_y, table_z, output_dir, debug_type=2):
  time0 = time.time()
  one = Cylinder_3d(rgb_path, depth_path, table_x, table_y, table_z, output_dir, debug_type=debug_type)
  location = one.find_cylinder()
  # _,boxes,_ = find_contour(image_path, output_dir=output_dir,debug_type=1)
  time1 = time.time()
  print('%s:%.2f' % (rgb_path, time1 - time0))
  print(location)


if __name__ == "__main__":
    # Enter the input image file
    base_dir, _ = os.path.split(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir,'images')
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(image_dir, 'cylinder')
    if os.path.isdir(output_dir):
        for image in os.listdir(output_dir):
            tmp_path = os.path.join(output_dir, image)
            if os.path.splitext(tmp_path)[-1] == '.png' or os.path.splitext(tmp_path)[-1] == '.jpg':
                os.remove(tmp_path)

    # for test
    rgb_path = os.path.join(image_dir, "2.jpg")
    depth_path = os.path.join(image_dir, "2.png")
    _inner_find_one(rgb_path, depth_path, 100, 100, 1230, output_dir, debug_type=2)

