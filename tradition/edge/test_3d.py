import os
import cv2
import time
import django
from django.conf import settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from arm.models import ArmImage
from tradition.edge.contour_detect_3d import Contour_3d


def _inner_find_one(rgb_path, depth_path, table_z, output_dir, is_mask=True, debug_type=2):
  time0 = time.time()
  one = Contour_3d(rgb_path, depth_path, table_z, output_dir, debug_type=debug_type, overlapthresh = .7)
  min_rectes, z, boxes = one.find_contour(is_mask)
  # _,boxes,_ = find_contour(image_path, output_dir=output_dir,debug_type=1)
  time1 = time.time()
  print('%s:%.2f, %d' % (rgb_path, time1 - time0, len(min_rectes)))
  index = 0
  for min_rect in min_rectes:
    print('center: %d,%d; w*h:%d,%d; theta:%d; z:%d, boxes: x1:%d, y1:%d, x2:%d, y2:%d' % (
    min_rect[0][0], min_rect[0][1], min_rect[1][0], min_rect[1][1], min_rect[2], z[index], boxes[index][0],
    boxes[index][1], boxes[index][2], boxes[index][3]))
    index += 1


if __name__ == "__main__":
    # Enter the input image file
    base_dir, _ = os.path.split(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir,'images')
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(image_dir, 'contour')
    if os.path.isdir(output_dir):
        for image in os.listdir(output_dir):
            tmp_path = os.path.join(output_dir, image)
            if os.path.splitext(tmp_path)[-1] == '.png' or os.path.splitext(tmp_path)[-1] == '.jpg':
                os.remove(tmp_path)

    # for test
    # rgb_path = os.path.join(image_dir, "03.jpg")
    # depth_path = os.path.join(image_dir, "03_d.png")
    # _inner_find_one(rgb_path, depth_path, 1230,output_dir,  is_mask=False, debug_type=1)

    test_dir = os.path.join(image_dir,'test')
    path_prefix = '\\\\192.168.1.60/Image/'

    if os.path.isdir(test_dir):
      arm_image_qs = ArmImage.objects.order_by('-id')[:10]
      for arm_image in arm_image_qs:
        image_path = path_prefix+str(arm_image.rgb_source)
        depth_path = path_prefix+str(arm_image.depth_source)
        # print(image_path)
        # image_path = os.path.join(test_dir,image)
        if os.path.isfile(image_path) and os.path.isfile(depth_path):
          _inner_find_one(image_path, depth_path, 1230, output_dir, is_mask=False, debug_type=2) # 不用掩码检测
        else:
          print('image file not exist: %s:%s' % (image_path, depth_path))

          # _inner_find_one(image_path, depth_path, 1230, output_dir, is_mask=True, debug_type=2)   # 用掩码后检测

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()