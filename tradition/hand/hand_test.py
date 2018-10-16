import os
import time
from tradition.hand.hand_detect import HandDetect

if __name__ == "__main__":

  base_dir, _ = os.path.split(os.path.realpath(__file__))
  image_dir = os.path.join(base_dir, 'images')
  for image in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image)
    split_image_path = os.path.splitext(image_path)
    hand_split_image_path = split_image_path[0].split('_')
    if split_image_path[-1] == '.jpg' and hand_split_image_path[-1] != 'hand':
      start_time = time.time()
      hand_detect = HandDetect(image_path,debug_type=1)
      ret = hand_detect.detect()
      end_time = time.time()
      print('%s: %s, %.2f' % (image, str(ret), end_time-start_time) )