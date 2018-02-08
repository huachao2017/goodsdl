# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
import functools
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_info_on_image_array(image,
                             bg_color='red',
                             font_color='black',
                             display_str_list=(),
                             groundtruth_image_path=None):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  groundtruth_image = None
  if groundtruth_image_path:
      groundtruth_image = Image.open(groundtruth_image_path)
  draw_info_on_image(image_pil, bg_color, font_color, display_str_list,groundtruth_image=groundtruth_image)
  np.copyto(image, np.array(image_pil))



def draw_info_on_image(image,
                       bg_color='red',
                       font_color='black',
                       display_str_list=(),
                       groundtruth_image=None):
  """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, top) = (0, 0)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  text_bottom = top
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom), (left + text_width,text_bottom + text_height + 2 * margin)],
        fill=bg_color)
    draw.text(
        (left + margin, text_bottom + margin),
        display_str,
        fill=font_color,
        font=font)
    text_bottom += text_height + 2 * margin
  if groundtruth_image is not None:
      width, height = image.size
      crop_width = int(width / 3)
      crop_height = int(height / 3)
      im = groundtruth_image.resize((crop_width, crop_height))

      box = (width - crop_width, height - crop_height, width, height)
      image.paste(im, box)

def visualize_truth_on_image_array(image,
                                   detection_class_label,
                                   detection_score,
                                   labels_to_names):
  """
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.

  display_str_list = []
  display_str_list.append('{}'.format(labels_to_names[detection_class_label]))
  display_str_list.append('{}%'.format(int(100 * detection_score)))
  bg_color = 'White'
  font_color = 'Black'

  # Draw all boxes onto image.
  draw_info_on_image_array(
    image,
    bg_color,
    font_color,
    display_str_list=display_str_list)

  return image

def visualize_false_on_image_array(image,
                                   detection_class_label,
                                   detection_score,
                                   groundtruth_class_label,
                                   labels_to_names,
                                   groundtruth_image_path=None):
  """
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.

  display_str_list = []
  display_str_list.append('{}'.format(labels_to_names[detection_class_label]))
  display_str_list.append('{}%'.format(int(100 * detection_score)))
  display_str_list.append('{}'.format(labels_to_names[groundtruth_class_label]))
  bg_color = 'Red'
  font_color = 'Black'

  # Draw all boxes onto image.
  draw_info_on_image_array(
    image,
    bg_color,
    font_color,
    display_str_list=display_str_list,
    groundtruth_image_path=groundtruth_image_path
  )

  return image
