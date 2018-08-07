"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from face.dl import facenet
import time


def main(args):
    time0 = time.time()
    image_files, images = load_and_preprocess_data(args.image_dir_path, args.image_size)
    time1 = time.time()
    with tf.Graph().as_default():

        with tf.Session() as sess:

            nrof_images = len(image_files)
            # Load the model
            facenet.load_model(args.model)
            time2 = time.time()

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            # feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            # emb = sess.run(embeddings, feed_dict=feed_dict)
            list_emb = []
            for image in images:
                np_image = np.expand_dims(image, 0)
                feed_dict = {images_placeholder: np.expand_dims(image, 0), phase_train_placeholder: False}
                one_emb = sess.run(embeddings, feed_dict=feed_dict)
                list_emb.append(one_emb[0])

            print(one_emb)

            emb = np.stack(list_emb)

            time3 = time.time()

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            time4 = time.time()

    print('%d张图片用时(总时间，图片压缩，模型加载，计算，比对): %.2f,%.2f,%.2f,%.2f,%.2f' % (
    nrof_images, time4 - time0, time1 - time0, time2 - time1, time3 - time2, time4 - time3))


def load_and_preprocess_data(image_dir_path, image_size, limit=-1):
    img_list = []
    image_names = []
    cnt = 0
    for image_name in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, image_name)
        example, ext = os.path.splitext(image_path)
        if ext != ".jpg":
            continue
        if limit > 0 and cnt >= limit:
            break
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        image_names.append(image_name)
        cnt += 1
    images = np.stack(img_list)
    return image_names, images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='../model/20180402-114759.pb')
                        # default='./model/20170512-110547.pb')
    parser.add_argument('--image_dir_path', type=str, help='Image dir path to compare', default='./data')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
