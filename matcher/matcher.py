#!/usr/bin/env python

'''
Uses SURF to match two images.
Based on the sample code from opencv:
  samples/python2/find_obj.py

Example:
    matcher = Matcher()
    for i in range(8):
        matcher.add_baseline_image(%imagepath%)
    match_key, cnt = matcher.match_image_info(%imagepath%)
    is_match = matcher.match_image(%imagepath%)

'''

import numpy
import cv2
import os

import sys
import time


###############################################################################
# Image Matching For Servicing
###############################################################################

def _filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = list(zip(mkp1, mkp2))
    return kp_pairs

class Matcher:
    def __init__(self):
        self.path_to_baseline_keypoint = {}
        self.detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def add_baseline_image(self, image_path):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)

        upc = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        key = upc + '_' + file_name.split('_')[0]
        self.path_to_baseline_keypoint[key] = (kp, desc)

    def match_image_info(self, image_path, match_points_cnt=10, debug=False):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)

        match_info = {}
        for key in self.path_to_baseline_keypoint:
            (b_kp,b_desc) = self.path_to_baseline_keypoint[key]
            raw_matches = self.matcher.knnMatch(b_desc, trainDescriptors=desc, k=2)  # 2
            kp_pairs = _filter_matches(b_kp, kp, raw_matches)
            if len(kp_pairs) >= 4:
                mkp1, mkp2 = zip(*kp_pairs)
                p1 = numpy.float32([kp.pt for kp in mkp1])
                p2 = numpy.float32([kp.pt for kp in mkp2])
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                if numpy.sum(status) >= match_points_cnt:
                    match_info[key] = numpy.sum(status)
        if debug:
            print(match_info)
        if len(match_info) == 0:
            return None,0
        sorted_match_info = sorted(match_info.items(), key=lambda d: d[1], reverse=True)
        return sorted_match_info[0]

    def match_image(self, image_path, match_points_cnt=10, debug=False):
        key, cnt = self.match_image_info(image_path,match_points_cnt=match_points_cnt,debug=debug)
        return key != None


###############################################################################
# Image Matching For Diplaying
###############################################################################

def match_images(img1, img2, debug = False):
    time0 = time.time()
    """Given two images, returns the matches"""
    detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    if debug:
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
        print(kp1)
        print(desc1)

    time1 = time.time()
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    time2 = time.time()
    kp_pairs = _filter_matches(kp1, kp2, raw_matches)
    print(kp_pairs)
    time3 = time.time()
    if debug:
        print('MATCH: %.2f, %.2f, %.2f, %.2f' %(time3-time0,time1-time0,time2-time1,time3-time2))
    return kp_pairs

def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1 + w2, 3), numpy.uint8)
    vis[:h1, :w1, :] = img1
    vis[:h2, w1:w1 + w2, :] = img2
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)

    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])

    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (numpy.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    if len(p1):
        explore_match(window_name, img1, img2, kp_pairs, status, H)


###############################################################################
# Test Main
###############################################################################

def test_1():
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    for i in range(8):
        matcher.add_baseline_image('images/%d.jpg' % (i+1))
    time2 = time.time()
    match_key, cnt = matcher.match_image_info('images/9.jpg', debug=True)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_2():
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image('images/t_1.jpg')
    time2 = time.time()
    match_key, cnt = matcher.match_image_info('images/t_3.jpg', debug=True)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

if __name__ == '__main__':
    """Test code: Uses the two specified"""

    # test_1()
    # sys.exit(0)

    # test_2()
    # sys.exit(0)

    fn1 = 'images/t_1.jpg'
    fn2 = 'images/t_2.jpg'

    img1 = cv2.imread(fn1)
    img2 = cv2.imread(fn2)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    kp_pairs = match_images(img1, img2, debug=True)

    if kp_pairs:
        draw_matches('find_obj', kp_pairs, img1, img2)
        cv2.waitKey()
        cv2.destroyAllWindows()
else:
    print("No matches found")