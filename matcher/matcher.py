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

class Matcher:
    def __init__(self):
        self.path_to_baseline_info = {}
        self.upc_to_cnt = {}
        self.detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def add_baseline_image(self, image_path):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if len(kp) == 0:
            print('error: no key point to base image:{}'.format(image_path))
            return

        upc = os.path.basename(os.path.dirname(image_path))
        if upc in self.upc_to_cnt:
            self.upc_to_cnt[upc] += 1
        else:
            self.upc_to_cnt[upc] = 1
        key = upc + '_'+ str(self.upc_to_cnt[upc])
        self.path_to_baseline_info[key] = (kp, desc,image)

    def match_image_all_info(self, image_path, solve_size=True, match_points_cnt=5):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        match_info = {}
        if len(kp) == 0:
            print('warn: no key point to match image:{}'.format(image_path))
            return match_info

        for key in self.path_to_baseline_info:
            (b_kp,b_desc, b_image) = self.path_to_baseline_info[key]
            raw_matches = self.matcher.knnMatch(b_desc, trainDescriptors=desc, k=2)  # 2
            kp_pairs = self.filter_matches(b_kp, kp, raw_matches)
            if len(kp_pairs) >= 4:
                mkp1, mkp2 = zip(*kp_pairs)
                p1 = numpy.float32([kp.pt for kp in mkp1])
                p2 = numpy.float32([kp.pt for kp in mkp2])
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                if numpy.sum(status) >= match_points_cnt:
                    if solve_size:
                        corners = numpy.float32([[0, 0], [b_image.shape[1], 0], [b_image.shape[1], b_image.shape[0]], [0, b_image.shape[0]]])
                        corners = numpy.int32(
                            cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
                        # area = cv2.contourArea(corners)
                        x = corners[:, 0]
                        y = corners[:, 1]
                        # print(numpy.min(x),numpy.min(y),numpy.max(x),numpy.max(y))
                        # print(image.shape[1],image.shape[0])
                        # print(area,image.shape[1]*image.shape[0])

                        # 面积接近或者（不再需要）
                        # 四个顶点都靠近边缘或者超出边缘，则正确匹配
                        if numpy.min(x) < image.shape[1]/5 and numpy.min(y) < image.shape[0]/5 and image.shape[1]-numpy.max(x) < image.shape[1]/5 and image.shape[0]-numpy.max(y) < image.shape[0]/5:
                            match_info[key] = (b_image,image,kp_pairs,status,H)
                    else:
                        match_info[key] = (b_image,image,kp_pairs,status,H)
        return match_info

    def filter_matches(self, kp1, kp2, matches, ratio=0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
        kp_pairs = list(zip(mkp1, mkp2))
        return kp_pairs

    def match_image_best_one_info(self, image_path, solve_size=True, match_points_cnt=5, visual=True):
        match_info = self.match_image_all_info(image_path, solve_size=solve_size, match_points_cnt=match_points_cnt)
        if len(match_info) == 0:
            return None,0
        sorted_match_info = sorted(match_info.items(), key=lambda d: numpy.sum(d[1][3]), reverse=True)
        if visual:
            visual_path = os.path.join(os.path.dirname(image_path),'{}_{}'.format(sorted_match_info[0][0],os.path.basename(image_path)) )
            self.match_visual(visual_path, sorted_match_info[0][1][0],sorted_match_info[0][1][1],sorted_match_info[0][1][2],sorted_match_info[0][1][3],sorted_match_info[0][1][4])
        return sorted_match_info[0][0],numpy.sum(sorted_match_info[0][1][3])

    def is_find_match(self, image_path, solve_size=True, match_points_cnt=5):
        key, cnt = self.match_image_best_one_info(image_path, solve_size=solve_size, match_points_cnt=match_points_cnt)
        return key != None

    def match_visual(self, visual_path, img1, img2, kp_pairs, status=None, H=None):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = numpy.zeros((max(h1, h2), w1 + w2, 3), numpy.uint8)
        vis[:h1, :w1, :] = img1
        vis[:h2, w1:w1 + w2, :] = img2
        # vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))
            # center = numpy.int32(numpy.sum(corners, 0) / len(corners))
            # print(center)
            # col = (255, 0, 0)
            # r = 2
            # thickness = 3
            # cv2.line(vis, (center[0] - r, center[1] - r), (center[0] + r, center[1] + r), col, thickness)
            # cv2.line(vis, (center[0] - r, center[1] + r), (center[0] + r, center[1] - r), col, thickness)

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

        cv2.imwrite(visual_path, vis)


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
    match_key, cnt = matcher.match_image_best_one_info('images/9.jpg')
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_2(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1)
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one_info(image2)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_3(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1)
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one_info(image2,solve_size=False)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

if __name__ == '__main__':
    """Test code: Uses the two specified"""

    test_1()
    sys.exit(0)
    fn1 = 'images/1.jpg'
    fn2 = 'images/2.jpg'
    test_3(fn1, fn2)