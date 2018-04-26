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

    def add_baseline_image(self, image_path, upc):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if len(kp) == 0:
            print('error: no key point to base image:{}'.format(image_path))
            return

        if upc in self.upc_to_cnt:
            self.upc_to_cnt[upc] += 1
        else:
            self.upc_to_cnt[upc] = 1
        key = upc + '_'+ str(self.upc_to_cnt[upc])
        self.path_to_baseline_info[key] = (kp, desc,image)

    def get_baseline_cnt(self):
        return len(self.path_to_baseline_info)

    def match_image_all_info(self, image_path, min_match_points_cnt=4, max_match_points=20):
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
            if len(kp_pairs) >= min_match_points_cnt:
                mkp1, mkp2 = zip(*kp_pairs)
                p1 = numpy.float32([kp.pt for kp in mkp1])
                p2 = numpy.float32([kp.pt for kp in mkp2])
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                if numpy.sum(status) >= min_match_points_cnt:
                    corners = numpy.float32([[0, 0], [b_image.shape[1], 0], [b_image.shape[1], b_image.shape[0]], [0, b_image.shape[0]]])
                    corners = numpy.int32(
                        cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

                    x = corners[:, 0]
                    y = corners[:, 1]
                    # print(numpy.min(x),numpy.min(y),numpy.max(x),numpy.max(y))
                    # print(image.shape[1],image.shape[0])
                    # print(area,image.shape[1]*image.shape[0])

                    # 四个顶点都靠近边缘或者超出边缘，则正确匹配
                    if numpy.min(x) < image.shape[1]/5 and numpy.min(y) < image.shape[0]/5 and image.shape[1]-numpy.max(x) < image.shape[1]/5 and image.shape[0]-numpy.max(y) < image.shape[0]/5:
                        # 转移必须是平行四边形
                        line1_delta = (corners[1][1]-corners[0][1])/(corners[1][0]-corners[0][0])
                        line3_delta = (corners[2][1]-corners[3][1])/(corners[2][0]-corners[3][0])
                        first_parallel_value = abs(line1_delta/line3_delta - 1)
                        line2_delta = (corners[3][1]-corners[0][1])/(corners[3][0]-corners[0][0])
                        line4_delta = (corners[2][1]-corners[1][1])/(corners[2][0]-corners[1][0])
                        second_parallel_value = abs(line2_delta/line4_delta - 1)
                        print(first_parallel_value, second_parallel_value)

                        if first_parallel_value < 0.5 and second_parallel_value < 0.5:
                            # 面积接近不能差20%
                            b_area = b_image.shape[1]*b_image.shape[0]
                            transfer_area = cv2.contourArea(corners)
                            area_distance = abs(transfer_area-b_area)/max(b_area,transfer_area)
                            print(area_distance)
                            if area_distance < 0.2:
                                match_info[key] = (b_image,image,kp_pairs,status,H)
                                if numpy.sum(status) >=max_match_points:
                                    break
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

    def caculate_score(self, cnt):
        if cnt < 10:
            score = 0.5
        else:
            score = (cnt-10)/20 + 0.5

        return score

    # def match_image_top_n(self, image_path, n=5, min_match_points_cnt=5, max_match_points=20, visual=True):
    #     match_info = self.match_image_all_info(image_path, min_match_points_cnt=min_match_points_cnt,max_match_points=max_match_points)
    #     if len(match_info) == 0:
    #         return None,0
    #     sorted_match_info = sorted(match_info.items(), key=lambda d: numpy.sum(d[1][3]), reverse=True)
    #     top_n = sorted_match_info[:n]
    #     ret = []
    #     for match in top_n:
    #         score = self.caculate_score(numpy.sum(match[1][3]))
    #         ret.append((match[0].split('_')[0],score))
    #     if visual:
    #         for i in range(len(top_n)):
    #             match = top_n[i]
    #             visual_path = os.path.join(os.path.dirname(image_path),'visual_{}_{}_{}'.format(match[0],i,os.path.basename(image_path)) )
    #             self.match_visual(visual_path, match[1][0],match[1][1],match[1][2],match[1][3],match[1][4])
    #     return ret

    def match_image_best_one(self, image_path, min_match_points_cnt=5, max_match_points=20, visual=True):
        match_info = self.match_image_all_info(image_path, min_match_points_cnt=min_match_points_cnt,max_match_points=max_match_points)
        if len(match_info) == 0:
            return None,0
        sorted_match_info = sorted(match_info.items(), key=lambda d: numpy.sum(d[1][3]), reverse=True)
        best_match = sorted_match_info[0]

        score = self.caculate_score(numpy.sum(best_match[1][3]))

        ret = (best_match[0].split('_')[0],score)
        if visual:
            visual_path = os.path.join(os.path.dirname(image_path),'visual_{}_{}'.format(best_match[0],os.path.basename(image_path)) )
            self.match_visual(visual_path, best_match[1][0],best_match[1][1],best_match[1][2],best_match[1][3],best_match[1][4])
        return ret

    def is_find_match(self, image_path, score_thresh = 0.5, min_match_points_cnt=5, visual=True):
        upc, score = self.match_image_best_one(image_path, min_match_points_cnt=min_match_points_cnt, visual=visual)
        return upc != None and score > score_thresh

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
        matcher.add_baseline_image('images/%d.jpg' % (i + 1), str(i))
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one('images/9.jpg')
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_2(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1, 'tt')
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one(image2)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_3(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1, 'tt')
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one(image2)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

if __name__ == '__main__':
    """Test code: Uses the two specified"""

    # test_1()
    # sys.exit(0)
    # fn1 = 'images/1.jpg'
    # fn2 = 'images/2.jpg'

    # fn1 = 'images/12.jpg'
    # fn2 = 'images/13.jpg'

    fn1 = 'images/error/3.jpg'
    fn2 = 'images/error/4.jpg'
    test_3(fn1, fn2)