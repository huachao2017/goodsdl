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

    def add_baseline_image(self, image_path, upc, debug=False):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if debug:
            print('b_image kp:{}'.format(len(kp)))
        if len(kp) == 0:
            print('error: no key point to base image:{}'.format(image_path))
            return False

        if len(kp)< 30:
            print('error: too less keypoint count to base image:{}/{}'.format(len(kp),image_path))
            return False
        if upc in self.upc_to_cnt:
            self.upc_to_cnt[upc] += 1
        else:
            self.upc_to_cnt[upc] = 1
        key = upc + '_'+ str(self.upc_to_cnt[upc])
        self.path_to_baseline_info[key] = (kp, desc,image)
        return True

    def get_baseline_cnt(self):
        return len(self.path_to_baseline_info)

    def match_image_all_info(self, image_path, within_upcs=None, filter_upcs=None, min_match_points_cnt=4, min_score_thresh=0.5, max_score_thresh=0.8, debug=False, visual=True):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if debug:
            print('image kp:{}'.format(len(kp)))
        match_info = {}
        if len(kp) < 30:
            print('warn: too less keypoint count to match image:{}/{}'.format(len(kp),image_path))
            return match_info

        for key in self.path_to_baseline_info:
            if within_upcs is not None:
                upc = key.split('_')[0]
                if upc not in within_upcs:
                    continue
            if filter_upcs is not None:
                upc = key.split('_')[0]
                if upc in filter_upcs:
                    continue
            (b_kp,b_desc, b_image) = self.path_to_baseline_info[key]
            raw_matches = self.matcher.knnMatch(b_desc, trainDescriptors=desc, k=2)  # 2
            if debug:
                print('raw_matches:{}'.format(len(raw_matches)))
            kp_pairs = self.filter_matches(b_kp, kp, raw_matches,debug=debug)
            if debug:
                print('kp_pairs:{}'.format(len(kp_pairs)))
            if len(kp_pairs) >= min_match_points_cnt:
                mkp1, mkp2 = zip(*kp_pairs)
                p1 = numpy.float32([kp.pt for kp in mkp1])
                p2 = numpy.float32([kp.pt for kp in mkp2])
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
                if debug:
                    print('kp_cnt:{}'.format(numpy.sum(status)))
                if numpy.sum(status) >= min_match_points_cnt:
                    corners = numpy.float32([[0, 0], [b_image.shape[1], 0], [b_image.shape[1], b_image.shape[0]], [0, b_image.shape[0]]])
                    corners = numpy.int32(
                        cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

                    x = corners[:, 0]
                    y = corners[:, 1]
                    # 四个顶点远离边缘的距离
                    corner_distance = max(
                        abs(numpy.min(x))/image.shape[1],
                        abs(numpy.min(y))/image.shape[0],
                        abs(numpy.max(x)-image.shape[1])/image.shape[1],
                        abs(numpy.max(y)-image.shape[0])/image.shape[0]
                    )
                    # # corners平行四边形判断
                    # line1_delta = (corners[1][1]-corners[0][1])/max(1,(corners[1][0]-corners[0][0]))
                    # line3_delta = (corners[2][1]-corners[3][1])/max(1,(corners[2][0]-corners[3][0]))
                    # first_parallel_distance = abs(line1_delta/max(1,line3_delta) - 1)
                    # line2_delta = (corners[3][1]-corners[0][1])/max(1,(corners[3][0]-corners[0][0]))
                    # line4_delta = (corners[2][1]-corners[1][1])/max(1,(corners[2][0]-corners[1][0]))
                    # second_parallel_distance = abs(line2_delta/max(1,line4_delta) - 1)
                    # parallel_distance = max(first_parallel_distance, second_parallel_distance)
                    # if debug:
                    #     print('parallel_distance:{}'.format(parallel_distance))
                    #
                    b_area = b_image.shape[1]*b_image.shape[0]
                    transfer_area = cv2.contourArea(corners)
                    area_distance = abs(transfer_area-b_area)/max(1,min(b_area,transfer_area))
                    if debug:
                        print('area_distance:{}'.format(area_distance))
                    score = self.caculate_score(numpy.sum(status),
                                                corner_distance,
                                                # parallel_distance,
                                                area_distance,
                                                debug=debug)

                    if visual and (score > min_score_thresh or debug):
                        visual_path = os.path.join(os.path.dirname(image_path),
                                                   'visual_{}_{}_{}'.format(int(score*100), key, os.path.basename(image_path)))
                        self.match_visual(visual_path, b_image, image, kp_pairs, status, H)
                    if score > min_score_thresh:
                        match_info[key] = score
                        if score >= max_score_thresh:
                            break
        return match_info

    def filter_matches(self, kp1, kp2, matches, ratio=0.75, debug=False):
        mkp1, mkp2 = [], []
        trainIdxs = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                # if m.queryIdx in queryIdxs:
                #     continue
                if m.trainIdx in trainIdxs:
                    continue
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
                trainIdxs.append(m.trainIdx)
        kp_pairs = list(zip(mkp1, mkp2))
        return kp_pairs

    def caculate_score(self, cnt, corner_distance, area_distance, debug=False):
        if cnt <= 10:
            cnt_score = 0.05*cnt
        elif cnt <= 20:
            cnt_score = 0.03*(cnt-10) + 0.5
        else:
            cnt_score = 0.01*(cnt-20) + 0.8
        if cnt_score >= 1:
            cnt_score = 0.99

        corner_score = 1 - corner_distance # 差距大于1倍, 则惩罚为负值
        if corner_score < -1:
            corner_score = -1

        # parallel_score = 0.05 * (20 - parallel_distance)# 平行角度差距大于20, 则惩罚为负值
        #
        area_score = 1 - area_distance # 面积接近差1倍,则惩罚为负值
        if area_score < -1:
            area_score = 1

        score = cnt_score * 0.5 + corner_score * 0.25 + area_score * 0.25

        if debug:
            print('score: %.2f -- %.2f, %.2f, %.2f' % (score, cnt_score,corner_score,area_score))

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

    def match_image_best_one(self, image_path, within_upcs=None, filter_upcs=None, min_match_points_cnt=4, min_score_thresh=0.5, max_score_thresh=0.8, visual=True, debug=False):
        match_info = self.match_image_all_info(image_path,
                                               within_upcs=within_upcs,
                                               filter_upcs=filter_upcs,
                                               min_match_points_cnt=min_match_points_cnt,
                                               min_score_thresh=min_score_thresh,
                                               max_score_thresh=max_score_thresh,
                                               debug=debug, visual=visual)
        if len(match_info) == 0:
            return None,0
        if debug:
            print('match_info:{}'.format(len(match_info)))
        sorted_match_info = sorted(match_info.items(), key=lambda d: d[1], reverse=True)
        best_match = sorted_match_info[0]
        ret = (best_match[0].split('_')[0], best_match[1])
        return ret

    def is_find_match(self, image_path, within_upcs=None, filter_upcs=None, min_match_points_cnt=5, debug=False, visual=True):
        upc, score = self.match_image_best_one(image_path, within_upcs=within_upcs,filter_upcs=filter_upcs, min_match_points_cnt=min_match_points_cnt, debug=debug, visual=visual)
        return upc != None

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
        matcher.add_baseline_image('images/%d.jpg' % (i + 1), str(i), debug=True)
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one('images/9.jpg', debug=True)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_2(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1, 'tt', debug=True)
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one(image2, debug=True)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

def test_3(image1,image2):
    time0 = time.time()
    matcher = Matcher()
    time1 = time.time()
    matcher.add_baseline_image(image1, 'tt', debug=True)
    time2 = time.time()
    match_key, cnt = matcher.match_image_best_one(image2, debug=True)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, cnt)

if __name__ == '__main__':
    """Test code: Uses the two specified"""

    # test_1()
    # sys.exit(0)
    fn1 = 'images/1.jpg'
    fn2 = 'images/2.jpg'

    # fn1 = 'images/12.jpg'
    # fn2 = 'images/13.jpg'

    fn1 = 'images/test/old/16.jpg'
    fn2 = 'images/test/old/17.jpg'

    # fn1 = 'images/test/2.jpg'
    # fn2 = 'images/test/1.jpg'
    #
    # fn1 = 'images/error/1.jpg'
    # fn2 = 'images/error/2.jpg'
    test_3(fn1, fn2)