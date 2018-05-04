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
import math
from tradition.matcher.thread_pool import ThreadPool


def _one_match(thread_name, matcher, task_cnt, key, image_path, image, kp, desc):
    if matcher.debug:
        print("begin match thread %s" % (thread_name))
    (b_kp, b_desc, b_image) = matcher.path_to_baseline_info[key]
    raw_matches = matcher.matcher.knnMatch(desc, trainDescriptors=b_desc, k=2)  # 2
    if matcher.debug:
        print('raw_matches:{}'.format(len(raw_matches)))
    kp_pairs = matcher.filter_matches(kp, b_kp, raw_matches)
    if matcher.debug:
        print('kp_pairs:{}'.format(len(kp_pairs)))
    if len(kp_pairs) >= matcher.min_match_points_cnt:
        mkp1, mkp2 = zip(*kp_pairs)
        p1 = numpy.float32([kp.pt for kp in mkp1])
        p2 = numpy.float32([kp.pt for kp in mkp2])
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        if matcher.debug:
            print('kp_cnt:{}'.format(numpy.sum(status)))
        if numpy.sum(status) >= matcher.min_match_points_cnt:
            corners = numpy.float32(
                [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
            corners = numpy.int32(
                cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

            x = corners[:, 0]
            y = corners[:, 1]
            corner_distance = max(
                abs(numpy.min(x)) / b_image.shape[1],
                abs(numpy.min(y)) / b_image.shape[0],
                abs(numpy.max(x) - b_image.shape[1]) / b_image.shape[1],
                abs(numpy.max(y) - b_image.shape[0]) / b_image.shape[0]
            )
            if matcher.debug:
                print('corner_distance:{}'.format(corner_distance))
            if corner_distance <= 1:  # 四个顶点远离边缘的距离过大，则不匹配 TODO maybe some problem
                # corners平行四边形判断
                line1_delta = math.atan(
                    (corners[1][1] - corners[0][1]) / (corners[1][0] - corners[0][0]) if corners[1][0] - corners[0][
                        0] != 0 else 10000) * 180 / math.pi
                line3_delta = math.atan(
                    (corners[2][1] - corners[3][1]) / (corners[2][0] - corners[3][0]) if corners[2][0] - corners[3][
                        0] != 0 else 10000) * 180 / math.pi
                first_parallel_distance = abs(line1_delta - line3_delta)
                if matcher.debug:
                    print(line1_delta, line3_delta, first_parallel_distance)
                line2_delta = math.atan(
                    (corners[3][1] - corners[0][1]) / (corners[3][0] - corners[0][0]) if corners[3][0] - corners[0][
                        0] != 0 else 10000) * 180 / math.pi
                line4_delta = math.atan(
                    (corners[2][1] - corners[1][1]) / (corners[2][0] - corners[1][0]) if corners[2][0] - corners[1][
                        0] != 0 else 10000) * 180 / math.pi
                second_parallel_distance = abs(line2_delta - line4_delta)
                if matcher.debug:
                    print(line2_delta, line4_delta, second_parallel_distance)
                parallel_distance = max(first_parallel_distance, second_parallel_distance)
                if matcher.debug:
                    print('parallel_distance:{},{},{}'.format(parallel_distance, first_parallel_distance,
                                                              second_parallel_distance))

                area = image.shape[1] * image.shape[0]
                transfer_area = cv2.contourArea(corners)
                area_distance = abs(transfer_area - area) / max(1, min(area, transfer_area))
                if matcher.debug:
                    print('area_distance:{}'.format(area_distance))
                score = matcher.caculate_score(numpy.sum(status),
                                               # corner_distance,
                                            parallel_distance,
                                               area_distance)

                if matcher.visual and (score > matcher.min_score_thresh or matcher.debug):
                    visual_path = os.path.join(os.path.dirname(image_path),
                                               'visual_{}_{}_{}'.format(int(score * 100), key,
                                                                        os.path.basename(image_path)))
                    matcher.match_visual(visual_path, image, b_image, kp_pairs, status, H)
                if score > matcher.min_score_thresh:
                    matcher.match_info[key] = score
                    # if score >= self.max_score_thresh:
                    #     break
    matcher.task_info[task_cnt] += 1

###############################################################################
# Image Matching For Servicing
###############################################################################
class Matcher:
    def __init__(self, min_match_points_cnt=4, min_score_thresh=0.5, max_score_thresh=0.8, debug=False, visual=False, max_thread=20):
        self.path_to_baseline_info = {}
        self.upc_to_cnt = {}
        self.detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_match_points_cnt = min_match_points_cnt
        self.min_score_thresh = min_score_thresh
        self.max_score_thresh = max_score_thresh
        self.debug = debug
        self.visual = visual
        self.task_cnt = 0
        self.task_info = {}
        self.match_info = None
        self.max_thread = max_thread
        self.thread_pool = ThreadPool(max_thread)


    def add_baseline_image(self, image_path, upc):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if self.debug:
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

    def removeall_baseline_image(self):
        self.path_to_baseline_info = {}

    def get_baseline_cnt(self):
        return len(self.path_to_baseline_info)

    # def get_thread_size(self):
    #     thread_size = int(len(self.path_to_baseline_info)/100)
    #     if thread_size > self.max_thread:
    #         thread_size = self.max_thread
    #
    #     return thread_size
    #
    def _all_match(self, image_path, within_upcs=None, filter_upcs=None):
        image = cv2.imread(image_path)
        kp, desc = self.detector.detectAndCompute(image, None)
        if self.debug:
            print('image kp:{}'.format(len(kp)))
        self.match_info = {}
        if len(kp) < 30:
            print('warn: too less keypoint count to match image:{}/{}'.format(len(kp),image_path))

        if self.debug:
            print('baseline image:{}'.format(len(self.path_to_baseline_info)))

        task_cnt = self.task_cnt + 1
        self.task_cnt += 1
        self.task_info[task_cnt] = 0
        need_task_cnt = 0
        for key in self.path_to_baseline_info:
            if within_upcs is not None:
                upc = key.split('_')[0]
                if upc not in within_upcs:
                    continue
            if filter_upcs is not None:
                upc = key.split('_')[0]
                if upc in filter_upcs:
                    continue

            if self.thread_pool is not None:
                need_task_cnt += 1
                self.thread_pool.put(_one_match, (self, task_cnt, key, image_path, image, kp, desc), None)
            else:
                _one_match('main_thread',self, task_cnt, key, image_path, image, kp, desc)

        if self.thread_pool is not None:
            time0 = time.time()
            i = 0
            while i < 30:
                i += 1
                if self.task_info[task_cnt] == need_task_cnt:
                    time1 = time.time()
                    print("\033[32;0m任务正常完成%s(%.2f秒)：目前线程池中有%s个线程，空闲的线程有%s个！\033[0m"
                          % (self.task_info[task_cnt], time1-time0, len(self.thread_pool.generate_list), len(self.thread_pool.free_list)))
                    break
                time.sleep(0.1)
            else:
                time1 = time.time()
                print("\033[31;0m任务没有完成%s(共%s,%.2f秒)：目前线程池中有%s个线程，空闲的线程有%s个！\033[0m"
                      % (self.task_info[task_cnt], need_task_cnt, time1-time0, len(self.thread_pool.generate_list), len(self.thread_pool.free_list)))

    def filter_matches(self, kp1, kp2, matches, ratio=0.75):
        mkp1, mkp2 = [], []
        trainIdxs = {}
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                # if m.queryIdx in queryIdxs:
                #     continue
                if m.trainIdx in trainIdxs:
                    if trainIdxs[m.trainIdx] > 2: # FIXME 匹配两个以上尚未支持
                        continue
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
                if m.trainIdx in trainIdxs:
                    trainIdxs[m.trainIdx] += 1
                else:
                    trainIdxs[m.trainIdx] = 1
        kp_pairs = list(zip(mkp1, mkp2))
        return kp_pairs

    def caculate_score(self, cnt, parallel_distance,area_distance):
        if cnt <= 10:
            cnt_score = 0.1*(cnt-5)
        elif cnt <= 20:
            cnt_score = 0.03*(cnt-10) + 0.5
        else:
            cnt_score = 0.01*(cnt-20) + 0.8
        if cnt_score >= 1:
            cnt_score = 0.99

        parallel_score = 0.02 * (50 - parallel_distance)# 平行角度差距大于20, 则惩罚为负值

        area_score = 1 - area_distance # 面积接近差1倍,则惩罚为负值
        if area_score < -1:
            area_score = -1

        score = cnt_score * 0.5 + min(parallel_score,area_score) * 0.5

        if self.debug:
            print('score: %.2f = %.2f*0.5+min(%.2f, %.2f)*0.5' % (score, cnt_score,parallel_score,area_score))

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

    def match_image_best_one(self, image_path, within_upcs=None, filter_upcs=None):
        self._all_match(image_path,
                        within_upcs=within_upcs,
                        filter_upcs=filter_upcs)
        if self.match_info is None or len(self.match_info) == 0:
            return None,0
        if self.debug:
            print('match_info:{}'.format(len(self.match_info)))
        sorted_match_info = sorted(self.match_info.items(), key=lambda d: d[1], reverse=True)
        best_match = sorted_match_info[0]
        ret = (best_match[0].split('_')[0], best_match[1])
        return ret

    def is_find_match(self, image_path, within_upcs=None, filter_upcs=None):
        upc, score = self.match_image_best_one(image_path, within_upcs=within_upcs,filter_upcs=filter_upcs)
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
    matcher = Matcher(debug=True, visual=True)
    time1 = time.time()
    for i in range(8):
        matcher.add_baseline_image('images/%d.jpg' % (i + 1), str(i))
    time2 = time.time()
    match_key, score = matcher.match_image_best_one('images/9.jpg')
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, score)

def test_2(image1,image2):
    time0 = time.time()
    matcher = Matcher(debug=False, visual=False)
    time1 = time.time()
    matcher.add_baseline_image(image1, 'tt')
    time2 = time.time()
    match_key, score = matcher.match_image_best_one(image2)
    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
    print(match_key, score)

def test_match_all():
    time0 = time.time()
    matcher = Matcher(debug=False, visual=False)
    time1 = time.time()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
    import django
    django.setup()
    from goods.models import SampleImageClass
    from django.conf import settings
    samples = SampleImageClass.objects.filter(deviceid='')
    upc_to_image_path = {}
    for sample in samples:
        image_path = sample.source.path
        image_path = image_path.replace(settings.MEDIA_ROOT, '\\\\192.168.1.170\Image')
        # image_path = image_path.replace('\\','/')
        # image_path = '\\' + image_path
        if os.path.isfile(image_path):
            matcher.add_baseline_image(image_path, sample.upc)
            upc_to_image_path[sample.upc] = image_path
    time2 = time.time()
    for upc in upc_to_image_path:
        image_path = upc_to_image_path[upc]
        # print(image_path)
        match_key, score = matcher.match_image_best_one(image_path, within_upcs=[upc])
        if score < 0.8:
            print(match_key, score)

    time3 = time.time()
    print('MATCH: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))

if __name__ == '__main__':
    """Test code: Uses the two specified"""

    # test_1()
    # sys.exit(0)
    fn1 = 'images/1.jpg'
    fn2 = 'images/2.jpg'

    # fn1 = 'images/12.jpg'
    # fn2 = 'images/13.jpg'

    # fn1 = 'images/test/old/15.jpg'
    # fn2 = 'images/test/old/14.jpg'
    # #
    # fn1 = 'images/test/1.jpg'
    # fn2 = 'images/test/2.jpg'
    #
    # fn1 = 'images/error/1.jpg'
    # fn2 = 'images/error/2.jpg'
    # test_2(fn1, fn2)
    test_match_all()