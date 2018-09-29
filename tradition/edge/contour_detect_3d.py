import os
import cv2
import numpy as np
import os
import shutil
import time
import random
import math
import functools



class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '[{},{}]'.format(self.x,self.y)


def cmp(a, b, c):
    if a.x-c.x >= 0 and b.x-c.x < 0:
        return -1
    if a.x-c.x == 0 and b.x-c.x == 0:
        # return a.y > b.y
        if a.y > b.y:
            return -1
        elif a.y < b.y:
            return 1
        return 0
    det = (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
    if det < 0:
        return 1
    if det > 0:
        return -1
    d1 = (a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y)
    d2 = (b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y)
    # return d1 > d2
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0

def _rotated_rectangle_intersection_area(s_rect,m_rect,debug=False):
    r1 = cv2.rotatedRectangleIntersection(s_rect, m_rect)
    if r1[0] == 0:
        return 0, None
    elif r1[0] == 2:
        return s_rect[1][0]*s_rect[1][1], None

    x = 0
    y = 0
    p = []
    len_p = r1[1].shape[0]
    for i in range(len_p):
        p.append(Point(r1[1][i][0][0], r1[1][i][0][1]))
        x += r1[1][i][0][0]
        y += r1[1][i][0][1]

    c = Point(x / len_p, y / len_p)

    # if debug:
    #     print('source:{}'.format(''.join(map(str,p))))

    pp = sorted(p, key=functools.cmp_to_key(lambda x, y: cmp(x, y, c)))
    # if debug:
    #     print('sorted:{}'.format(''.join(map(str,pp))))
    r = np.full((len_p, 2), 0.0, dtype='float32')
    for i in range(len(pp)):
        r[i][0] = pp[i].x
        r[i][1] = pp[i].y
    r2 = cv2.contourArea(r)
    return (r2,r)

class Contour_3d:
    def __init__(self, rgb_path, depth_path, table_z, output_dir=None, debug_type=1, thresh_x = 120, overlapthresh=.3):
        image_dir, image_name = os.path.split(rgb_path)
        if output_dir is None:
            output_dir = image_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)


        self.image_name = image_name
        self.table_z = table_z
        self.rgb_img = cv2.imread(rgb_path)
        self.mask_rgb_img = self.rgb_img
        self.depth_img = cv2.imread(depth_path)
        self.depth_data = self.depth_img[:, :, 0] + self.depth_img[:, :, 1] * 256 + self.depth_img[:, :, 2] * 256 * 256

        self.output_dir = output_dir
        self.debug_type = debug_type
        self.thresh_x = thresh_x
        self.overlapthresh = overlapthresh

    def _solve_mask_image(self):

        depth_mask_code = np.full(self.depth_data.shape, self.table_z)
        mask_depth_img = np.where(self.depth_data > self.table_z, depth_mask_code, self.depth_data)
        mask_depth_img = np.where(mask_depth_img < 10, depth_mask_code, mask_depth_img)
        mask_depth_img = self.table_z - mask_depth_img
        mask_depth_img = np.where(mask_depth_img < 20, np.zeros(self.depth_data.shape), mask_depth_img)
        # print(mask_depth_img)
        mask_depth_img = np.expand_dims(mask_depth_img, 2)
        mask_depth_img = mask_depth_img.repeat(3, axis=2)
        rgb_mask_code = np.zeros(self.rgb_img.shape, np.uint8)
        self.mask_rgb_img = np.where(mask_depth_img > 0, self.rgb_img, rgb_mask_code)
        if self.debug_type > 0:
            output_path = os.path.join(self.output_dir, '_mask_' + self.image_name)
            cv2.imwrite(output_path, self.mask_rgb_img)

        return self.mask_rgb_img

    def _get_min_z(self,points):
        mask_points_img = np.zeros(self.depth_img.shape)
        cv2.drawContours(mask_points_img, [points], 0, (255,255,255), cv2.FILLED)
        if self.debug_type > 1:
            output_path = os.path.join(self.output_dir, 'depth_mask_' + self.image_name)
            cv2.imwrite(output_path, mask_points_img)

        mask_points_data = mask_points_img[:,:,0]
        mask_depth_data = np.where(mask_points_data > 0, self.depth_data, self.table_z)
        mask_depth_data = np.where(mask_depth_data < 10, self.table_z, mask_depth_data)
        min_z = np.min(mask_depth_data)
        print(min_z)

        return self.table_z-min_z

    def _non_max_suppression_minrect(self,min_rectes):
        # if there are no boxes, return an empty list
        if len(min_rectes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area_list = []
        for min_rect in min_rectes:
            area_list.append(min_rect[1][0] * min_rect[1][1])

        area = np.asarray(area_list)
        idxs = np.argsort(area)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # if self.debug_type>1:
            #     print('main:index:%d, center:%d,%d;w*h:%d,%d;theta:%d' % (i,
            #                                                               min_rectes[i][0][0], min_rectes[i][0][1],
            #                                                               min_rectes[i][1][0], min_rectes[i][1][1],
            #                                                               min_rectes[i][2]))

            need_del = []
            for index in range(last):
                j = idxs[index]
                # print(j, min_rectes[j])
                intersection_area, intersection = _rotated_rectangle_intersection_area(min_rectes[j], min_rectes[i])

                # compute the ratio of overlap
                overlap = intersection_area / area[j]
                # if self.debug_type>1:
                #     print('%d,%d,%.2f' % (j, i, overlap))
                if overlap > self.overlapthresh:
                    need_del.append(index)
                elif self.debug_type>1:
                    # print('comp:index:%d, center:%d,%d;w*h:%d,%d;theta:%d' % (j,
                    #                                                           min_rectes[j][0][0], min_rectes[j][0][1],
                    #                                                           min_rectes[j][1][0], min_rectes[j][1][1],
                    #                                                           min_rectes[j][2]))
                    drawing_contours = np.zeros(self.rgb_img.shape, np.uint8)
                    points = cv2.boxPoints(min_rectes[j])
                    points = np.int0(points)
                    color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
                    cv2.drawContours(drawing_contours, [points], 0, color, 1)
                    points = cv2.boxPoints(min_rectes[i])
                    points = np.int0(points)
                    color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
                    cv2.drawContours(drawing_contours, [points], 0, color, 1)
                    if intersection is not None:
                        cv2.drawContours(drawing_contours, [np.asarray(intersection, np.int0)], 0, color, 1)
                    output_path = os.path.join(self.output_dir, 'intersection_%d_%d.jpg' % (j, i))
                    cv2.imwrite(output_path, drawing_contours)

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.asarray(need_del))))

        # if self.debug_type>1:
        #     print(pick)
        # return only the bounding boxes that were picked using the
        # integer data type
        ret_min_rectes = []
        for index in pick:
            ret_min_rectes.append(min_rectes[index])
        return ret_min_rectes

    def _find_2d_minrect(self):
        # param@debug_type:0,not debug; 1,store bbox file; 2,store middle caculate file; 3,show window
        source = self.mask_rgb_img.copy()

        # step1: blur image
        max_area = source.shape[0] * source.shape[1]
        # Apply gaussian blur to the grayscale image
        # source = cv2.pyrMeanShiftFiltering(source, 31, 91)
        sharpen = source
        # blur = cv2.pyrMeanShiftFiltering(source, 21, 51)
        # kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
        #                            [-1,2,2,2,-1],
        #                            [-1,2,8,2,-1],
        #                            [-2,2,2,2,-1],
        #                            [-1,-1,-1,-1,-1]])/8.0
        # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharpen = cv2.filter2D(sharpen, -1, kernel_sharpen)
        sharpen = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)

        # 双向滤波比较不错
        # sharpen = cv2.bilateralFilter(sharpen, 3, 30, 30)
        # sharpen = cv2.split(sharpen)[0]
        # sharpen = cv2.equalizeHist(sharpen)
        # sharpen = cv2.GaussianBlur(sharpen, (5, 5), 0)
        if self.debug_type > 1:
            sharpen_path = os.path.join(self.output_dir, 'sharpen_' + self.image_name)
            cv2.imwrite(sharpen_path, sharpen)

        # step2: sobel caculate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        x = cv2.Sobel(sharpen, cv2.CV_64F, 1, 0, ksize=-1)
        y = cv2.Sobel(sharpen, cv2.CV_64F, 0, 1, ksize=-1)
        edges = cv2.subtract(x, y)
        edges = cv2.convertScaleAbs(edges)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        edges = cv2.bilateralFilter(edges, 5, 75, 75)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.erode(edges, kernel)
        edges = cv2.erode(edges, kernel)
        edges = cv2.erode(edges, kernel)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        if self.debug_type > 1:
            edges_path = os.path.join(self.output_dir, 'edges_' + self.image_name)
            cv2.imwrite(edges_path, edges)

        # step3: binary edges
        _, thresh1 = cv2.threshold(edges, self.thresh_x, 255, cv2.THRESH_BINARY)
        thresh2 = thresh1
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # thresh2 = cv2.erode(thresh2, kernel)
        # if morphology:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #     thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        # thresh2 = cv2.dilate(thresh2, kernel)
        # thresh2 = cv2.dilate(thresh2, kernel)
        # thresh2 = cv2.dilate(thresh2, kernel)
        # thresh2 = cv2.dilate(thresh2, kernel)
        # thresh2 = cv2.dilate(thresh2, kernel)
        # thresh2 = cv2.erode(thresh2, kernel)
        # thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
        # _, thresh = cv2.threshold(gray, x, 255, cv2.THRESH_BINARY_INV)
        # thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        if self.debug_type > 1:
            thresh1_path = os.path.join(self.output_dir, 'thresh1_' + self.image_name)
            cv2.imwrite(thresh1_path, thresh1)
            # if morphology:
            #     thresh2_path = os.path.join(output_dir, channel + '_' + 'thresh2_' + image_name)
            #     cv2.imwrite(thresh2_path, thresh2)

        # Find the edges
        # edges = cv2.Canny(gray,x1,x2)
        # edges = gray

        # step4: Detect contours
        _, contours, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print('find contours: {}'.format(len(contours)))
        # print('first contour: {}'.format(contours[0]))

        # step5: contour filter with area
        area_to_contour = {}
        for cnt in contours:
            cnt = cv2.convexHull(cnt, returnPoints=True)
            leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
            rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
            topmost = cnt[cnt[:, :, 1].argmin()][0][1]
            bottommost = cnt[cnt[:, :, 1].argmax()][0][1]

            # print('%d,%d,%d,%d' %(leftmost,rightmost,topmost,bottommost))
            # return
            area = (bottommost - topmost) * (rightmost - leftmost)
            if area < max_area / 200:  # 去除面积过小的物体
                continue
            # if area > max_area * .2:  # 去除面积过大的物体
            #     continue
            area_to_contour[area] = cnt
            # print(tuple(cnt[cnt[:, :, 0].argmin()][0]))
            # print(tuple(cnt[cnt[:, :, 0].argmax()][0]))

        # step6: caculate bounding box and draw contours
        drawing_contours = np.zeros(source.shape, np.uint8)
        areas = sorted(area_to_contour, reverse=True)
        index = 0
        min_rectes = []
        for area in areas:
            index += 1
            # if index > top_n:
            #     break
            cnt = area_to_contour[area]
            color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
            if self.debug_type > 1:
                cv2.drawContours(drawing_contours, [cnt], 0, color, 1)
            min_rect = cv2.minAreaRect(cnt)
            min_rectes.append(min_rect)
            # if debug_type > 1:
            #     drawing_contours = cv2.rectangle(drawing_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.debug_type > 1:
            contours_path = os.path.join(self.output_dir, 'contours_' + self.image_name)
            cv2.imwrite(contours_path, drawing_contours)

        # step7: nms min rect
        # min_rectes = _non_max_suppression_minrect(min_rectes, .3)
        if self.debug_type > 1 and len(min_rectes) > 0:
            minrect = np.copy(source)
            for min_rect in min_rectes:
                points = cv2.boxPoints(min_rect)
                points = np.int0(points)
                minrect = cv2.drawContours(minrect, [points], 0, (0, 0, 255), 1)
            minrect_path = os.path.join(self.output_dir, 'minrect_' + self.image_name)
            cv2.imwrite(minrect_path, minrect)

        if self.debug_type > 2:
            cv2.imshow('input', sharpen)
            cv2.imshow('edges', edges)
            cv2.imshow('thresh1', thresh1)
            # if morphology:
            #     cv2.imshow(channel + '_' + 'thresh2', thresh2)
            cv2.imshow('drawing_contours', drawing_contours)

        return min_rectes

    def find_contour(self,is_mask=True):

        # step0: read image
        if is_mask:
            mask_rgb_img = self._solve_mask_image()
        else:
            mask_rgb_img = self.rgb_img

        concate_minrectes = self._find_2d_minrect()
        # if self.debug_type > 1:
        #     print('{}'.format(len(concate_minrectes)))

        if self.debug_type > 1 and len(concate_minrectes)>0:
            drawing_contours = np.zeros(mask_rgb_img.shape, np.uint8)
            for minrect in concate_minrectes:
                points = cv2.boxPoints(minrect)
                points = np.int0(points)
                color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
                cv2.drawContours(drawing_contours, [points], 0, color, 1)
            output_path = os.path.join(self.output_dir, '_contour_' + self.image_name)
            cv2.imwrite(output_path, drawing_contours)

        suppression_minrectes = self._non_max_suppression_minrect(concate_minrectes)
        ret_minrectes = []
        ret_boxes = []
        ret_z = []
        for minrect in suppression_minrectes:
            # 做长宽转换并修订角度
            if minrect[1][0]>minrect[1][1]:
                ret_minrectes.append(((minrect[0][0],minrect[0][1]),(minrect[1][1],minrect[1][0]),90 + minrect[2]))
            else:
                ret_minrectes.append(minrect)
        for minrect in ret_minrectes:
            points = cv2.boxPoints(minrect)
            points = np.int0(points)
            x, y, w, h = cv2.boundingRect(points)
            ret_boxes.append([x,y,x+w,y+h])
            ret_z.append(self._get_min_z(points))

        if self.debug_type > 0 and len(ret_minrectes)>0:
            output = self.rgb_img
            index = 0
            for minrect in ret_minrectes:
                points = cv2.boxPoints(minrect)
                points = np.int0(points)
                cv2.drawContours(output, [points], 0, (255, 0, 0), 1)
                r = 3
                thickness = 2
                cv2.line(output, (int(minrect[0][0]) - r, int(minrect[0][1]) - r),
                         (int(minrect[0][0]) + r, int(minrect[0][1]) + r), (0, 0, 255), thickness)
                cv2.line(output, (int(minrect[0][0]) - r, int(minrect[0][1]) + r),
                         (int(minrect[0][0]) + r, int(minrect[0][1]) - r), (0, 0, 255), thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX

                index += 1
                cv2.putText(output, '%d,%d,%.2f,%d' % (minrect[0][0],minrect[0][1],minrect[2],ret_z[index-1]), (0, 12*index), font, 0.4, (255, 255, 255), 1)
            output_path = os.path.join(self.output_dir, '_output_'+self.image_name)
            cv2.imwrite(output_path, output)

        # scores = np.ones((len(concate_minrectes)))
        return ret_minrectes, ret_z, ret_boxes


