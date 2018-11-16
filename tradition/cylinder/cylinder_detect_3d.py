import os
import cv2
import numpy as np
import os
import shutil
import time
import random
import math
import functools


class Cylinder_3d:
    def __init__(self, rgb_path, depth_path, table_z, output_dir=None, debug_type=1, thresh_white = 100):
        image_dir, image_name = os.path.split(rgb_path)
        if output_dir is None:
            output_dir = image_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)


        self.image_name = image_name
        self.table_z = table_z
        self.rgb_img = cv2.imread(rgb_path)
        self.gray_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY)
        self.mask_rgb_img = self.gray_img
        self.depth_img = cv2.imread(depth_path)
        self.depth_data = self.depth_img[:, :, 0] + self.depth_img[:, :, 1] * 256 + self.depth_img[:, :, 2] * 256 * 256

        self.output_dir = output_dir
        self.debug_type = debug_type
        self.thresh_white = thresh_white

    def _find_lines(self):
        # 转成g灰度
        if self.debug_type > 1:
            gray_path = os.path.join(self.output_dir, 'gray_' + self.image_name)
            cv2.imwrite(gray_path, self.gray_img)

        # 白色阈值
        # white_thresh = self.gray_img.copy()
        height, width = self.gray_img.shape
        white = 255
        black = 0
        # 将非白区域变白
        white_thresh = np.where(self.gray_img > white-self.thresh_white, white, black)
        white_thresh = white_thresh.astype(np.uint8)
        if self.debug_type > 1:
            white_path = os.path.join(self.output_dir, 'white1_' + self.image_name)
            cv2.imwrite(white_path, white_thresh)

        _, cnts, _ = cv2.findContours(white_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(cnts))
        # cnts = cnts[0]
        # 将轮廓按面积大小降序排序
        c = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 将小的区域填充成黑色
        for i in range(len(c) - 1):
            x, y, w, h = cv2.boundingRect(c[i + 1])
            white_thresh[y:(y + h), x:(x + w)] = black
        if self.debug_type > 1:
            white_path = os.path.join(self.output_dir, 'white2_' + self.image_name)
            cv2.imwrite(white_path, white_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_thresh = cv2.erode(white_thresh,kernel=kernel)
        white_thresh = cv2.erode(white_thresh,kernel=kernel)
        white_thresh = cv2.erode(white_thresh,kernel=kernel)
        white_thresh = cv2.erode(white_thresh,kernel=kernel)
        if self.debug_type > 1:
            erode_path = os.path.join(self.output_dir, 'erode_' + self.image_name)
            cv2.imwrite(erode_path, white_thresh)

        # 掩码
        mask_img = np.where(self.gray_img > white - 30, white, self.gray_img)
        mask_img = np.where(white_thresh == black, white, mask_img)
        mask_img.astype(np.uint8)
        # mask_img = self.gray_img.copy()
        # for i in range(height):
        #     for j in range(width):
        #         if white_thresh[i,j] == black:
        #             mask_img[i,j] = white
        #         elif white - self.gray_img[i, j] < 30:
        #             mask_img[i, j] = white
        #         else:
        #             pass
        if self.debug_type > 1:
            mask_path = os.path.join(self.output_dir, 'mask_' + self.image_name)
            cv2.imwrite(mask_path, mask_img)

        _, thresh = cv2.threshold(mask_img, 225, 255, cv2.THRESH_BINARY)
        if self.debug_type > 1:
            thresh_path = os.path.join(self.output_dir, 'thresh1_' + self.image_name)
            cv2.imwrite(thresh_path, thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        drawing_contours = np.zeros(self.gray_img.shape, np.uint8)
        points = None
        for cnt in contours:
            leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
            rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
            topmost = cnt[cnt[:, :, 1].argmin()][0][1]
            bottommost = cnt[cnt[:, :, 1].argmax()][0][1]
            area = (bottommost - topmost) * (rightmost - leftmost)
            if area < 20:  # 去除面积过小的物体
                continue
            if area > 0.8*self.gray_img.shape[0]*self.gray_img.shape[1]:  # 去除面积过大的物体
                continue
            # color = np.random.randint(0, 255, (3)).tolist()
            cv2.drawContours(drawing_contours, [cnt], 0, 128, 1)

            min_rect = cv2.minAreaRect(cnt)
            points = cv2.boxPoints(min_rect)
            points = np.int0(points)
            cv2.drawContours(drawing_contours, [points], 0, 255, 1)

        # print(points)
        if points is None:
            return None,None
        edge1 = math.pow(points[0][0]-points[1][0],2) + math.pow(points[0][1]-points[1][1],2)
        edge2 = math.pow(points[1][0]-points[2][0],2) + math.pow(points[1][1]-points[2][1],2)
        if edge1>edge2:
            point0_x = int((points[1][0] + points[2][0]) / 2)
            point0_y = int((points[1][1] + points[2][1]) / 2)
            point1_x = int((points[0][0] + points[3][0]) / 2)
            point1_y = int((points[0][1] + points[3][1]) / 2)
        else:
            point0_x = int((points[0][0] + points[1][0]) / 2)
            point0_y = int((points[0][1] + points[1][1]) / 2)
            point1_x = int((points[2][0] + points[3][0]) / 2)
            point1_y = int((points[2][1] + points[3][1]) / 2)

        if self.debug_type > 1:
            # contour_path = os.path.join(self.output_dir, 'contour_' + self.image_name)
            # cv2.imwrite(contour_path, drawing_contours)
            r = 3
            thickness = 2
            cv2.line(drawing_contours, (int(point0_x) - r, int(point0_y) - r),
                     (int(point0_x) + r, int(point0_y) + r), 255, thickness)
            cv2.line(drawing_contours, (int(point0_x) - r, int(point0_y) + r),
                     (int(point0_x) + r, int(point0_y) - r), 255, thickness)
            cv2.line(drawing_contours, (int(point1_x) - r, int(point1_y) - r),
                     (int(point1_x) + r, int(point1_y) + r), 255, thickness)
            cv2.line(drawing_contours, (int(point1_x) - r, int(point1_y) + r),
                     (int(point1_x) + r, int(point1_y) - r), 255, thickness)
            minrect_path = os.path.join(self.output_dir, 'minrect_' + self.image_name)
            cv2.imwrite(minrect_path, drawing_contours)
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # erode = cv2.erode(thresh,kernel=kernel)
        # if self.debug_type > 1:
        #     erode_path = os.path.join(self.output_dir, 'erode2_' + self.image_name)
        #     cv2.imwrite(erode_path, erode)
        #
        # thresh = cv2.Canny(erode,225,255,apertureSize=3)
        # # _, thresh = cv2.threshold(mask_img, 225, 255, cv2.THRESH_BINARY)
        # if self.debug_type > 1:
        #     thresh_path = os.path.join(self.output_dir, 'thresh2_' + self.image_name)
        #     cv2.imwrite(thresh_path, thresh)
        #
        # # 计算直线
        # lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=2)
        # # lines = cv2.HoughLines(thresh, 1, np.pi / 180, 30, 0, 0 )
        # print(lines.shape)
        # lines1 = lines[:, 0, :]  # 提取为二维
        # if self.debug_type > 1:
        #     lines_img = self.gray_img.copy()
        #     for x1, y1, x2, y2 in lines1[:]:
        #         cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        #     lines_path = os.path.join(self.output_dir, 'lines_' + self.image_name)
        #     cv2.imwrite(lines_path, lines_img)

        print(self.depth_data.shape)
        return (point0_x,point0_y,self.depth_data[point0_y,point0_x]),(point1_x,point1_y,self.depth_data[point1_y,point1_x])

    def _get_point_z(self,point):
        return self.depth_data[point[0],point[1]]

    def _caculate_angle(self, x, y):
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y)/(Lx*Ly)
        angle = np.arccos(cos_angle)
        angle2 = angle*360/2/np.pi
        return angle2

    def find_cylinder(self):

        # 寻找标定点
        point0,point1 = self._find_lines()
        print(point0)
        print(point1)

        # 计算圆柱体姿态
        if point0[2]>point1[2]:
            x0 = point1[0] - point0[0]
            y0 = point0[1] - point1[1]
            z0 = point0[2] - point1[2]
        else:
            x0 = point0[0] - point1[0]
            y0 = point1[1] - point0[1]
            z0 = point1[2] - point0[2]

        print(x0,y0,z0)

        beta = self._caculate_angle(np.array([x0,y0,z0]),np.array([0,abs(y0),abs(z0)]))
        alpha = self._caculate_angle(np.array([0,y0,z0]),np.array([0,abs(y0),0]))
        print(alpha,beta)

        # 计算圆心位置
        x = None
        y = None
        z = None
        return [alpha, beta, x, y, z]


