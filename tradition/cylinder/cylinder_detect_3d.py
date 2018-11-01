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
            thresh_path = os.path.join(self.output_dir, 'thresh_' + self.image_name)
            cv2.imwrite(thresh_path, thresh)

        # 计算直线
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=5)
        lines1 = lines[:, 0, :]  # 提取为二维
        if self.debug_type > 1:
            lines_img = self.gray_img.copy()
            for x1, y1, x2, y2 in lines1[:]:
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            lines_path = os.path.join(self.output_dir, 'lines_' + self.image_name)
            cv2.imwrite(lines_path, lines_img)

        return lines1

    def _get_target_z(self,points):
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

    def find_cylinder(self):

        # 寻找标定点
        points = self._find_lines()

        # 计算圆柱体姿态和上圆心位置
        alpha = None
        beta = None
        gama = None
        x = None
        y = None
        z = None
        return [alpha, beta, gama, x, y, z]


