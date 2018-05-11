import os
import cv2
import numpy as np
import os
import shutil
import time
import random
import math

class TableContour:
    def __init__(self, image_path, output_dir = None, debug_type=0):
        self.image_path = image_path
        self.output_dir = output_dir
        self.debug_type=debug_type
        self.contour = None
        self.contour_img = None
        image_dir, image_name = os.path.split(image_path)
        if output_dir is None:
            self.output_dir = image_dir
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        img = cv2.imread(self.image_path)
        self.image_name = image_name

        self._find_contour(img)

    def _find_area_max_hist(self,source, rect):
        # histogram = cv2.calcHist([area], [0], None, [256], [0, 256])
        # histogram = np.reshape(histogram,[256])
        # print(histogram)
        # maxv = np.argmax(histogram)
        # print(maxv)
        x0,y0,w,h = rect
        x0 = int(x0)
        y0 = int(y0)
        # print(source.shape)
        # print(x0,y0,w,h)

        maxv = 0
        histogram = np.zeros(256)

        for i in range(w):
            x = x0 + i
            for j in range(h):
                y = y0 + j
                m = source[y][x]
                histogram[m] += 1
                if histogram[maxv] < histogram[m]:
                    maxv = m

        mid = histogram[maxv] / 2
        low = maxv
        high = maxv
        while low > 0 and histogram[low] >= mid:
            low = low - 1
        while high < 256 and histogram[high] >= mid:
            high = high + 1

        return low, high


    def _find_contour(self,img, channel='all'):
        # param@debug_type:0,not debug; 1,store bbox file; 2,store middle caculate file; 3,show window
        source = img.copy()

        if channel == 'all':
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            b, g, r = cv2.split(source)
            if channel == 'b':
                source = b
            elif channel == 'g':
                source = g
            elif channel == 'r':
                source = r
            else:
                source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        # step1 thresh:
        block_scale = 10
        width = source.shape[1]
        height = source.shape[0]
        block_width = int(width / block_scale)
        block_height = int(height / block_scale)

        table_total_val = 0
        table_top_left_arr = [[width / 2 - block_width / 2 - 1, height / 2 - block_height / 2 - 1],
                              [width / 4 - block_width / 2 - 1, height / 4 - block_height / 2 - 1],
                              [width / 4 - block_width / 2 - 1, height * 3 / 4 - block_height / 2 - 1],
                              [width * 3 / 4 - block_width / 2 - 1, height / 4 - block_height / 2 - 1],
                              [width * 3 / 4 - block_width / 2 - 1, height * 3 / 4 - block_height / 2 - 1]]
        for top_left in table_top_left_arr:
            table_thresh, _ = self._find_area_max_hist(source,
                                    [top_left[0],
                                     top_left[1],
                                     block_width,
                                     block_height])
            if self.debug_type > 1:
                print(table_thresh)
            table_total_val += table_thresh

        table_var = int(table_total_val/len(table_top_left_arr))
        if self.debug_type > 1:
            print(table_var)

        corner_vals = []
        corner_top_left_arr = [[0,0],[width - block_width-1,0],[0,height-block_height-1],[width - block_width-1,height-block_height-1]]

        # print(top_left_arr)
        for top_left in corner_top_left_arr:
            _, corner_thresh = self._find_area_max_hist(source,
                                    [top_left[0],
                                     top_left[1],
                                     block_width,
                                     block_height])
            if self.debug_type > 1:
                print(corner_thresh)
            corner_vals.append(corner_thresh)

        thresh = 0
        min_corner_var = min(corner_vals)
        for corner_var in corner_vals:
            if corner_var - min_corner_var < table_var - corner_var:
                if corner_var > thresh:
                    thresh = corner_var

        if self.debug_type > 1:
            print(thresh)
        _, thresh_img = cv2.threshold(source, thresh, 255, cv2.THRESH_BINARY)
        if self.debug_type>1:
            thresh_path = os.path.join(self.output_dir, channel+'_'+'thresh_'+self.image_name)
            cv2.imwrite(thresh_path, thresh_img)

        # step2 erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erode = cv2.erode(thresh_img, kernel)
        erode = cv2.erode(erode, kernel)
        if self.debug_type>1:
            erode_path = os.path.join(self.output_dir, channel+'_'+'erode_'+self.image_name)
            cv2.imwrite(erode_path, erode)

        # step3: contour and filter
        _, contours, _ = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if self.debug_type>1:
            print('find contours: {}'.format(len(contours)))

        area_to_contour = {}
        for cnt in contours:
            # cnt = cv2.convexHull(cnt, returnPoints=True)
            leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
            rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
            topmost = cnt[cnt[:, :, 1].argmin()][0][1]
            bottommost = cnt[cnt[:, :, 1].argmax()][0][1]

            # print('%d,%d,%d,%d' %(leftmost,rightmost,topmost,bottommost))
            # return
            area = (bottommost-topmost) * (rightmost-leftmost)
            if area < width*height/10: # 去除面积过小的物体
                continue
            area_to_contour[area] = cnt

        # step6: caculate bounding box and draw contours
        if len(area_to_contour) > 0:
            areas = sorted(area_to_contour, reverse=True)
            self.contour = area_to_contour[areas[0]]
            self.contour_img = np.zeros((height,width), np.uint8)
            cv2.fillPoly(self.contour_img, [self.contour],1)
            # cv2.fillPoly(self.contour_img,)
            if self.debug_type > 1:
                #print(self.contour)
                # drawing_contours = np.zeros(source.shape, np.uint8)
                # cv2.drawContours(drawing_contours, [self.contour], 0, (255,255,255), 1)
                # cv2.fillPoly(drawing_contours, [self.contour], (255, 255, 255))
                contours_path = os.path.join(self.output_dir, channel+'_'+'contours_'+self.image_name)

                cv2.imwrite(contours_path, self.contour_img * 255)

    def check_box(self,x,y,w,h, index, thresh=0.6):
        if self.contour is None:
            return True
        height = self.contour_img.shape[0]
        width = self.contour_img.shape[1]

        box_img = np.zeros((height, width), np.uint8)
        box_contour = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
        box_contour.reshape((4,1,2))
        cv2.fillPoly(box_img, [box_contour], 1)
        intersect_image = self.contour_img*box_img
        box_ratio = np.sum(intersect_image)/(w*h)
        table_ratio = np.sum(intersect_image)/np.sum(self.contour_img)
        if self.debug_type > 0:
            print(index,box_contour, box_ratio)
            # print(self.contour_img.shape)
            # print(self.box_img.shape)
            intersect_path = os.path.join(self.output_dir, 'intersect_%d_%.2f_%.2f_%s' %  (index, box_ratio, table_ratio, self.image_name))
            cv2.imwrite(intersect_path, (self.contour_img+box_img) * 100)
            # only_intersect_path = os.path.join(self.output_dir, 'only_intersect_{}_{}'.format(index,self.image_name))
            # cv2.imwrite(only_intersect_path, intersect_image * 255)

        if box_ratio < thresh:
            return False
        if table_ratio > 0.8:
            return False

        return True

if __name__ == "__main__":
    # Enter the input image file
    base_dir, _ = os.path.split(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir,'images')
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(image_dir, 'table_contour')
    if os.path.isdir(output_dir):
        for image in os.listdir(output_dir):
            tmp_path = os.path.join(output_dir, image)
            if os.path.splitext(tmp_path)[-1] == '.jpg':
                os.remove(tmp_path)

    # for test
    image_path = os.path.join(image_dir, "t1.jpg")
    time0 = time.time()
    table_contour = TableContour(image_path, output_dir=output_dir, debug_type=2)
    time1 = time.time()
    table_contour.check_box(100,100,100,100,1)
    table_contour.check_box(0,0,100,100,2)
    time2 = time.time()
    print('%s:%.2f,%.2f,%.2f' %(image_path, time2-time0, time1-time0, time2-time1))
