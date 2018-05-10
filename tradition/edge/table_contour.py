import os
import cv2
import numpy as np
import os
import shutil
import time
import random
import math

def _find_area_max_hist(source, rect):
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


def _find_contour(img, image_name, output_dir=None, debug_type=0, channel='all'):
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
        table_thresh, _ = _find_area_max_hist(source,
                                [top_left[0],
                                 top_left[1],
                                 block_width,
                                 block_height])
        if debug_type > 1:
            print(table_thresh)
        table_total_val += table_thresh

    table_var = int(table_total_val/len(table_top_left_arr))
    if debug_type > 1:
        print(table_var)

    corner_vals = []
    corner_top_left_arr = [[0,0],[width - block_width-1,0],[0,height-block_height-1],[width - block_width-1,height-block_height-1]]

    # print(top_left_arr)
    for top_left in corner_top_left_arr:
        _, corner_thresh = _find_area_max_hist(source,
                                [top_left[0],
                                 top_left[1],
                                 block_width,
                                 block_height])
        if debug_type > 1:
            print(corner_thresh)
        corner_vals.append(corner_thresh)

    thresh = 0
    min_corner_var = min(corner_vals)
    for corner_var in corner_vals:
        if corner_var - min_corner_var < table_var - corner_var:
            if corner_var > thresh:
                thresh = corner_var

    if debug_type > 1:
        print(thresh)
    _, thresh_img = cv2.threshold(source, thresh, 255, cv2.THRESH_BINARY)
    if debug_type>1:
        thresh_path = os.path.join(output_dir, channel+'_'+'thresh_'+image_name)
        cv2.imwrite(thresh_path, thresh_img)

    # step2 erode:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode = cv2.erode(thresh_img, kernel)
    erode = cv2.erode(erode, kernel)
    if debug_type>1:
        erode_path = os.path.join(output_dir, channel+'_'+'erode_'+image_name)
        cv2.imwrite(erode_path, erode)

    # step3: contour and filter
    _, contours, _ = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if debug_type>1:
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
    drawing_contours = np.zeros(source.shape, np.uint8)
    areas = sorted(area_to_contour, reverse=True)
    index = 0
    boxes = []
    min_rectes = []
    for area in areas:
        index += 1
        # if index > top_n:
        #     break
        cnt = area_to_contour[area]
        color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
        if debug_type > 1:
            cv2.drawContours(drawing_contours, [cnt], 0, color, 1)
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x,y,x+w,y+h])
        min_rect = cv2.minAreaRect(cnt)
        min_rectes.append(min_rect)
        break
        # if debug_type > 1:
        #     drawing_contours = cv2.rectangle(drawing_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if debug_type>1:
        contours_path = os.path.join(output_dir, channel+'_'+'contours_'+image_name)
        cv2.imwrite(contours_path, drawing_contours)
    return None

def find_contour(input_path, output_dir=None, debug_type=0):
    image_dir, image_name = os.path.split(input_path)
    if output_dir is None:
        output_dir = image_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # step0: read image
    img = cv2.imread(input_path)

    contour = _find_contour(img, image_name, output_dir, debug_type=debug_type, channel='all')

    return contour


def _inner_find_one(image_path, output_dir, debug_type=2):
    time0 = time.time()
    contour = find_contour(image_path,  output_dir=output_dir, debug_type=debug_type)
    time1 = time.time()
    print('%s:%.2f' %(image_path,time1-time0))

if __name__ == "__main__":
    # Enter the input image file
    base_dir, _ = os.path.split(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir,'images')
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(base_dir, 'table_contour')
    if os.path.isdir(output_dir):
        for image in os.listdir(output_dir):
            tmp_path = os.path.join(output_dir, image)
            if os.path.splitext(tmp_path)[-1] == '.jpg':
                os.remove(tmp_path)

    # for test
    image_path = os.path.join(image_dir, "t1.jpg")
    _inner_find_one(image_path, output_dir, debug_type=2)
