import os
import cv2
import numpy as np
import os
import shutil
import time
import random
import math
import functools

def _find_minrect(img, image_name, output_dir=None, debug_type=0, thresh_x = 120, morphology = False, channel='all', overlapthresh=.3):
    # param@debug_type:0,not debug; 1,store bbox file; 2,store middle caculate file; 3,show window
    source = img.copy()

    # step1: blur image
    max_area = source.shape[0] * source.shape[1]
    # Apply gaussian blur to the grayscale image
    # blur = cv2.pyrMeanShiftFiltering(source, 31, 91)
    sharpen = source
    # blur = cv2.pyrMeanShiftFiltering(source, 21, 51)
    # kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
    #                            [-1,2,2,2,-1],
    #                            [-1,2,8,2,-1],
    #                            [-2,2,2,2,-1],
    #                            [-1,-1,-1,-1,-1]])/8.0
    # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(sharpen, -1, kernel_sharpen)
    if channel == 'all':
        sharpen = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)
    else:
        b, g, r = cv2.split(sharpen)
        if channel == 'b':
            sharpen = b
        elif channel == 'g':
            sharpen = g
        elif channel == 'r':
            sharpen = r
        else:
            sharpen = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)


    # 双向滤波比较不错
    # blur = cv2.bilateralFilter(blur, 3, 30, 30)
    # blur = cv2.split(blur)[0]
    # blur = cv2.equalizeHist(blur)
    # blur = cv2.GaussianBlur(blur, (5, 5), 0)
    if debug_type>1:
        sharpen_path = os.path.join(output_dir, channel+'_'+'sharpen_'+image_name)
        cv2.imwrite(sharpen_path, sharpen)

    # step2: sobel caculate edges
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    x = cv2.Sobel(sharpen, cv2.CV_64F, 1, 0, ksize=-1)
    y = cv2.Sobel(sharpen, cv2.CV_64F, 0, 1, ksize=-1)
    edges = cv2.subtract(x, y)
    edges = cv2.convertScaleAbs(edges)
    # absX = cv2.convertScaleAbs(x)  # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    #
    # edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # edges = cv2.bilateralFilter(edges, 5, 75, 75)
    # edges = cv2.GaussianBlur(edges, (5, 5), 0)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    # edges = cv2.GaussianBlur(edges, (9, 9),0)
    if debug_type>1:
        edges_path = os.path.join(output_dir, channel+'_'+'edges_'+image_name)
        cv2.imwrite(edges_path, edges)

    # step3: binary edges
    _, thresh1 = cv2.threshold(edges, thresh_x, 255, cv2.THRESH_BINARY)
    thresh2 = thresh1
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh2 = cv2.erode(thresh2, kernel)
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    # thresh2 = cv2.dilate(thresh2, kernel)
    # thresh2 = cv2.dilate(thresh2, kernel)
    # thresh2 = cv2.dilate(thresh2, kernel)
    # thresh2 = cv2.dilate(thresh2, kernel)
    # thresh2 = cv2.dilate(thresh2, kernel)
    # thresh2 = cv2.erode(thresh2, kernel)
    # thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    # _, thresh = cv2.threshold(gray, x, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    if debug_type>1:
        thresh1_path = os.path.join(output_dir, channel+'_'+'thresh1_'+image_name)
        cv2.imwrite(thresh1_path, thresh1)
        if morphology:
            thresh2_path = os.path.join(output_dir, channel+'_'+'thresh2_' + image_name)
            cv2.imwrite(thresh2_path, thresh2)

    # Find the edges
    # edges = cv2.Canny(gray,x1,x2)
    # edges = gray

    # step4: Detect contours
    _, contours, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print('find contours: {}'.format(len(contours)))
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
        area = (bottommost-topmost) * (rightmost-leftmost)
        if area < max_area/100: # 去除面积过小的物体
            continue
        # if area > max_area*.9: # 去除面积过大的物体
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
        if debug_type > 1:
            cv2.drawContours(drawing_contours, [cnt], 0, color, 1)
        min_rect = cv2.minAreaRect(cnt)
        min_rectes.append(min_rect)
        # if debug_type > 1:
        #     drawing_contours = cv2.rectangle(drawing_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if debug_type>1:
        contours_path = os.path.join(output_dir, channel+'_'+'contours_'+image_name)
        cv2.imwrite(contours_path, drawing_contours)

    # step7: nms min rect
    # min_rectes = _non_max_suppression_minrect(min_rectes, .3)
    if debug_type > 1 and len(min_rectes) > 0:
        minrect = np.copy(source)
        for min_rect in min_rectes:
            points = cv2.boxPoints(min_rect)
            points = np.int0(points)
            minrect = cv2.drawContours(minrect,[points],0,(0, 0, 255),1)
        minrect_path = os.path.join(output_dir, channel+'_'+'minrect_'+image_name)
        cv2.imwrite(minrect_path, minrect)


    if debug_type>2:
        cv2.imshow(channel+'_'+'input', sharpen)
        cv2.imshow(channel+'_'+'edges', edges)
        cv2.imshow(channel+'_'+'thresh1', thresh1)
        if morphology:
            cv2.imshow(channel+'_'+'thresh2', thresh2)
        cv2.imshow(channel+'_'+'drawing_contours', drawing_contours)

    return min_rectes


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

    if debug:
        print('source:{}'.format(''.join(map(str,p))))

    pp = sorted(p, key=functools.cmp_to_key(lambda x, y: cmp(x, y, c)))
    if debug:
        print('sorted:{}'.format(''.join(map(str,pp))))
    r = np.full((len_p, 2), 0.0, dtype='float32')
    for i in range(len(pp)):
        r[i][0] = pp[i].x
        r[i][1] = pp[i].y
    r2 = cv2.contourArea(r)
    return (r2,r)

def _non_max_suppression_minrect(min_rectes, overlapThresh, debug = False, source_image=None, output_dir=None):
    # if there are no boxes, return an empty list
    if len(min_rectes) == 0:
        return []


    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area_list = []
    for min_rect in min_rectes:
        area_list.append(min_rect[1][0]*min_rect[1][1])

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
        if debug:
            print('main:index:%d, center:%d,%d;w*h:%d,%d;theta:%d' % ( i,
                min_rectes[i][0][0], min_rectes[i][0][1], min_rectes[i][1][0], min_rectes[i][1][1], min_rectes[i][2]))

        need_del = []
        for index in range(last):
            j = idxs[index]
            # print(j, min_rectes[j])
            intersection_area, intersection = _rotated_rectangle_intersection_area(min_rectes[j],min_rectes[i])

            # compute the ratio of overlap
            overlap = intersection_area / area[j]
            if debug:
                print('%d,%d,%.2f' %(j,i,overlap))
            if overlap > overlapThresh:
                need_del.append(index)
            elif debug:
                print('comp:index:%d, center:%d,%d;w*h:%d,%d;theta:%d' % (j,
                                                                     min_rectes[j][0][0], min_rectes[j][0][1],
                                                                     min_rectes[j][1][0], min_rectes[j][1][1],
                                                                     min_rectes[j][2]))
                drawing_contours = np.zeros(source_image.shape, np.uint8)
                points = cv2.boxPoints(min_rectes[j])
                points = np.int0(points)
                color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
                cv2.drawContours(drawing_contours, [points], 0, color, 1)
                points = cv2.boxPoints(min_rectes[i])
                points = np.int0(points)
                color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
                cv2.drawContours(drawing_contours, [points], 0, color, 1)
                cv2.drawContours(drawing_contours, [np.asarray(intersection,np.int0)], 0, color, 1)
                output_path = os.path.join(output_dir, 'intersection_%d_%d.jpg' % (j, i))
                cv2.imwrite(output_path, drawing_contours)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.asarray(need_del))))

    if debug:
        print(pick)
    # return only the bounding boxes that were picked using the
    # integer data type
    ret_min_rectes = []
    for index in pick:
        ret_min_rectes.append(min_rectes[index])
    return ret_min_rectes


def _get_mask_image(rgb_img, depth_path, table_z, image_name, output_dir=None, debug_type=1):

    mask_rgb_img = rgb_img
    if depth_path:
        depth_source_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_source_img[:, :, 0] + depth_source_img[:, :, 1] * 256 + depth_source_img[:, :, 2] * 256 * 256
        depth_img = np.expand_dims(depth_img,2)
        depth_img = depth_img.repeat(3,axis=2)
        mask_code = np.zeros(rgb_img.shape, np.uint8)
        mask_rgb_img = np.where(depth_img>table_z, mask_code, rgb_img)
        mask_rgb_img = np.where(mask_rgb_img < 10, mask_code, mask_rgb_img)
        if debug_type > 0:
            output_path = os.path.join(output_dir, '_mask_' + image_name)
            cv2.imwrite(output_path, mask_rgb_img)

    return mask_rgb_img

def find_contour(rgb_path, depth_path, table_z,output_dir=None, debug_type=1, thresh_x = 120, morphology = False, overlapthresh=.3):
    image_dir, image_name = os.path.split(rgb_path)
    if output_dir is None:
        output_dir = image_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # step0: read image
    rgb_img = cv2.imread(rgb_path)
    mask_rgb_img = _get_mask_image(rgb_img, depth_path, table_z, image_name, output_dir, debug_type)

    concate_minrectes = _find_minrect(mask_rgb_img, image_name, output_dir, debug_type=debug_type, thresh_x=thresh_x, morphology=morphology, channel='all', overlapthresh=overlapthresh)
    if debug_type > 1:
        print('{}'.format(len(concate_minrectes)))

    if debug_type > 1 and len(concate_minrectes)>0:
        drawing_contours = np.zeros(mask_rgb_img.shape, np.uint8)
        for minrect in concate_minrectes:
            points = cv2.boxPoints(minrect)
            points = np.int0(points)
            color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
            cv2.drawContours(drawing_contours, [points], 0, color, 1)
        output_path = os.path.join(output_dir, '_contour_' + image_name)
        cv2.imwrite(output_path, drawing_contours)

    suppression_minrectes = _non_max_suppression_minrect(concate_minrectes, overlapthresh, debug=(debug_type>1),source_image=mask_rgb_img, output_dir=output_dir)
    ret_minrectes = []
    ret_boxes = []
    ret_z = []
    for minrect in suppression_minrectes:
        ret_z.append(table_z) # FIXME
        if minrect[1][0]>minrect[1][1]:
            ret_minrectes.append(((minrect[0][0],minrect[0][1]),(minrect[1][1],minrect[1][0]),90 + minrect[2]))
        else:
            ret_minrectes.append(minrect)
    for minrect in ret_minrectes:
        points = cv2.boxPoints(minrect)
        x, y, w, h = cv2.boundingRect(points)
        ret_boxes.append([x,y,x+w,y+h])

    if debug_type > 0 and len(ret_minrectes)>0:
        output = rgb_img
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
            cv2.putText(output, '%d,%d,%.2f' % (minrect[0][0],minrect[0][1],minrect[2]), (0, 12*index), font, 0.4, (255, 255, 255), 1)
        output_path = os.path.join(output_dir, '_output_'+image_name)
        cv2.imwrite(output_path, output)

    # scores = np.ones((len(concate_minrectes)))
    return ret_minrectes, ret_z, ret_boxes


def _inner_find_one(rgb_path, depth_path, table_z, output_dir, debug_type=2):
    time0 = time.time()
    min_rectes, z, boxes= find_contour(rgb_path, depth_path, table_z,output_dir=output_dir, debug_type=debug_type, overlapthresh=.7)
    # _,boxes,_ = find_contour(image_path, output_dir=output_dir,debug_type=1)
    time1 = time.time()
    print('%s:%.2f, %d' % (rgb_path, time1 - time0, len(min_rectes)))
    index = 0
    for min_rect in min_rectes:
        print('center: %d,%d; w*h:%d,%d; theta:%d; z:%d, boxes: x1:%d, y1:%d, x2:%d, y2:%d' %(min_rect[0][0],min_rect[0][1],min_rect[1][0],min_rect[1][1],min_rect[2], z[index], boxes[index][0], boxes[index][1], boxes[index][2], boxes[index][3]))
        index += 1

if __name__ == "__main__":
    # Enter the input image file
    base_dir, _ = os.path.split(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir,'images')
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(image_dir, 'contour')
    if os.path.isdir(output_dir):
        for image in os.listdir(output_dir):
            tmp_path = os.path.join(output_dir, image)
            if os.path.splitext(tmp_path)[-1] == '.png' or os.path.splitext(tmp_path)[-1] == '.jpg':
                os.remove(tmp_path)

    # for test
    rgb_path = os.path.join(image_dir, "02.jpg")
    depth_path = os.path.join(image_dir, "02_d.png")
    _inner_find_one(rgb_path, depth_path, 1235,output_dir,  debug_type=1)


    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
