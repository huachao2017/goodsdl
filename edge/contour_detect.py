import cv2
import numpy as np
import os

def _find_contour(img, image_name, output_dir=None, debug_type=0, thresh_x = 120, morphology = False, channel='all'):
    # param@debug_type:0,not debug; 1,store bbox file; 2,store middle caculate file; 3,show window
    source = img.copy()
    compress = 1
    if source.shape[0]>1000 or source.shape[1]>1000:
        source = cv2.pyrDown(source)
        compress = 2
        if source.shape[0]>1000 or source.shape[1]>1000:
            source = cv2.pyrDown(source)
            compress = 4

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
        if area > max_area*.8: # 去除面积过大的物体
            continue
        area_to_contour[area] = cnt
        # print(tuple(cnt[cnt[:, :, 0].argmin()][0]))
        # print(tuple(cnt[cnt[:, :, 0].argmax()][0]))

    # step6: caculate bounding box and draw contours
    drawing_contours = np.zeros(source.shape, np.uint8)
    areas = sorted(area_to_contour, reverse=True)
    index = 0
    boxes = []
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
        # if debug_type > 1:
        #     drawing_contours = cv2.rectangle(drawing_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if debug_type>1:
        contours_path = os.path.join(output_dir, channel+'_'+'contours_'+image_name)
        cv2.imwrite(contours_path, drawing_contours)

    # step7: nms and store last bounding box
    boxes = _non_max_suppression_fast(boxes, 0.5)
    boxes = boxes*compress
    if debug_type > 0:
        for box in boxes:
            output = cv2.rectangle(source, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    if debug_type>0 and len(boxes)>0:
        output_path = os.path.join(output_dir, channel+'_'+'bbox_'+image_name)
        cv2.imwrite(output_path, output)

    if debug_type>2:
        cv2.imshow(channel+'_'+'input', sharpen)
        cv2.imshow(channel+'_'+'edges', edges)
        cv2.imshow(channel+'_'+'thresh1', thresh1)
        if morphology:
            cv2.imshow(channel+'_'+'thresh2', thresh2)
        cv2.imshow(channel+'_'+'drawing_contours', drawing_contours)
        if len(boxes) > 0:
            cv2.imshow(channel+'_'+'output', output)

    return boxes

def _non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    # if boxes.dtype.kind == "i":
    #     boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def find_contour(input_path, area=None, output_dir=None, debug_type=0, thresh_x = 120, morphology = False):
    image_dir, image_name = os.path.split(input_path)
    if output_dir is None:
        output_dir = image_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # step0: read image
    img = cv2.imread(input_path)
    if area is not None:
        area_img = img[area[1]:area[3],area[0]:area[2],:]
    all_boxes = _find_contour(area_img, image_name, output_dir, debug_type=debug_type, thresh_x=thresh_x, morphology=morphology, channel='all')
    r_boxes = _find_contour(area_img, image_name, output_dir, debug_type=debug_type, thresh_x=thresh_x, morphology=morphology, channel='r')
    g_boxes = _find_contour(area_img, image_name, output_dir, debug_type=debug_type, thresh_x=thresh_x, morphology=morphology, channel='g')
    b_boxes = _find_contour(area_img, image_name, output_dir, debug_type=debug_type, thresh_x=thresh_x, morphology=morphology, channel='b')


    concate_boxes = np.concatenate((all_boxes,r_boxes,g_boxes,b_boxes))
    concate_boxes = _non_max_suppression_fast(concate_boxes, 0.5)
    if area is not None:
        concate_boxes[:, 0] = concate_boxes[:, 0]+area[0]
        concate_boxes[:, 1] = concate_boxes[:, 1] + area[1]
        concate_boxes[:, 2] = concate_boxes[:, 2] + area[0]
        concate_boxes[:, 3] = concate_boxes[:, 3] + area[1]
    if debug_type > 0:
        for box in concate_boxes:
            output = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    if debug_type > 0 and len(concate_boxes)>0:
        output_path = os.path.join(output_dir, 'output_'+image_name)
        cv2.imwrite(output_path, output)

    scores = np.ones((len(concate_boxes)))
    return img, concate_boxes, scores


if __name__ == "__main__":
    # Enter the input image file
    image_dir, _ = os.path.split(os.path.realpath(__file__))
    image_path = os.path.join(image_dir, "7.jpg")
    # image_path = os.path.join(image_dir, "7_1.jpg")
    output_dir = os.path.join(image_dir, 'contour')

    # cv2.createTrackbar('canny threshold2:','input',x2,max_x,find_contour_x2)
    import time
    time0 = time.time()
    _,boxes,_ = find_contour(image_path, area=(69,86,901,516),output_dir=output_dir,debug_type=1)
    # _,boxes,_ = find_contour(image_path, output_dir=output_dir,debug_type=1)
    time1 = time.time()
    print('%.2f, %d' %(time1-time0, len(boxes)))

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
