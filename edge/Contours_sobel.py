import cv2
import numpy as np
import skimage
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters import gaussian

def find_contour(thresh_x):
    x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.dilate(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    # edges = cv2.erode(edges, kernel)
    _, thresh = cv2.threshold(edges, thresh_x, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    # _, thresh = cv2.threshold(gray, x, 255, cv2.THRESH_BINARY_INV)

    # Find the edges
    # edges = cv2.Canny(gray,x1,x2)
    # edges = gray

    # Image to draw the contours
    drawing = np.zeros(img.shape, np.uint8)

    # Detect and store the contours
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('find contours: {}'.format(len(contours)))
    # print('first contour: {}'.format(contours[0]))

    # Display the contours using different colors
    area_to_contour = {}
    for cnt in contours:
        leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
        rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
        topmost = cnt[cnt[:, :, 1].argmin()][0][1]
        bottommost = cnt[cnt[:, :, 1].argmax()][0][1]

        # print('%d,%d,%d,%d' %(leftmost,rightmost,topmost,bottommost))
        # return
        area = (bottommost-topmost) * (rightmost-leftmost)
        area_to_contour[area] = cnt
        # print(tuple(cnt[cnt[:, :, 0].argmin()][0]))
        # print(tuple(cnt[cnt[:, :, 0].argmax()][0]))

    areas = sorted(area_to_contour, reverse=True)
    index = 0
    boxes = []
    for area in areas:
        index += 1
        if index == 1:# TODO 第一条去除还得研究更好的方法
            continue
        if index > top_n:
            break
        cnt = area_to_contour[area]
        color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
        cv2.drawContours(drawing, [cnt], 0, color, 1)
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x,y,x+w,y+h])
        drawing = cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)

    boxes = non_max_suppression_fast(boxes,0.5)
    for box in boxes:
        drawing = cv2.rectangle(drawing, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
        output = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)


    cv2.imshow('input', blur)
    cv2.imshow('middle_1', edges)
    cv2.imshow('middle_2', thresh)
    cv2.imshow('middle3', drawing)
    cv2.imshow('output', output)


def non_max_suppression_fast(boxes, overlapThresh):
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

# Enter the input image file
img_name = "1_1.jpg"
img = cv2.imread(img_name)
print(img.shape)

# Apply gaussian blur to the grayscale image
# blur = cv2.pyrMeanShiftFiltering(img, 31, 91)
blur = cv2.pyrMeanShiftFiltering(img, 21, 51)
blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(blur, (5, 5), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# blur = cv2.dilate(blur, kernel)
# blur = cv2.dilate(blur, kernel)
# blur = cv2.dilate(blur, kernel)
# blur = cv2.erode(blur, kernel)
# blur = cv2.erode(blur, kernel)
# blur = cv2.erode(blur, kernel)

cv2.namedWindow('input', cv2.WINDOW_NORMAL)

# Set the default and max threshold value
thresh_x = 30
max_x = 100
top_n = 20
cv2.createTrackbar('binary thresh:', 'input', thresh_x, max_x, find_contour)
# cv2.createTrackbar('canny threshold2:','input',x2,max_x,find_contour_x2)
find_contour(thresh_x)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
