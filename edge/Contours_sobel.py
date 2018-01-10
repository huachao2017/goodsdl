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
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
    for area in areas:
        index += 1
        if index > top_n:
            break
        cnt = area_to_contour[area]
        # color = np.random.randint(0, 255, (3)).tolist()  # Select a random color
        cv2.drawContours(drawing, [cnt], 0, (255,0,0), 1)
        x, y, w, h = cv2.boundingRect(cnt)
        drawing = cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('output', drawing)
    cv2.imshow('input', blur)
    cv2.imshow('middle_1', edges)
    cv2.imshow('middle_2', thresh)

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
