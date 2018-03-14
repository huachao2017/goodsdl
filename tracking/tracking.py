import numpy as np
import cv2
# import imutils
from skimage.measure import compare_ssim
import time
import logging
logger = logging.getLogger("detect")

def compare_same(img_path1,img_path2, score_thresh=.9):
    score, image1, image2, diff, thresh = _compare(img_path1,img_path2)
    if score < score_thresh:
        return False
    else:
        max_area = image1.shape[0] * image1.shape[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]  # if imutils.is_cv2() else cnts[1]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            if x == 0 or x == w: # 边缘不同忽略
                continue
            if y == 0 or y == h: # 边缘不同忽略
                continue
            area = w * h
            if area > max_area / 100:  # 面积较大视为变化
                return False

        logger.info('SSIM: %.5f, %d' %(score, len(cnts)))
        return True

def _compare(img_path1, img_path2, debug = False):
    time0 = time.time()
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)
    image1 = cv2.pyrDown(image1)
    image2 = cv2.pyrDown(image2)

    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    time1 = time.time()
    (score, diff) = compare_ssim(gray1, gray2, full=True, gaussian_weights=True, sigma=5)
    time2 = time.time()

    diff = (diff * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    if debug:
        print('SSIM: %.5f, %.2f, %.2f' %(score,time1-time0,time2-time1))
    return score, image1, image2, diff, thresh

if __name__ == "__main__":
    score,image1,image2,diff,thresh = _compare('images/1.jpg',
                                               'images/2.jpg',
                                               debug=True,
                                               )
    # show the output images
    # loop over the contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] # if imutils.is_cv2() else cnts[1]
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Original", image1)
    cv2.imshow("Modified", image2)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
