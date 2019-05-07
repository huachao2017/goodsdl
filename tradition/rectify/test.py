import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('book.jpg')
rows, cols = img.shape[:2]
# 原图中书本的四个角点
pts1 = np.float32([[69, 163], [704, 62], [162, 675], [970, 411]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 1000], [0, 0], [750, 1000], [750, 0]])
# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(img, M, (750, 1000))
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# img[:, :, ::-1]是将BGR转化为RGB
plt.show()