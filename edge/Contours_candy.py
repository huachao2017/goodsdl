import cv2
import numpy as np

def find_contour_x1(x):
    find_contour(x,x2)

def find_contour_x2(x):
    find_contour(x1,x)

def find_contour(x1,x2):

    # Find the edges
    edges = cv2.Canny(gray,x1,x2)
    # edges = gray

    # Image to draw the contours
    drawing = np.zeros(img.shape,np.uint8) 
    
    # Detect and store the contours
    _,contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    print('find contours: {}'.format(len(contours)))
    # print('first contour: {}'.format(contours[0]))

    # Display the contours using different colors
    for cnt in contours:
        color = np.random.randint(0,255,(3)).tolist() # Select a random color
        cv2.drawContours(drawing,[cnt],0,color,2)
        cv2.imshow('output',drawing)
    cv2.imshow('input',edges)
    
# Enter the input image file
img_name = "1.jpg"
img = cv2.imread(img_name)

# Apply gaussian blur to the grayscale image
blur = cv2.pyrMeanShiftFiltering(img,31,91)
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('input',cv2.WINDOW_NORMAL)

# Set the default and max threshold value
x1 = 10
x2 = 20
max_x = 255
cv2.createTrackbar('canny threshold1:','input',x1,max_x,find_contour_x1)
cv2.createTrackbar('canny threshold2:','input',x2,max_x,find_contour_x2)
find_contour(x1,x2)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
