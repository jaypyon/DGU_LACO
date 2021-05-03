import numpy as np
import cv2
import imutils

img_color = cv2.imread('./correct_2_red.jpg') 
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor hsv
#blurred = cv2.blur(img_hsv, (10, 10))
mask = cv2.inRange(img_hsv, (0,0,168), (172,111,255))
#mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
#mask = cv2.bitwise_or(mask1)
croped = cv2.bitwise_and(img_color, img_color, mask=mask)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_color, [box], -1, (0, 255, 255), 3)
print('Resized Dimensions : ',closed.shape)
print(box)

scale_percent = 100 # percent of original size
width = int(img_color.shape[1] * scale_percent / 100)
height = int(img_color.shape[0] * scale_percent / 100)
dim = (width, height)
    
# resize image
resized = cv2.resize(img_color, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(closed, dim, interpolation = cv2.INTER_AREA)
#resized3 = cv2.resize(croped, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow('img_color', resized3)
cv2.imshow('img_color2', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
