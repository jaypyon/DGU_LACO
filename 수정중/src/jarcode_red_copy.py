import numpy as np
import cv2
import imutils

img_color = cv2.imread("BRC.jpg")
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환
#blurred = cv2.blur(img_hsv, (10, 10))
mask = cv2.inRange(img_hsv,  (0,103,0), (179,255,255))
mask_red= cv2.inRange(img_hsv,  (0,50,173), (23,255,255))
# mask = cv2.bitwise_or(mask1, mask)
croped = cv2.bitwise_and(img_color, img_color, mask=255-mask)
cropped = cv2.bitwise_and(croped, croped, mask=mask_red)
#gradient = cv2.convertScaleAbs(cropped)

gradient = cv2.erode(cropped, None, iterations = 4)
gradient = cv2.dilate(cropped, None, iterations = 4)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
#blurred = cv2.blur(gradient, (3, 3))
#(_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
(_, thresh) = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY)
closed = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_color, [box], -1, (0, 255, 255), 3)

scale_percent = 50 # percent of original size
width = int(img_color.shape[1] * scale_percent / 100)
height = int(img_color.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
#resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(img_color, dim, interpolation = cv2.INTER_AREA)

cv2.imshow('grad', resized2)

#resized2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

##cv2.imshow('grad', resized2)

cv2.waitKey(0)
cv2.destroyAllWindows()