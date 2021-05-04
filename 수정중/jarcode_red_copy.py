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

gradient = cv2.convertScaleAbs(cropped)
cv2.imshow('grad', gradient)


cv2.waitKey(0)
cv2.destroyAllWindows()