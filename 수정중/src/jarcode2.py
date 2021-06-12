import cv2 as cv2
import numpy as np
import imutils
import pyzbar.pyzbar as pyzbar


img = cv2.imread('./BRC.jpg')

if img is not None:
    print('Original Dimensions : ',img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradY, gradX)
    gradient = cv2.convertScaleAbs(gradient)
    decode=pyzbar.decode(gray)
    print(decode)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

