import cv2 as cv2
import numpy as np
import imutils
#img = cv2.imread('./hello2.jpg')
def jarcode_black_detection(img):
    if img is not None:
        print('Original Dimensions : ',img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradY, gradX)
        gradient = cv2.convertScaleAbs(gradient)

    
        print('Resized Dimensions : ',gradient.shape)
        blurred = cv2.blur(gradient, (5, 5))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
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

        cv2.drawContours(img, [box], -1, (0, 255, 255), 3)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow("filtered image", resized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

