import numpy as np
import cv2
import imutils

def jarcode_red_detection(img_color): 

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,  (0,103,0), (179,255,255))

    maskr1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    maskr2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask_red = cv2.bitwise_or(maskr1, maskr2 )

    croped = cv2.bitwise_and(img_color, img_color, mask=255-mask)
    cropped = cv2.bitwise_and(croped, croped, mask=mask_red)

    gradient = cv2.erode(cropped, None, iterations = 4)
    gradient = cv2.dilate(cropped, None, iterations = 4)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

    (_, thresh) = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY)
    closed = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts)==0 : return 
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
    resized2 = cv2.resize(img_color, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("result", resized2)
    return jarcode_red_boxsize(box,img_color)

def jarcode_red_boxsize(box,img): 
    height = max(pow(box[0][0]-box[1][0],2)+pow(box[0][1]-box[1][1],2),pow(box[1][0]-box[2][0],2)+pow(box[1][1]-box[2][1],2))
    width = min(pow(box[0][0]-box[1][0],2)+pow(box[0][1]-box[1][1],2),pow(box[1][0]-box[2][0],2)+pow(box[1][1]-box[2][1],2))

    
    low_height = pow(img.shape[0]*0.2,2)
    high_height =  pow(img.shape[0]*0.4,2)

    low_width = pow(img.shape[1]*0.2,2)
    high_width = pow(img.shape[1]*0.67,2)

    # print(img.shape[0],img.shape[1])
    # print(low_height,high_height, height)
    # print(low_width,high_width,width)

    h_flag = False
    w_flag = False

    if (low_height<=height) and (height<=high_height):
        h_flag = True
        print("높이 정상")
    else:
        print("높이 불량") 
        return False
    
    if (low_width<=width) and (width<=high_width):
        w_flag = True
        print("너비 정상")
    else:         
        print("너비 불량")
        return False
    if h_flag and w_flag:
        return True
    else: return False