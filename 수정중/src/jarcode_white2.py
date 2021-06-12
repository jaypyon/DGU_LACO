import numpy as np
import cv2
import imutils

def jarcode_white_boxsize(box,img): 
    height = max(pow(box[0][0]-box[1][0],2)+pow(box[0][1]-box[1][1],2),pow(box[1][0]-box[2][0],2)+pow(box[1][1]-box[2][1],2))
    width = min(pow(box[0][0]-box[1][0],2)+pow(box[0][1]-box[1][1],2),pow(box[1][0]-box[2][0],2)+pow(box[1][1]-box[2][1],2))

    
    low_height = pow(img.shape[0]*0.5,2)
    high_height =  pow(img.shape[0]*0.9,2)

    low_width = pow(img.shape[1]*0.5,2)
    high_width = pow(img.shape[1]*0.9,2)

    print(img.shape[0],img.shape[1])
    print(low_height,high_height, height)
    print(low_width,high_width,width)

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

img_color = cv2.imread('./testhell6.jpg') 
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(img_hsv, (136,98,0), (179,255,255))
mask_white = cv2.inRange(img_hsv, (48,0,151), (87,170,255))
mask_final = cv2.bitwise_and(255-mask,mask_white)

croped = cv2.bitwise_and(img_color, img_color, mask=mask_final)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
closed = cv2.dilate(closed, None, iterations = 4)
closed = cv2.erode(closed, None, iterations = 4)


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
print(jarcode_white_boxsize(box,img_color))

scale_percent = 50 # percent of original size
width = int(img_color.shape[1] * scale_percent / 100)
height = int(img_color.shape[0] * scale_percent / 100)
dim = (width, height)
    
# resize image
resized = cv2.resize(img_color, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(croped, dim, interpolation = cv2.INTER_AREA)

cv2.imshow('img_color1', resized)
cv2.imshow('img_color2', resized2)
cv2.waitKey(0)
cv2.destroyAllWindows()

