#!/usr/bin/python

import numpy as np
import cv2
from morphology import *

def nothing():
    pass

def create_all_trackbar():
    cv2.namedWindow('Control')
    cv2.createTrackbar('k_gauss','Control',0,50,nothing)
    cv2.createTrackbar('k_open','Control',0,50,nothing)
    cv2.createTrackbar('f','Control',0,255,nothing)

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
img_manual = cv2.imread('21_manual1.tif',0)
cv2.namedWindow('Image')

create_all_trackbar()

img_b, img_g, img_r = cv2.split(img)

while (1):

    k = cv2.getTrackbarPos('k_gauss','Control')
    k = 2*k+1

    k_open = cv2.getTrackbarPos('k_open','Control')
    k_open = k_open*2+1

    f = cv2.getTrackbarPos('f','Control')

    #img_b, img_g, img_r = cv2.split(img)
    #get the gree chanel
    img2 = img[:,:,1]
    TH = 7

    img2 = tophat_image(img2,TH)
    img2 = cv2.GaussianBlur(img2,(k,k),0)
    img2[img2 > f] = 255


    #img2 = open_img(img2,k_open)

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.rectangle(img2,(8,530),(130,570),(0,255,0),-1)
    #cv2.putText(img,'OpenCV',(10,560), font,1,(0,0,0),2,False)

    cv2.imshow('Image', img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.imwrite('img_train.tif',img2)
        cv2.destroyAllWindows()
        break
