import numpy as np
import cv2


def nothing():
    pass

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
cv2.namedWindow('Image')

img_b, img_g, img_r = cv2.split(img)

cv2.createTrackbar('A','Image',1,10,nothing)

while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    A = cv2.getTrackbarPos('A','Image')
    if A == 0:
        A = 1
    img2 = img_g
    w = 9*(A/100.0) -1
    k_passa_alta = (1.0/9)*np.array([[-2,-2,-2],[-2,w,-2],[-2,-2,-2]])
    img2 = cv2.filter2D(img2,0,k_passa_alta)
    img2 = (A-1)*img_g + img2


    cv2.imshow('Image', img2)
