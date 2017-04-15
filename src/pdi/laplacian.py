import cv2
import numpy as np


img = cv2.imread('21_training.tif')
cv2.namedWindow('Image')


sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

def nothing(x):
    pass

cv2.createTrackbar('s','Image',1,100,nothing)
#cv2.createTrackbar('sigma','Image',1,100,nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    img_b, img_g, img_r = cv2.split(img)

    img2 = img_g
    
    s = cv2.getTrackbarPos('s','Image')
    s = 2*s+1
    img2 = cv2.GaussianBlur(img2,(s,s),0)

    T = 1
    kernel = 7

    sobel = cv2.Sobel(img2,cv2.CV_64F,1,1,ksize=kernel)
    laplacian = cv2.Laplacian(img2,cv2.CV_64F)
    canny = cv2.Canny(img2,T,3*T)

    cv2.imshow('Sobel',sobel)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('Canny',canny)

cv2.destroyAllWindows()
