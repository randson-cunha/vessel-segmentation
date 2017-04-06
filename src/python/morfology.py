import cv2
import numpy as np
from matplotlib import pyplot as plt


def  nothing():
    pass


img = cv2.imread('21_training.tif',0)
cv2.namedWindow('Image')

cv2.createTrackbar('Canny','Image',1,100,nothing)
cv2.createTrackbar('k1','Image',1,10,nothing)
cv2.createTrackbar('k2','Image',1,10,nothing)
cv2.createTrackbar('k3','Image',1,100,nothing)

# aplica filtro gaussiano pra reduzir o ruido


while (1):

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    T = cv2.getTrackbarPos('Canny','Image')
    k1 = cv2.getTrackbarPos('k1','Image')
    k2 = cv2.getTrackbarPos('k2','Image')
    k3 = cv2.getTrackbarPos('k3','Image')
    A = cv2.getTrackbarPos('A','Image')

    if k1 == 0:
        k1 = 1

    if k2 == 0:
        K2 = 1

    if A == 0:
        A = 1

    img2 = img
    #img2 = cv2.Canny(img2,T,3*T)

    #kernel_dilate = np.ones((k1,k1),np.uint8)
    #img2 = cv2.dilate(img2,Kernel_dilate,iterations=1)

    #Kernel_erode = np.ones((k_erode,k_erode),np.uint8)
    #img2 = cv2.erode(img2,Kernel_erode,iterations= 1)
    #--


    img2 = cv2.medianBlur(img2,5)
    #img2 = cv2.Laplacian(img2,cv2.CV_64F)

    A = 4
    w = 9*(A/100.0) -1
    k_passa_alta = (1/9.0)*np.array([[-2,-2,-2],[-2,w,-2],[-2,-2,-2]])
    img3 = cv2.filter2D(img2,0,k_passa_alta)
    img3 = (A-1)*img + img3

    #cv2.imshow('Image_',img3)

    img2 = cv2.Canny(img2,T,3*T)
    Kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k1,k1))
    Kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k2,k2))
    img2 = cv2.dilate(img2,Kernel_dilate,iterations=k3)
    img2 = cv2.erode(img2,Kernel_erode, iterations= k3)


    #kernel = np.ones((k,k),np.uint8)
    #img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    #img2= cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Image", img2)


    #cv2.waitKey(0)
    #cv2.imwrite('canny2.tif',img)
cv2.destroyAllWindows()
