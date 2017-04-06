import numpy as np
import cv2


def nothing():
    pass

def borde_filters(img,A ):
    w = 9*(A/100.0) -1
    k_passa_alta = A*np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    img2 = cv2.filter2D(img,0,k_passa_alta)
    return img2

def detec_boards(img, A):
    k_hlines = A*np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
    k_vlines = A*np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
    k_p45 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    k_m45 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    img2 = cv2.filter2D(img,0,k_hlines) + cv2.filter2D(img,0,k_vlines) + cv2.filter2D(img,0,k_p45) + cv2.filter2D(img,0,k_m45)

    return img2

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
img_b, img_g, img_r = cv2.split(img)
cv2.namedWindow('Image')

cv2.createTrackbar('A','Image',1,10,nothing)
cv2.createTrackbar('s','Image',0,255,nothing)

while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    A = cv2.getTrackbarPos('A','Image')
    if A == 0:
        A = 1
    s = cv2.getTrackbarPos('s','Image')

    img2 = img_g
    img2 = cv2.GaussianBlur(img2,(7,7),0)
    img2 = detec_boards(img2,A)

    #img2 = borde_filters(img, A)
    img2 = cv2.GaussianBlur(img2,(7,7),0)

    #img2[img2 > s] = 255

    cv2.imshow('Image', img2)
